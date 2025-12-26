#!/usr/bin/env python3
"""
Single/Ad‑hoc Image Scanner.

Given one or more fully qualified image names (FQIN) e.g.
  myregistry.azurecr.io/public/azureml/base:1.0.0
this script will:
  1. Ensure an SBOM artifact is attached (runs ACR task if missing)
  2. Download the SBOM
  3. Run a CoMET vulnerability scan
  4. Evaluate the scan against policy (with optional QID ignores)
  5. Fail (exit code 1) if any image is non‑compliant or any operational step fails

Differences vs batch_image_scan.py:
  - Accepts explicit FQINs instead of inventory
  - Automatically attaches SBOM if not present (unless already there)
  - Optional --ignore-qid allows ignoring specific vulnerability QIDs

Examples:
  python scan_images.py myregistry.azurecr.io/public/azureml/base:1.0.0
  python scan_images.py --ignore-qid 123456 789012 --auth interactive mcr.microsoft.com/public/azureml/base:2.0.0
  python scan_images.py --config custom.yaml --output ./scan-out myreg.azurecr.io/ns/image:tag other.azurecr.io/x/y:latest
"""

import argparse
import os
import sys
import json
import tempfile
from typing import List, Tuple, Dict, Optional
import concurrent.futures

from vienna_container_management.settings import Settings
from vienna_container_management.core.container_builder import ContainerBuilder
from vienna_container_management.core.acr_helper import ACRHelper
from vienna_container_management.core.vulnerability_scanner import VulnerabilityScanner
from vienna_container_management.core.auth import AuthenticationManager


class ImageScanResult:
    """Container for the results of scanning a single container image.

    Captures SBOM attach outcome, vulnerability evaluation details, and any
    operational errors so callers can later aggregate status across many
    parallel scans.
    """

    def __init__(self, image: str):
        """Initialize a new image scan result record.

        Parameters
        ----------
        image : str
            Fully qualified image name (registry/repository:tag).
        """
        self.image = image
        self.attach_success: Optional[bool] = None
        self.attach_message: str = ""
        self.sbom_path: Optional[str] = None
        self.scan_evaluation: Optional[Dict] = None
        self.success: bool = False
        self.error: Optional[str] = None
        self.vulnerability_summary: Optional[Dict] = None

    def to_dict(self):
        """Convert this result to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary containing scan status, evaluation metadata, and
            derived vulnerability breakdown if available.
        """
        result = {
            "image": self.image,
            "attach_success": self.attach_success,
            "attach_message": self.attach_message,
            "sbom_path": self.sbom_path,
            "evaluation": self.scan_evaluation,
            "success": self.success,
            "error": self.error,
        }

        # Add detailed vulnerability breakdown if available
        if self.scan_evaluation:
            result["vulnerability_count"] = self.scan_evaluation.get('total', 0)
            result["compliance_status"] = self.scan_evaluation.get('status', 'unknown')

            # Add risk breakdown
            if self.scan_evaluation.get('summary'):
                summary = self.scan_evaluation['summary']
                risk_counts = {}
                vulnerabilities_by_risk = {}

                for qid, vuln_info in summary.items():
                    risk = vuln_info.get('risk', 'UNKNOWN')
                    risk_counts[risk] = risk_counts.get(risk, 0) + 1

                    if risk not in vulnerabilities_by_risk:
                        vulnerabilities_by_risk[risk] = []
                    vulnerabilities_by_risk[risk].append({
                        'qid': qid,
                        'risk': risk,
                        'due_date': vuln_info.get('dueDate', 'Unknown')
                    })

                result["risk_breakdown"] = risk_counts
                result["vulnerabilities_by_risk"] = vulnerabilities_by_risk

        return result


def parse_fqin(fqin: str) -> Tuple[str, str, str]:
    """Parse fully qualified image name into (registry, repository, tag).

    Accepts forms:
      registry/repository:tag
      registry/repository (defaults tag=latest)
    """
    if '/' not in fqin:
        raise ValueError(f"Invalid image '{fqin}' – must include registry and repository")
    registry, remainder = fqin.split('/', 1)
    # tag
    if ':' in remainder:
        repository, tag = remainder.rsplit(':', 1)
    else:
        repository, tag = remainder, 'latest'
    if not registry or not repository:
        raise ValueError(f"Invalid image '{fqin}'")
    return registry, repository, tag


def ensure_dir(path: str):
    """Create directory if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path to create (including parents as needed).
    """
    os.makedirs(path, exist_ok=True)


def scan_image(fqin: str, settings: Settings, auth_method: str, ignore_qids: Optional[List[str]], output_dir: str) -> ImageScanResult:
    """Execute the full scan workflow for a single image.

    Steps performed:
      1. Attach SBOM artifact if missing (non-fatal if already present).
      2. Download SBOM from registry.
      3. Acquire CoMET token using the selected auth method.
      4. Evaluate vulnerabilities with optional QID ignore list.

    Parameters
    ----------
    fqin : str
        Fully qualified image name (registry/repository[:tag]).
    settings : Settings
        Configuration object for scanning utilities.
    auth_method : str
        Authentication method for CoMET ('default' or 'interactive').
    ignore_qids : list[str] | None
        Optional list of vulnerability QIDs to ignore during evaluation.
    output_dir : str
        Directory where per-image artifacts (SBOM, logs) are stored.

    Returns
    -------
    ImageScanResult
        Populated result including compliance status and any error message.
    """
    result = ImageScanResult(fqin)
    try:
        registry, repository, tag = parse_fqin(fqin)
        image_safe_name = fqin.replace('/', '_').replace(':', '_')
        image_dir = os.path.join(output_dir, image_safe_name)
        ensure_dir(image_dir)

        # 1. Attach SBOM if missing
        builder = ContainerBuilder(settings)
        attach_res = builder.attach_sbom(
            repository=repository,
            tag=tag,
            registry=registry,
            working_directory=image_dir,
            dry_run=False,
            force=False
        )
        result.attach_success = attach_res.get('success', False)
        result.attach_message = attach_res.get('message', '')
        # If we skipped because SBOM exists, that's fine; treat as success for continuing
        if attach_res.get('skip_reason') == 'sbom_exists':
            result.attach_success = True

        if not result.attach_success:
            result.error = f"SBOM attach failed: {result.attach_message}"
            return result

        # 2. Download SBOM
        # If registry lacks a dot (e.g. short ACR name), append suffix for ACRHelper expectations
        download_registry = registry if '.' in registry else f"{registry}.azurecr.io"
        sbom_downloader = ACRHelper(download_registry, repository, tag)
        if not sbom_downloader.authenticate():
            result.error = "Failed to authenticate with registry"
            return result

        sbom_path = os.path.join(image_dir, 'sbom.json')
        if not sbom_downloader.download_sbom(sbom_path):
            result.error = "Failed to download SBOM"
            return result
        result.sbom_path = sbom_path

        # 3. Get CoMET token
        auth_mgr = AuthenticationManager(settings)
        try:
            if auth_method == 'interactive':
                comet_token = auth_mgr.get_comet_token_interactive()
            else:
                comet_token = auth_mgr.get_comet_token_default()
        except Exception as e:
            result.error = f"Failed to obtain CoMET token: {e}"
            return result

        # 4. Evaluate
        vuln_scanner = VulnerabilityScanner(settings)
        evaluation = vuln_scanner.evaluate_sbom_file(
            sbom_path=sbom_path,
            comet_token=comet_token,
            verbose=False,
            ignore_qid=ignore_qids
        )
        result.scan_evaluation = evaluation
        status = evaluation.get('status')
        result.success = status == 'compliant'
        if not result.success:
            result.error = f"Non-compliant: {evaluation.get('total', 0)} vulnerabilities"

    except Exception as e:
        result.error = str(e)
    return result


def print_summary(results: List[ImageScanResult], output_dir: str):
    """Print a human-readable summary and persist JSON results file.

    Parameters
    ----------
    results : list[ImageScanResult]
        Collection of per-image scan results.
    output_dir : str
        Directory where the consolidated JSON report will be written.
    """
    compliant = [r for r in results if r.success]
    non_compliant = [r for r in results if r.scan_evaluation and not r.success]
    failed = [r for r in results if r.error and r.scan_evaluation is None]

    print("\n" + '=' * 70)
    print("IMAGE VULNERABILITY SCAN SUMMARY")
    print('=' * 70)
    print(f"Total images: {len(results)}")
    print(f"Compliant: {len(compliant)}")
    print(f"Non-compliant: {len(non_compliant)}")
    print(f"Failed (operational): {len(failed)}")

    # Calculate overall vulnerability statistics
    if non_compliant:
        total_vulns = sum(r.scan_evaluation.get('total', 0) for r in non_compliant if r.scan_evaluation)
        overall_risk_counts = {}

        for r in non_compliant:
            if r.scan_evaluation and r.scan_evaluation.get('summary'):
                summary = r.scan_evaluation['summary']
                for qid, vuln_info in summary.items():
                    risk = vuln_info.get('risk', 'UNKNOWN')
                    overall_risk_counts[risk] = overall_risk_counts.get(risk, 0) + 1

        print("\nOverall vulnerability statistics:")
        print(f"Total vulnerabilities: {total_vulns}")
        if overall_risk_counts:
            for risk, count in sorted(overall_risk_counts.items()):
                print(f"  {risk}: {count}")

    if compliant:
        print("\n✅ Compliant:")
        for r in compliant:
            print(f"  - {r.image}")
    if non_compliant:
        print("\n⚠️  Non-compliant:")
        for r in non_compliant:
            total = r.scan_evaluation.get('total', 0) if r.scan_evaluation else 'N/A'
            print(f"  - {r.image}: {total} vulnerabilities")

            # Show detailed vulnerability breakdown
            if r.scan_evaluation and r.scan_evaluation.get('summary'):
                summary = r.scan_evaluation['summary']
                risk_counts = {}
                for qid, vuln_info in summary.items():
                    risk = vuln_info.get('risk', 'UNKNOWN')
                    risk_counts[risk] = risk_counts.get(risk, 0) + 1

                risk_breakdown = ', '.join([f"{count} {risk}" for risk, count in sorted(risk_counts.items())])
                print(f"    Breakdown: {risk_breakdown}")

                # Show vulnerabilities with details
                if len(summary) > 0:
                    print("    Top vulnerabilities:")
                    vuln_items = list(summary.items())
                    for qid, vuln_info in vuln_items:
                        risk = vuln_info.get('risk', 'UNKNOWN')
                        due_date = vuln_info.get('dueDate', 'Unknown')
                        print(f"      QID {qid}: {risk} (Due: {due_date})")
                    details = r.scan_evaluation.get("non_compliant_vulnerabilities", None)
                    if details:
                        print("    Detailed vulnerability info:")
                        for vuln in details:
                            print(f"- QID {vuln.get('id')}:\n{vuln}")

    if failed:
        print("\n❌ Failed:")
        for r in failed:
            print(f"  - {r.image}: {r.error}")

    status_file = os.path.join(output_dir, 'scan_results.json')
    try:
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump({r.image: r.to_dict() for r in results}, f, indent=2)
        print(f"\n📄 Detailed results written to {status_file}")
    except Exception as e:
        print(f"⚠️  Failed to write results file: {e}")


def main():
    """Entry point for CLI argument parsing and parallel scan execution.

    Returns
    -------
    int
        Process exit code (0 = all compliant, 1 = any non-compliance or failure).
    """
    parser = argparse.ArgumentParser(
        description="Scan one or more fully qualified container images (attach SBOM if needed, download, scan, evaluate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('images', nargs='+', help='Fully qualified image names (registry/repository[:tag])')
    parser.add_argument('--ignore-qid', nargs='*', help='QIDs to ignore during vulnerability evaluation (space-separated list)')
    parser.add_argument('--auth', choices=['default', 'interactive'], default='default', help='Authentication method for CoMET (default: default)')
    parser.add_argument('--config', help='Optional settings override YAML')
    parser.add_argument('--output', help='Output directory (default: temp)')
    parser.add_argument('--workers', type=int, default=4, help='Maximum parallel workers (default: 4)')

    args = parser.parse_args()

    output_dir = args.output or tempfile.mkdtemp(prefix='scan_image_')
    ensure_dir(output_dir)
    print(f"📂 Output directory: {output_dir}")
    print(f"🚀 Scanning {len(args.images)} image(s) with up to {args.workers} parallel worker(s)")

    # Load settings (optionally override)
    try:
        settings = Settings(config_path=args.config) if args.config else Settings()
        # Override ignore_qid in settings if provided
        if getattr(args, 'ignore_qid', None):
            settings.set('evaluation.ignore_qid', args.ignore_qid)
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return 1

    results: List[ImageScanResult] = []

    # Use ThreadPoolExecutor for parallel scans
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_image = {}
        for img in args.images:
            print(f"Submitting {img} ...")
            future = executor.submit(
                scan_image,
                fqin=img,
                settings=settings,
                auth_method=args.auth,
                ignore_qids=getattr(args, 'ignore_qid', None),
                output_dir=output_dir
            )
            future_to_image[future] = img

        for future in concurrent.futures.as_completed(future_to_image):
            img = future_to_image[future]
            try:
                res = future.result()
            except Exception as e:
                res = ImageScanResult(img)
                res.error = f"Unhandled exception: {e}"
            # Log minimal per-image status
            header = f"=== {img} ==="
            print(f"\n{header}")
            if res.error and not res.scan_evaluation:
                print(f"❌ Failure: {res.error}")
            elif res.scan_evaluation and not res.success:
                total = res.scan_evaluation.get('total', 0)
                print(f"⚠️  Non-compliant: {total} vulnerabilities")

                # Show immediate vulnerability breakdown
                if res.scan_evaluation.get('summary'):
                    summary = res.scan_evaluation['summary']
                    risk_counts = {}
                    for qid, vuln_info in summary.items():
                        risk = vuln_info.get('risk', 'UNKNOWN')
                        risk_counts[risk] = risk_counts.get(risk, 0) + 1

                    if risk_counts:
                        risk_breakdown = ', '.join([f"{count} {risk}" for risk, count in sorted(risk_counts.items())])
                        print(f"   Risk breakdown: {risk_breakdown}")
            else:
                print("✅ Compliant")
            results.append(res)

    # Preserve original input order in final summary
    results.sort(key=lambda r: args.images.index(r.image))
    print_summary(results, output_dir)

    # Exit code logic: any operational failure OR any non-compliant evaluation => 1
    any_failure = any((not r.success) for r in results)
    return 1 if any_failure else 0


if __name__ == '__main__':
    sys.exit(main())
