# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Resolve a requirements file into its full transitive set and audit it.

Used by ``fm-serve-vuln-audit.yaml`` (Stage-2 transitive sweep) and by
``fm-serve-vuln-audit-postbuild.yaml`` so that both the pre-build agent and
the post-build scan share identical "resolve → audit → suggest bump" logic.

Behaviour
---------
1. Ask pip for a fully-resolved install report via ``pip install --dry-run
   --report`` (does not touch the environment).
2. Materialise the resolved tree as a flat ``name==version`` list.
3. Run ``pip-audit`` on that flat list to surface **transitive** advisories
   that a top-level-only audit would miss.
4. Classify advisories and produce a JSON summary with suggested fixes:
     - ``parent_bump``   — bump a direct dep whose resolved tree drops the
                           vulnerable transitive. Preferred over pinning.
     - ``transitive_pin``— explicitly pin the transitive in
                           ``requirements.txt`` with a "drop when" marker.
     - ``residual``      — no fix available / resolver conflict; surfaced in
                           the PR body so reviewers can decide.
5. Detect ``ResolutionImpossible`` from pip and emit a structured
   ``resolver_conflict`` block instead of crashing.

The script is intentionally side-effect free: it **reads** a requirements
file and **writes** a JSON summary. Applying the suggested bumps is the
caller's responsibility (the workflow) — keeps this helper easy to unit-test
and reusable.

Exit codes
----------
0   Audit completed (clean or with findings). Summary written.
2   Resolver conflict detected. Summary still written with
    ``resolver_conflict`` populated.
1   Unexpected error.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

SEVERE = {"CRITICAL", "HIGH"}


def _run(cmd: list[str], check: bool = False,
         capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Thin wrapper around subprocess.run that always decodes text."""
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def resolve_tree(requirements: pathlib.Path) -> tuple[list[str], str | None]:
    """Return ``(resolved_pins, conflict_block)``.

    ``resolved_pins`` is a sorted list of ``name==version`` for the full
    transitive closure as pip would install it. ``conflict_block`` is the
    ``ResolutionImpossible`` text when pip fails, else ``None``.
    """
    with tempfile.TemporaryDirectory() as td:
        report = pathlib.Path(td) / "report.json"
        proc = _run([
            sys.executable, "-m", "pip", "install",
            "--dry-run", "--ignore-installed", "--quiet",
            "--report", str(report),
            "-r", str(requirements),
        ])
        if proc.returncode != 0:
            # Surface the ResolutionImpossible block if present.
            m = re.search(
                r"ResolutionImpossible.*?(?=\n\n|\Z)",
                proc.stderr or "",
                re.DOTALL,
            )
            return [], (m.group(0) if m else proc.stderr.strip())

        if not report.exists() or not report.stat().st_size:
            return [], "pip did not produce an install report"

        data = json.loads(report.read_text())
        pins: list[str] = []
        for item in data.get("install", []) or []:
            meta = (item or {}).get("metadata") or {}
            name = meta.get("name")
            version = meta.get("version")
            if name and version:
                pins.append(f"{name}=={version}")
        return sorted(set(pins), key=str.lower), None


def audit_pins(pins: list[str]) -> list[dict[str, Any]]:
    """Run pip-audit on a flat ``name==version`` list and flatten findings."""
    if not pins:
        return []
    if shutil.which("pip-audit") is None:
        raise RuntimeError("pip-audit is not installed on PATH")
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "resolved.txt"
        tmp.write_text("\n".join(pins) + "\n")
        out = pathlib.Path(td) / "audit.json"
        # --disable-pip: we already have a fully-resolved set; don't re-resolve.
        _run([
            "pip-audit", "-r", str(tmp),
            "--format=json", "--disable-pip",
            "--output", str(out),
        ])
        if not out.exists() or not out.stat().st_size:
            return []
        doc = json.loads(out.read_text())

    findings: list[dict[str, Any]] = []
    for dep in doc.get("dependencies", []) or []:
        for v in dep.get("vulns", []) or []:
            sev = (v.get("severity") or "").upper() or "UNKNOWN"
            findings.append({
                "name": dep.get("name"),
                "version": dep.get("version"),
                "id": v.get("id"),
                "severity": sev,
                "fix_versions": v.get("fix_versions") or [],
                "description": (v.get("description") or "")
                    .strip().splitlines()[0][:200],
            })
    return findings


def classify(findings: list[dict[str, Any]],
             direct_names: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Split findings into parent_bump / transitive_pin / residual buckets.

    A finding is *transitive* when its package name is not among the direct
    requirements. For transitive findings we emit a ``transitive_pin``
    suggestion by default; callers that can compute a parent upgrade should
    promote that suggestion to ``parent_bump``. Findings with no
    ``fix_versions`` become ``residual`` with reason ``no-fix-available``.
    """
    parent_bump: list[dict[str, Any]] = []
    transitive_pin: list[dict[str, Any]] = []
    residual: list[dict[str, Any]] = []

    for f in findings:
        name_lc = (f["name"] or "").lower()
        is_direct = name_lc in direct_names
        if not f["fix_versions"]:
            residual.append({**f, "reason": "no-fix-available"})
            continue
        if is_direct:
            # Caller already patches direct pins in Stage 1; leave a marker
            # so the workflow can confirm it was picked up.
            parent_bump.append({
                **f,
                "suggested_version": min(f["fix_versions"]),
                "kind": "direct",
            })
        else:
            transitive_pin.append({
                **f,
                "suggested_pin": f"{f['name']}=={min(f['fix_versions'])}",
                "drop_when": (
                    "drop when the parent direct dependency's resolved "
                    "tree includes "
                    f"{f['name']}>={min(f['fix_versions'])}"
                ),
            })
    return {
        "parent_bump": parent_bump,
        "transitive_pin": transitive_pin,
        "residual": residual,
    }


def direct_names_from(requirements: pathlib.Path) -> set[str]:
    """Extract direct-requirement package names (lowercased, normalised)."""
    names: set[str] = set()
    for raw in requirements.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)", line)
        if m:
            names.add(m.group(1).lower().replace("_", "-"))
    return names


def build_summary(requirements: pathlib.Path) -> tuple[dict[str, Any], int]:
    """Return ``(summary_dict, exit_code)``."""
    pins, conflict = resolve_tree(requirements)
    if conflict is not None:
        return (
            {
                "requirements": str(requirements),
                "resolver_conflict": conflict,
                "resolved_pins": [],
                "findings": [],
                "suggestions": {
                    "parent_bump": [],
                    "transitive_pin": [],
                    "residual": [],
                },
                "severe_unresolved": 0,
            },
            2,
        )

    findings = audit_pins(pins)
    directs = direct_names_from(requirements)
    suggestions = classify(findings, directs)

    severe_unresolved = sum(
        1 for f in suggestions["residual"]
        if (f.get("severity") or "").upper() in SEVERE
    )
    return (
        {
            "requirements": str(requirements),
            "resolver_conflict": None,
            "resolved_pins": pins,
            "findings": findings,
            "suggestions": suggestions,
            "severe_unresolved": severe_unresolved,
        },
        0,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run resolve+audit, write the summary JSON, return rc."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--requirements", required=True,
                    help="Path to the requirements.txt file to resolve+audit.")
    ap.add_argument("--output", required=True,
                    help="Path to write the JSON summary to.")
    args = ap.parse_args(argv)

    req = pathlib.Path(args.requirements)
    if not req.exists():
        print(f"requirements file not found: {req}", file=sys.stderr)
        return 1

    summary, rc = build_summary(req)
    pathlib.Path(args.output).write_text(json.dumps(summary, indent=2))
    # Brief stdout summary for the workflow log.
    print(
        f"resolved={len(summary['resolved_pins'])} "
        f"findings={len(summary['findings'])} "
        f"parent_bump={len(summary['suggestions']['parent_bump'])} "
        f"transitive_pin={len(summary['suggestions']['transitive_pin'])} "
        f"residual={len(summary['suggestions']['residual'])} "
        f"severe_unresolved={summary['severe_unresolved']} "
        f"resolver_conflict={'yes' if summary['resolver_conflict'] else 'no'}"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
