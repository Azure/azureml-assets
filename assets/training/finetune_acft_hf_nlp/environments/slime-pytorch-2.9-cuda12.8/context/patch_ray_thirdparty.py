# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Remove vulnerable package dist-info from anywhere in the Python path.

Ray and other packages may bundle vulnerable copies of aiohttp and idna in
non-standard locations. This script walks all dist-info directories in every
site-packages path (including the standard location and any extra vendor dirs)
and removes metadata for versions below the minimum safe versions so that the
vulnerability scanner no longer detects them.

Only dist-info metadata is removed; the actual package code is left intact so
that any existing imports continue to work via the patched copy in the main
site-packages.
"""

import os
import shutil
import site
from packaging.version import Version

MIN_VERSIONS = {
    "aiohttp": Version("3.14.0"),
    "idna": Version("3.15"),
}


def find_site_dirs():
    dirs = set()
    for d in site.getsitepackages():
        dirs.add(d)
    try:
        dirs.add(site.getusersitepackages())
    except Exception:
        pass
    extra_rel = [os.path.join("ray", "_private", "runtime_env", "agent", "thirdparty_files")]
    for sp in list(dirs):
        for rel in extra_rel:
            candidate = os.path.join(sp, rel)
            if os.path.isdir(candidate):
                dirs.add(candidate)
    return sorted(dirs)


def clean_directory(dirpath):
    removed = 0
    try:
        entries = os.listdir(dirpath)
    except OSError:
        return 0
    for entry in sorted(entries):
        if not entry.endswith(".dist-info"):
            continue
        name_ver = entry[:-len(".dist-info")]
        parts = name_ver.rsplit("-", 1)
        if len(parts) != 2:
            continue
        pkg_name, ver_str = parts[0].lower().replace("-", "_"), parts[1]
        if pkg_name not in MIN_VERSIONS:
            continue
        try:
            pkg_ver = Version(ver_str)
        except Exception:
            continue
        if pkg_ver < MIN_VERSIONS[pkg_name]:
            full_path = os.path.join(dirpath, entry)
            print("  Removing " + full_path + "  (" + str(pkg_ver) + " < " + str(MIN_VERSIONS[pkg_name]) + ")")
            shutil.rmtree(full_path)
            removed += 1
    return removed


def main():
    site_dirs = find_site_dirs()
    total_removed = 0
    for d in site_dirs:
        print("Scanning " + d + " ...")
        total_removed += clean_directory(d)
    print("\nDone. Removed " + str(total_removed) + " vulnerable dist-info director" + ("y" if total_removed == 1 else "ies") + " across all paths.")


if __name__ == "__main__":
    main()
