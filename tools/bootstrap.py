#!/usr/bin/env python3
"""Fetch the assets the tutorial samples need.

Reads bootstrap.json and downloads each entry into assets/. Everything it fetches is
openly licensed and none of it is committed to this repository, so a fresh clone runs
this once before building.

    python3 tools/bootstrap.py            # fetch anything missing
    python3 tools/bootstrap.py --force    # re-fetch everything
    python3 tools/bootstrap.py --list     # show what would be fetched
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "bootstrap.json"
ASSET_DIR = ROOT / "assets"


def fetch_git(entry: dict, target: Path) -> None:
    cmd = ["git", "clone"]
    if entry.get("shallow", True):
        cmd += ["--depth", "1"]
    revision = entry.get("revision", "main")
    cmd += ["--branch", revision, entry["url"], str(target)]
    subprocess.run(cmd, check=True)


def fetch_archive(entry: dict, target: Path) -> None:
    url = entry["url"]
    target.mkdir(parents=True, exist_ok=True)
    archive = target.with_suffix(".download")

    request = urllib.request.Request(url, headers={"User-Agent": entry.get("user-agent", "learn-vulkan-bootstrap")})
    print(f"  downloading {url}")
    with urllib.request.urlopen(request) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)

    print(f"  extracting into {target.relative_to(ROOT)}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    else:
        shutil.unpack_archive(str(archive), str(target))
    archive.unlink()


FETCHERS = {"git": fetch_git, "archive": fetch_archive}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-fetch assets that are already present")
    parser.add_argument("--list", action="store_true", help="list assets and exit")
    parser.add_argument("--only", metavar="NAME", help="fetch just this one asset")
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"error: {MANIFEST} not found", file=sys.stderr)
        return 1

    entries = json.loads(MANIFEST.read_text())["assets"]
    if args.only:
        entries = [e for e in entries if e["name"] == args.only]
        if not entries:
            print(f"error: no asset named '{args.only}'", file=sys.stderr)
            return 1

    if args.list:
        for entry in entries:
            target = ASSET_DIR / entry["name"]
            state = "present" if target.exists() else "missing"
            print(f"{entry['name']:<28} {state:<8} {entry.get('description', '')}")
        return 0

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for entry in entries:
        name = entry["name"]
        target = ASSET_DIR / name

        if target.exists() and not args.force:
            print(f"{name}: already present, skipping")
            continue
        if target.exists():
            shutil.rmtree(target)

        fetcher = FETCHERS.get(entry["type"])
        if fetcher is None:
            print(f"{name}: unknown type '{entry['type']}'", file=sys.stderr)
            failures.append(name)
            continue

        print(f"{name}: fetching")
        try:
            fetcher(entry, target)
        except Exception as error:  # noqa: BLE001 - report and continue to the next asset
            print(f"{name}: FAILED ({error})", file=sys.stderr)
            shutil.rmtree(target, ignore_errors=True)
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} asset(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    print("\nAll assets present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
