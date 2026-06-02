#!/usr/bin/env python3
"""Restore and verify the Round96 source package.

Preferred input is the code-only Round96 zip plus its manifest. For backward
compatibility, the older split zip parts are also accepted. The code-only
package intentionally excludes the medium fastText vector file:

    seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

CODE_ONLY_ZIP = "eve_v3_round96_code_only_no_medium_vectors.zip"
CODE_ONLY_MANIFEST = "eve_v3_round96_code_only_manifest.json"
SPLIT_PACKAGE_STEM = "eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip"
SPLIT_PART_CANDIDATES = (
    (f"{SPLIT_PACKAGE_STEM}.part01", "part01"),
    (f"{SPLIT_PACKAGE_STEM}.part02", "part02"),
)
SPLIT_MANIFEST_CANDIDATES = ("eve_v3_round96_split_manifest.json", "manifest")
EXTRACT_DIR = "round96_source"
MEDIUM_VECTOR_RELATIVE_PATH = "seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy"
SHA256_KEYS = ("source_sha256", "zip_sha256", "sha256")


@dataclass(frozen=True)
class PackageInput:
    mode: str
    zip_path: Path
    manifest_path: Path
    part_paths: tuple[Path, ...] = ()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_sha256(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in SHA256_KEYS:
        expected = manifest.get(key)
        if isinstance(expected, str) and expected:
            return expected.lower()
    accepted = ", ".join(SHA256_KEYS)
    raise ValueError(f"{manifest_path.name} must contain one of these SHA-256 fields: {accepted}")


def first_existing(package_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = package_dir / name
        if path.exists():
            return path
    return None


def resolve_code_only_input(package_dir: Path) -> PackageInput | None:
    zip_path = package_dir / CODE_ONLY_ZIP
    manifest_path = package_dir / CODE_ONLY_MANIFEST
    if zip_path.exists() and manifest_path.exists():
        return PackageInput(mode="code_only", zip_path=zip_path, manifest_path=manifest_path)
    return None


def resolve_split_input(package_dir: Path) -> PackageInput | None:
    part_paths: list[Path] = []
    for candidates in SPLIT_PART_CANDIDATES:
        path = first_existing(package_dir, candidates)
        if path is None:
            return None
        part_paths.append(path)

    manifest_path = first_existing(package_dir, SPLIT_MANIFEST_CANDIDATES)
    if manifest_path is None:
        return None
    return PackageInput(
        mode="split",
        zip_path=package_dir / SPLIT_PACKAGE_STEM,
        manifest_path=manifest_path,
        part_paths=tuple(part_paths),
    )


def describe_missing_inputs(package_dir: Path) -> str:
    code_only_missing = [
        name
        for name in (CODE_ONLY_ZIP, CODE_ONLY_MANIFEST)
        if not (package_dir / name).exists()
    ]
    split_missing: list[str] = []
    for candidates in SPLIT_PART_CANDIDATES:
        if first_existing(package_dir, candidates) is None:
            split_missing.append(" or ".join(candidates))
    if first_existing(package_dir, SPLIT_MANIFEST_CANDIDATES) is None:
        split_missing.append(" or ".join(SPLIT_MANIFEST_CANDIDATES))

    code_only_list = "\n".join(f"- {name}" for name in code_only_missing) or "- none"
    split_list = "\n".join(f"- {name}" for name in split_missing) or "- none"
    return (
        "Round96 package inputs are missing. Provide either the preferred code-only package:\n"
        f"{code_only_list}\n\n"
        "or the legacy split package inputs:\n"
        f"{split_list}\n\n"
        "Upload them into eve_v3_autonomous_handoff/packages/ and rerun this script."
    )


def resolve_package_input(package_dir: Path) -> PackageInput:
    code_only = resolve_code_only_input(package_dir)
    if code_only is not None:
        return code_only
    split = resolve_split_input(package_dir)
    if split is not None:
        return split
    raise FileNotFoundError(describe_missing_inputs(package_dir))


def restore_split_zip(part_paths: tuple[Path, ...], output_path: Path) -> None:
    with output_path.open("wb") as output:
        for part_path in part_paths:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, output, length=1024 * 1024)


def verify_zip_integrity(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Corrupt zip member detected: {bad_member}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        extract_dir.mkdir(parents=True, exist_ok=True)
        archive.extractall(extract_dir)


def medium_vector_present(extract_dir: Path) -> bool:
    return (extract_dir / MEDIUM_VECTOR_RELATIVE_PATH).exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore and verify the Round96 source package.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify SHA-256 and zip integrity but do not extract.",
    )
    args = parser.parse_args()

    package_dir = Path(__file__).resolve().parent
    extract_dir = package_dir / EXTRACT_DIR

    try:
        package_input = resolve_package_input(package_dir)
        print(f"Using Round96 package mode: {package_input.mode}")
        print(f"- zip: {package_input.zip_path.name}")
        print(f"- manifest: {package_input.manifest_path.name}")
        if package_input.mode == "split":
            for part_path in package_input.part_paths:
                print(f"- part: {part_path.name}")
            restore_split_zip(package_input.part_paths, package_input.zip_path)
            print(f"Restored split zip: {package_input.zip_path.name}")

        expected_sha256 = load_expected_sha256(package_input.manifest_path)
        actual_sha256 = sha256_file(package_input.zip_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Round96 zip SHA-256 mismatch:\n"
                f"expected: {expected_sha256}\n"
                f"actual:   {actual_sha256}"
            )
        print(f"SHA-256 verified: {actual_sha256}")

        verify_zip_integrity(package_input.zip_path)
        print("Zip integrity verified")

        if args.verify_only:
            return 0

        extract_zip(package_input.zip_path, extract_dir)
        print(f"Extracted Round96 source to: {extract_dir}")
        if package_input.mode == "code_only" and not medium_vector_present(extract_dir):
            print(f"Code-only package note: omitted {MEDIUM_VECTOR_RELATIVE_PATH}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should report any restore blocker clearly.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
