"""Round102 GitHub Release medium vector restoration workflow.

This module consumes operator-supplied GitHub Release assets for the medium 30k
fastText subset without committing any binary artifact to the repository.  It may
write only to an explicit temporary work directory, and it installs
``vectors.npy`` into the ignored seed path only when the reconstructed zip and
inner vector file pass the recorded checksum/shape/dtype audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from adapters.external_seed_manifest import FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH
from adapters.medium_vector_restoration import audit_operator_supplied_medium_vectors

ROUND102_RELEASE_RESTORE_VERSION = "v3_round102_medium_vector_release_restore"
RELEASE_OWNER = "MS0502"
RELEASE_REPO = "eve-project"
RELEASE_TAG = "eve-medium-30k-20260603"
PART_STEM = "subset_medium_30k-20260603T024008Z-3-001.zip"
WRAPPER_ASSETS: tuple[str, str] = (
    f"{PART_STEM}.part01.upload.zip",
    f"{PART_STEM}.part02.upload.zip",
)
RAW_PARTS: tuple[str, str] = (
    f"{PART_STEM}.part01",
    f"{PART_STEM}.part02",
)
SPLIT_MANIFEST_ASSET = "subset_medium_30k_split_manifest.json"
EXPECTED_RESTORED_ZIP_SHA256 = "ecfeabea37cf947a09c3d2d11f83f5ac9b7bc29cb026d32b18c4be616970b98f"
INTERNAL_VECTORS_PATH = "subset_medium_30k/vectors.npy"
GITHUB_RELEASE_DOWNLOAD_BASE = f"https://github.com/{RELEASE_OWNER}/{RELEASE_REPO}/releases/download/{RELEASE_TAG}"


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "eve-v3-round102-artifact-restore"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as out:
            shutil.copyfileobj(response, out)
    return {
        "url": url,
        "path": str(destination),
        "downloaded": True,
        "size_bytes": destination.stat().st_size if destination.is_file() else 0,
        "sha256": _sha256_file(destination),
    }


def _safe_zip_member(zip_file: zipfile.ZipFile, expected_name: str) -> str | None:
    normalized_expected = expected_name.replace("\\", "/")
    for name in zip_file.namelist():
        normalized = name.replace("\\", "/")
        if normalized == normalized_expected or normalized.endswith("/" + normalized_expected):
            return name
    return None


def _extract_wrapper_part(wrapper_path: Path, raw_part_name: str, output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "wrapper_path": str(wrapper_path),
        "raw_part_name": raw_part_name,
        "raw_part_path": str(output_dir / raw_part_name),
        "extracted": False,
        "error": None,
    }
    if not wrapper_path.is_file():
        result["error"] = "wrapper_missing"
        return result
    try:
        with zipfile.ZipFile(wrapper_path) as zf:
            member = _safe_zip_member(zf, raw_part_name)
            if member is None:
                result["error"] = "raw_part_missing_from_wrapper"
                return result
            target = output_dir / raw_part_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        result.update(
            {
                "extracted": True,
                "size_bytes": (output_dir / raw_part_name).stat().st_size,
                "sha256": _sha256_file(output_dir / raw_part_name),
            }
        )
        return result
    except zipfile.BadZipFile:
        result["error"] = "wrapper_bad_zip"
        return result


def _concatenate_parts(part_paths: Iterable[Path], restored_zip_path: Path) -> dict[str, Any]:
    restored_zip_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in part_paths if not path.is_file()]
    result: dict[str, Any] = {
        "restored_zip_path": str(restored_zip_path),
        "expected_sha256": EXPECTED_RESTORED_ZIP_SHA256,
        "actual_sha256": None,
        "sha256_match": False,
        "concatenated": False,
        "missing_parts": missing,
    }
    if missing:
        return result
    with restored_zip_path.open("wb") as out:
        for path in part_paths:
            with path.open("rb") as src:
                shutil.copyfileobj(src, out)
    actual = _sha256_file(restored_zip_path)
    result.update(
        {
            "actual_sha256": actual,
            "sha256_match": actual == EXPECTED_RESTORED_ZIP_SHA256,
            "concatenated": True,
            "size_bytes": restored_zip_path.stat().st_size,
        }
    )
    return result


def _test_and_extract_vectors(restored_zip_path: Path, vectors_out: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "restored_zip_path": str(restored_zip_path),
        "internal_vectors_path": INTERNAL_VECTORS_PATH,
        "zip_integrity_ok": False,
        "extracted_vectors_path": str(vectors_out),
        "vectors_extracted": False,
        "error": None,
    }
    if not restored_zip_path.is_file():
        result["error"] = "restored_zip_missing"
        return result
    try:
        with zipfile.ZipFile(restored_zip_path) as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                result["error"] = f"zip_integrity_failed:{bad_member}"
                return result
            result["zip_integrity_ok"] = True
            member = _safe_zip_member(zf, INTERNAL_VECTORS_PATH)
            if member is None:
                result["error"] = "internal_vectors_missing"
                return result
            vectors_out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, vectors_out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        result.update(
            {
                "vectors_extracted": True,
                "vectors_size_bytes": vectors_out.stat().st_size,
                "vectors_sha256": _sha256_file(vectors_out),
            }
        )
        return result
    except zipfile.BadZipFile:
        result["error"] = "restored_bad_zip"
        return result


def _copy_verified_vectors(candidate: Path, repo_root: Path) -> dict[str, Any]:
    target = repo_root / FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH / "vectors.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, target)
    return {
        "installed": True,
        "target_path": str(target),
        "target_sha256": _sha256_file(target),
        "git_add_allowed": False,
        "pr_diff_allowed": False,
    }


def restore_medium_vectors_from_release(
    *,
    work_dir: str | Path,
    repo_root: str | Path = ".",
    asset_dir: str | Path | None = None,
    download: bool = True,
    install_to_repo: bool = False,
) -> dict[str, Any]:
    """Download/reconstruct/audit the Round102 medium vectors release assets.

    When ``install_to_repo`` is true, the ignored ``vectors.npy`` target is
    copied only after all release and vector audits pass.  The function never
    stages files and never writes wrapper/raw/restored zip assets into the repo.
    """
    root = Path(repo_root)
    work = Path(work_dir)
    source_dir = Path(asset_dir) if asset_dir is not None else work
    work.mkdir(parents=True, exist_ok=True)
    downloads: list[dict[str, Any]] = []
    download_errors: list[dict[str, Any]] = []

    asset_names = [*WRAPPER_ASSETS, SPLIT_MANIFEST_ASSET]
    if download:
        for asset_name in asset_names:
            url = f"{GITHUB_RELEASE_DOWNLOAD_BASE}/{asset_name}"
            destination = work / asset_name
            try:
                downloads.append(_download(url, destination))
            except Exception as exc:  # data-only fail-closed network boundary
                download_errors.append(
                    {
                        "asset": asset_name,
                        "url": url,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    else:
        downloads = [
            {
                "asset": name,
                "path": str(source_dir / name),
                "downloaded": False,
                "exists": (source_dir / name).is_file(),
                "sha256": _sha256_file(source_dir / name),
            }
            for name in asset_names
        ]

    extraction_rows = []
    for wrapper_name, raw_name in zip(WRAPPER_ASSETS, RAW_PARTS, strict=True):
        wrapper_path = source_dir / wrapper_name if not download else work / wrapper_name
        extraction_rows.append(_extract_wrapper_part(wrapper_path, raw_name, work))

    restored_zip = work / PART_STEM
    concat = _concatenate_parts([work / name for name in RAW_PARTS], restored_zip)
    extracted_vectors = work / "subset_medium_30k" / "vectors.npy"
    zip_extract = _test_and_extract_vectors(restored_zip, extracted_vectors) if concat["sha256_match"] else {
        "restored_zip_path": str(restored_zip),
        "internal_vectors_path": INTERNAL_VECTORS_PATH,
        "zip_integrity_ok": False,
        "extracted_vectors_path": str(extracted_vectors),
        "vectors_extracted": False,
        "error": "restored_zip_sha256_mismatch_or_missing",
    }
    vectors_audit = audit_operator_supplied_medium_vectors(extracted_vectors, repo_root=root)
    release_ok = bool(
        not download_errors
        and concat["sha256_match"]
        and zip_extract["zip_integrity_ok"]
        and zip_extract["vectors_extracted"]
        and vectors_audit["acceptable_for_manual_install"]
    )
    install_result = None
    if install_to_repo and release_ok:
        install_result = _copy_verified_vectors(extracted_vectors, root)

    status = "medium_vector_artifact_available" if release_ok else "blocked_release_restore_incomplete"
    if install_result is not None:
        status = "medium_vector_artifact_installed_untracked"
    return {
        "version": ROUND102_RELEASE_RESTORE_VERSION,
        "round": 102,
        "status": status,
        "release": {
            "owner": RELEASE_OWNER,
            "repo": RELEASE_REPO,
            "tag": RELEASE_TAG,
            "assets": asset_names,
            "download_base": GITHUB_RELEASE_DOWNLOAD_BASE,
        },
        "policy": {
            "operator_supplied_artifact": True,
            "commit_wrapper_zip_to_pr": False,
            "commit_raw_parts_to_pr": False,
            "commit_restored_zip_to_pr": False,
            "commit_vectors_npy_to_pr": False,
            "create_dummy_vectors": False,
            "weaken_tests": False,
            "agp_bypass": False,
        },
        "paths": {
            "work_dir": str(work),
            "asset_dir": str(source_dir),
            "repo_root": str(root),
            "target_vectors_path": str(root / FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH / "vectors.npy"),
        },
        "downloads": downloads,
        "download_errors": download_errors,
        "wrapper_extractions": extraction_rows,
        "restored_zip": concat,
        "zip_extract": zip_extract,
        "vectors_audit": vectors_audit,
        "hard_stop_released": bool(release_ok),
        "install_result": install_result,
        "next_step": (
            "rerun_round97_98_validation" if release_ok else "manual_download_or_network_enabled_restore_required"
        ),
    }


def write_round102_release_restore_status(path: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    payload = deepcopy(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "version": ROUND102_RELEASE_RESTORE_VERSION,
        "path": str(out),
        "json_written": True,
        "binary_artifact_written_to_pr": False,
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round102 medium vector GitHub Release restore")
    parser.add_argument("--work-dir", required=True, help="temporary work directory outside the PR diff")
    parser.add_argument("--repo-root", default=".", help="repository root")
    parser.add_argument("--asset-dir", default=None, help="existing local release asset directory")
    parser.add_argument("--no-download", action="store_true", help="use --asset-dir or --work-dir assets without network download")
    parser.add_argument("--install-to-repo", action="store_true", help="copy verified vectors.npy to ignored seed path")
    parser.add_argument("--output", default=None, help="optional JSON output path")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    data = restore_medium_vectors_from_release(
        work_dir=args.work_dir,
        repo_root=args.repo_root,
        asset_dir=args.asset_dir,
        download=not args.no_download,
        install_to_repo=args.install_to_repo,
    )
    if args.output:
        write_round102_release_restore_status(args.output, data)
    print(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if data.get("hard_stop_released") else 2


__all__ = [
    "EXPECTED_RESTORED_ZIP_SHA256",
    "INTERNAL_VECTORS_PATH",
    "RAW_PARTS",
    "RELEASE_TAG",
    "ROUND102_RELEASE_RESTORE_VERSION",
    "SPLIT_MANIFEST_ASSET",
    "WRAPPER_ASSETS",
    "restore_medium_vectors_from_release",
    "write_round102_release_restore_status",
]


if __name__ == "__main__":
    raise SystemExit(main())
