"""Download and verify local model assets declared by Module C configs."""

from __future__ import annotations

import json
from hashlib import md5, sha256
from pathlib import Path, PurePosixPath
import shutil
import struct
from typing import Any
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
import zlib


UMOF_FIGSHARE_ARTICLE_ID = "33311829"
UMOF_DEFAULT_DIRECTORY = Path("external/models/umof")
UMOF_DEFAULT_MODELS = {
    "umof_mh": {
        "filename": "umof-mh.model",
        "archive_member": "umof_models/uMOF-MH.model",
        "patterns": ["umof-mh", "umof_mh", "umofmh"],
    },
    "umof_polar": {
        "filename": "umof-polar.model",
        "archive_member": "umof_models/uMOF-POLAR.model",
        "patterns": ["umof-polar", "umof_polar", "umofpolar"],
    },
}


def ensure_model_asset(
    specification: dict[str, Any],
    *,
    download_missing: bool = False,
) -> dict[str, Any]:
    """Ensure a configured local model exists and return its provenance."""
    model_value = specification.get("model")
    if not model_value:
        raise ValueError("The model specification requires a local model path.")
    model_path = Path(str(model_value)).expanduser()
    expected_sha256 = specification.get("sha256")
    if model_path.is_file():
        digest = _file_digest(model_path, "sha256")
        if expected_sha256 and digest.casefold() != str(expected_sha256).casefold():
            raise ValueError(
                f"Model checksum mismatch for {model_path}: expected "
                f"{expected_sha256}, found {digest}."
            )
        return {
            "status": "ready",
            "path": str(model_path),
            "sha256": digest,
            "downloaded": False,
        }

    asset = specification.get("asset")
    if not isinstance(asset, dict):
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}. No downloadable asset "
            "is configured."
        )
    if not download_missing:
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}. Run the benchmark with "
            "--setup-mlip-mc, --setup-golddac, or the corresponding "
            "--install-missing workflow to download the configured model asset."
        )

    asset = _resolve_figshare_asset(asset, model_path)
    direct_url = str(asset.get("url", "")).strip()
    if direct_url:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = model_path.with_suffix(model_path.suffix + ".part")
        try:
            _download(direct_url, temporary_path)
            temporary_path.replace(model_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return _finalize_model_asset(
            model_path,
            expected_sha256=expected_sha256,
            source_url=direct_url,
            archive_member=model_path.name,
            transfer_mode="direct_download",
        )

    archive_url = str(asset.get("archive_url", "")).strip()
    member_name = str(
        asset.get("member_path")
        or asset.get("member_basename", model_path.name)
    ).strip()
    if not archive_url or not member_name:
        raise ValueError(
            "A downloadable model asset requires archive_url and "
            "member_basename."
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".part")
    ranged_member = None
    if archive_url.startswith(("https://", "http://")):
        ranged_member = _extract_remote_zip_member(
            archive_url,
            member_name,
            temporary_path,
        )
    if ranged_member is not None:
        temporary_path.replace(model_path)
        return _finalize_model_asset(
            model_path,
            expected_sha256=expected_sha256,
            source_url=archive_url,
            archive_member=ranged_member,
            transfer_mode="http_range",
        )

    cache_directory = model_path.parent / ".downloads"
    cache_directory.mkdir(parents=True, exist_ok=True)
    archive_name = str(asset.get("archive_filename", "model_archive.zip"))
    archive_path = cache_directory / Path(archive_name).name
    if not archive_path.is_file():
        _download(archive_url, archive_path)

    expected_md5 = str(asset.get("archive_md5", "")).removeprefix("md5:")
    if expected_md5:
        actual_md5 = _file_digest(archive_path, "md5")
        if actual_md5.casefold() != expected_md5.casefold():
            raise ValueError(
                f"Archive checksum mismatch for {archive_path}: expected "
                f"{expected_md5}, found {actual_md5}."
            )
    member = _find_archive_member(archive_path, member_name)
    try:
        with ZipFile(archive_path) as archive, archive.open(member) as source:
            with temporary_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    if not bool(asset.get("keep_archive", False)):
        archive_path.unlink(missing_ok=True)
    return _finalize_model_asset(
        model_path,
        expected_sha256=expected_sha256,
        source_url=archive_url,
        archive_member=member,
        transfer_mode="full_archive",
    )


def ensure_umof_model_asset(
    specification: dict[str, Any],
    *,
    download_missing: bool = False,
) -> dict[str, Any]:
    """Ensure a uMOF-MH/uMOF-POLAR checkpoint is available locally.

    The uMOF paper release is distributed through Figshare. Config files may
    either provide an explicit downloadable ``asset`` block or only identify the
    backend as ``umof_mh``/``umof_polar``. In the latter case, this function
    fills in the standard local path and the Figshare lookup metadata.
    """
    if not is_umof_specification(specification):
        return ensure_model_asset(
            specification,
            download_missing=download_missing,
        )
    model_key = _umof_model_key(specification)
    if model_key not in UMOF_DEFAULT_MODELS:
        raise ValueError(
            "uMOF model specifications must be named 'umof_mh' or "
            "'umof_polar'."
        )
    defaults = UMOF_DEFAULT_MODELS[model_key]
    specification.setdefault(
        "model",
        str(UMOF_DEFAULT_DIRECTORY / defaults["filename"]),
    )
    specification.setdefault(
        "asset",
        {
            "figshare_article_id": UMOF_FIGSHARE_ARTICLE_ID,
            "file_name_patterns": defaults["patterns"],
            "candidate_extensions": [".model", ".pt", ".pth"],
        },
    )
    asset = specification["asset"]
    if isinstance(asset, dict) and asset.get("figshare_article_id") == UMOF_FIGSHARE_ARTICLE_ID:
        asset.setdefault("file_name", "umof_models.zip")
        if not asset.get("member_path") and not asset.get("member_basename"):
            asset["member_path"] = defaults["archive_member"]
    return ensure_model_asset(
        specification,
        download_missing=download_missing,
    )


def is_umof_specification(specification: dict[str, Any]) -> bool:
    """Return True when a model specification describes a uMOF checkpoint."""
    name = str(specification.get("name", "")).strip().casefold().replace("-", "_")
    backend = str(
        specification.get("backend", specification.get("type", ""))
    ).strip().casefold().replace("-", "_")
    family = str(
        specification.get("model_family", "")
    ).strip().casefold().replace("-", "_")
    return (
        name in {"umof_mh", "umof_polar"}
        or backend in {"umof", "umof_mh", "umof_polar"}
        or family == "umof"
    )


def _umof_model_key(specification: dict[str, Any]) -> str:
    for key in ("name", "backend", "type"):
        value = str(specification.get(key, "")).strip().casefold()
        normalized = value.replace("-", "_")
        if normalized in {"umof_mh", "umof_polar"}:
            return normalized
    model = str(specification.get("model", "")).strip().casefold()
    if "polar" in model:
        return "umof_polar"
    return "umof_mh"


def _resolve_figshare_asset(asset: dict[str, Any], model_path: Path) -> dict[str, Any]:
    article_id = str(asset.get("figshare_article_id", "")).strip()
    if not article_id:
        return asset
    files = _figshare_files(article_id)
    selected = _select_figshare_file(files, asset, model_path)
    download_url = str(selected.get("download_url", "")).strip()
    if not download_url:
        file_id = selected.get("id")
        if not file_id:
            raise ValueError(
                f"Figshare file metadata for article {article_id} has no "
                "download_url or file id."
            )
        download_url = f"https://api.figshare.com/v2/file/download/{file_id}"
    resolved = dict(asset)
    file_name = str(selected.get("name", model_path.name))
    if file_name.casefold().endswith(".zip") and (
        asset.get("member_path") or asset.get("member_basename")
    ):
        resolved["archive_url"] = download_url
        resolved.setdefault("archive_filename", file_name)
        if selected.get("supplied_md5"):
            resolved.setdefault("archive_md5", str(selected["supplied_md5"]))
    else:
        resolved["url"] = download_url
    resolved["figshare_file_name"] = file_name
    resolved["figshare_file_id"] = selected.get("id")
    return resolved


def _figshare_files(article_id: str) -> list[dict[str, Any]]:
    url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    with urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Unexpected Figshare response for article {article_id}.")
    return [item for item in data if isinstance(item, dict)]


def _select_figshare_file(
    files: list[dict[str, Any]],
    asset: dict[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    if not files:
        raise FileNotFoundError(
            f"No files were returned for Figshare article "
            f"{asset.get('figshare_article_id')}."
        )
    file_id = asset.get("file_id")
    if file_id is not None:
        for item in files:
            if str(item.get("id")) == str(file_id):
                return item
        raise FileNotFoundError(f"Figshare file id {file_id!r} was not found.")

    names = [
        str(asset.get("file_name", "")).strip(),
        *[str(value).strip() for value in asset.get("file_names", [])],
    ]
    names = [value for value in names if value]
    lower_names = {value.casefold() for value in names}
    for item in files:
        if str(item.get("name", "")).casefold() in lower_names:
            return item

    patterns = [
        str(value).casefold().replace("_", "-")
        for value in asset.get("file_name_patterns", [])
        if str(value).strip()
    ]
    extensions = [
        str(value).casefold()
        for value in asset.get("candidate_extensions", [])
        if str(value).strip()
    ]
    if not patterns:
        patterns = [model_path.stem.casefold().replace("_", "-")]

    matches = []
    for item in files:
        name = str(item.get("name", ""))
        comparable = name.casefold().replace("_", "-")
        if extensions and not any(comparable.endswith(ext) for ext in extensions):
            continue
        if any(pattern in comparable for pattern in patterns):
            matches.append(item)
    if not matches:
        available = ", ".join(str(item.get("name", "")) for item in files[:20])
        raise FileNotFoundError(
            "No matching uMOF model file was found in the Figshare release. "
            f"Patterns: {patterns}. First available files: {available}"
        )
    if len(matches) > 1:
        names = ", ".join(str(item.get("name", "")) for item in matches)
        raise ValueError(
            "uMOF Figshare file selection is ambiguous. Add asset.file_name "
            f"or asset.file_id. Matches: {names}"
        )
    return matches[0]


def _finalize_model_asset(
    model_path: Path,
    *,
    expected_sha256: Any,
    source_url: str,
    archive_member: str,
    transfer_mode: str,
) -> dict[str, Any]:
    digest = _file_digest(model_path, "sha256")
    if expected_sha256 and digest.casefold() != str(expected_sha256).casefold():
        model_path.unlink(missing_ok=True)
        raise ValueError(
            f"Extracted model checksum mismatch: expected {expected_sha256}, "
            f"found {digest}."
        )
    return {
        "status": "ready",
        "path": str(model_path),
        "sha256": digest,
        "downloaded": True,
        "source_url": source_url,
        "archive_member": archive_member,
        "transfer_mode": transfer_mode,
    }


def _extract_remote_zip_member(
    url: str,
    member_name: str,
    target: Path,
) -> str | None:
    """Extract one member using ZIP metadata and HTTP byte ranges."""
    total_size = _http_content_length(url)
    if total_size is None:
        return None
    tail_size = min(total_size, 65_557)
    tail = _http_range(url, total_size - tail_size, total_size - 1)
    if tail is None:
        return None
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0 or len(tail) - eocd_offset < 22:
        return None
    eocd = struct.unpack_from("<4s4H2IH", tail, eocd_offset)
    disk_number, central_disk, entries, total_entries = eocd[1:5]
    central_size, central_offset = eocd[5:7]
    if disk_number or central_disk or entries != total_entries:
        return None
    if central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        return None
    central = _http_range(
        url,
        central_offset,
        central_offset + central_size - 1,
    )
    if central is None:
        return None
    matches = _parse_central_directory(central, member_name)
    if not matches:
        raise FileNotFoundError(
            f"Model {member_name!r} was not found in remote archive {url}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Model name {member_name!r} is ambiguous in {url}: "
            + ", ".join(item["name"] for item in matches)
        )
    member = matches[0]
    local_header = _http_range(
        url,
        member["local_header_offset"],
        member["local_header_offset"] + 29,
    )
    if local_header is None or len(local_header) != 30:
        return None
    local = struct.unpack("<4s5H3I2H", local_header)
    if local[0] != b"PK\x03\x04":
        raise ValueError("Remote ZIP local header is invalid.")
    flags, compression = local[2], local[3]
    if flags & 0x1:
        raise ValueError("Encrypted ZIP model assets are not supported.")
    data_offset = member["local_header_offset"] + 30 + local[9] + local[10]
    payload = _http_range(
        url,
        data_offset,
        data_offset + member["compressed_size"] - 1,
    )
    if payload is None:
        return None
    if compression == 0:
        content = payload
    elif compression == 8:
        content = zlib.decompress(payload, -zlib.MAX_WBITS)
    else:
        raise ValueError(
            f"Unsupported ZIP compression method {compression} for "
            f"{member['name']}."
        )
    if len(content) != member["uncompressed_size"]:
        raise ValueError("Remote ZIP member size check failed.")
    if zlib.crc32(content) & 0xFFFFFFFF != member["crc32"]:
        raise ValueError("Remote ZIP member CRC check failed.")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_bytes(content)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return member["name"]


def _parse_central_directory(
    data: bytes,
    member_name: str,
) -> list[dict[str, Any]]:
    matches = []
    offset = 0
    header_size = 46
    while offset + header_size <= len(data):
        values = struct.unpack_from("<4s6H3I5H2I", data, offset)
        if values[0] != b"PK\x01\x02":
            break
        flags = values[3]
        name_length, extra_length, comment_length = values[10:13]
        name_start = offset + header_size
        name_end = name_start + name_length
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = data[name_start:name_end].decode(encoding)
        exact_path = "/" in member_name
        matches_member = (
            name == member_name
            if exact_path
            else PurePosixPath(name).name == member_name
        )
        if matches_member:
            matches.append(
                {
                    "name": name,
                    "crc32": values[7],
                    "compressed_size": values[8],
                    "uncompressed_size": values[9],
                    "local_header_offset": values[16],
                }
            )
        offset = name_end + extra_length + comment_length
    return matches


def _http_content_length(url: str) -> int | None:
    request = Request(url, method="HEAD")
    try:
        with urlopen(request) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def _http_range(url: str, start: int, end: int) -> bytes | None:
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    try:
        with urlopen(request) as response:
            if getattr(response, "status", None) != 206:
                return None
            payload = response.read()
    except Exception:
        return None
    expected = end - start + 1
    return payload if len(payload) == expected else None


def _download(url: str, target: Path) -> None:
    temporary_path = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(url) as response, temporary_path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary_path.replace(target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _find_archive_member(archive_path: Path, member_name: str) -> str:
    try:
        with ZipFile(archive_path) as archive:
            matches = [
                name
                for name in archive.namelist()
                if (
                    name == member_name
                    if "/" in member_name
                    else PurePosixPath(name).name == member_name
                )
            ]
    except BadZipFile as exc:
        raise ValueError(f"Downloaded model archive is not a ZIP: {archive_path}") from exc
    if not matches:
        raise FileNotFoundError(
            f"Model {member_name!r} was not found in {archive_path}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Model name {member_name!r} is ambiguous in {archive_path}: "
            + ", ".join(matches)
        )
    return matches[0]


def _file_digest(path: Path, algorithm: str) -> str:
    digest = md5() if algorithm == "md5" else sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
