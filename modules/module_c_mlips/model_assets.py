"""Download and verify local model assets declared by Module C configs."""

from __future__ import annotations

from hashlib import md5, sha256
from pathlib import Path, PurePosixPath
import shutil
import struct
from typing import Any
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
import zlib


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
            "--setup-mlip-mc or --run-mlip-mc --install-missing to download "
            "the configured model asset."
        )

    archive_url = str(asset.get("archive_url", "")).strip()
    member_basename = str(asset.get("member_basename", model_path.name)).strip()
    if not archive_url or not member_basename:
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
            member_basename,
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
    member = _find_archive_member(archive_path, member_basename)
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
    member_basename: str,
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
    matches = _parse_central_directory(central, member_basename)
    if not matches:
        raise FileNotFoundError(
            f"Model {member_basename!r} was not found in remote archive {url}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Model basename {member_basename!r} is ambiguous in {url}: "
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
    member_basename: str,
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
        if PurePosixPath(name).name == member_basename:
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


def _find_archive_member(archive_path: Path, basename: str) -> str:
    try:
        with ZipFile(archive_path) as archive:
            matches = [
                name
                for name in archive.namelist()
                if PurePosixPath(name).name == basename
            ]
    except BadZipFile as exc:
        raise ValueError(f"Downloaded model archive is not a ZIP: {archive_path}") from exc
    if not matches:
        raise FileNotFoundError(
            f"Model {basename!r} was not found in {archive_path}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Model basename {basename!r} is ambiguous in {archive_path}: "
            + ", ".join(matches)
        )
    return matches[0]


def _file_digest(path: Path, algorithm: str) -> str:
    digest = md5() if algorithm == "md5" else sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
