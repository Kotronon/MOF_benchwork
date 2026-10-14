from __future__ import annotations

from hashlib import md5, sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from modules.module_c_mlips.model_assets import (
    ensure_model_asset,
    ensure_umof_model_asset,
)


class ModelAssetTests(unittest.TestCase):
    def test_downloads_direct_model_with_sha256_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.model"
            source.write_bytes(b"direct-model")
            target = root / "models" / "target.model"

            result = ensure_model_asset(
                {
                    "model": str(target),
                    "sha256": sha256(b"direct-model").hexdigest(),
                    "asset": {"url": source.as_uri()},
                },
                download_missing=True,
            )

            self.assertEqual(target.read_bytes(), b"direct-model")
            self.assertEqual(result["transfer_mode"], "direct_download")

    def test_extracts_remote_member_with_byte_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "source.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "data/training/model_ZIF8_N=1000.pth",
                    b"range-model",
                )
            archive_bytes = archive_path.read_bytes()
            target = root / "model.pth"
            with patch(
                "modules.module_c_mlips.model_assets._http_content_length",
                return_value=len(archive_bytes),
            ), patch(
                "modules.module_c_mlips.model_assets._http_range",
                side_effect=lambda _url, start, end: archive_bytes[start:end + 1],
            ):
                result = ensure_model_asset(
                    {
                        "model": str(target),
                        "asset": {
                            "archive_url": "https://example.test/data.zip",
                            "member_basename": "model_ZIF8_N=1000.pth",
                        },
                    },
                    download_missing=True,
                )

            self.assertEqual(target.read_bytes(), b"range-model")
            self.assertEqual(result["transfer_mode"], "http_range")
            self.assertEqual(
                result["archive_member"],
                "data/training/model_ZIF8_N=1000.pth",
            )

    def test_extracts_named_model_from_verified_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "source.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("models/reference.pth", b"model-data")
            checksum = md5(archive_path.read_bytes()).hexdigest()
            target = root / "models" / "reference.pth"
            specification = {
                "model": str(target),
                "asset": {
                    "archive_url": archive_path.as_uri(),
                    "archive_filename": "download.zip",
                    "archive_md5": checksum,
                    "member_basename": "reference.pth",
                    "keep_archive": False,
                },
            }

            result = ensure_model_asset(specification, download_missing=True)

            self.assertTrue(result["downloaded"])
            self.assertEqual(target.read_bytes(), b"model-data")
            self.assertFalse((target.parent / ".downloads" / "download.zip").exists())

    def test_missing_model_requires_explicit_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "missing.pth"
            with self.assertRaisesRegex(FileNotFoundError, "--setup-mlip-mc"):
                ensure_model_asset(
                    {
                        "model": str(target),
                        "asset": {
                            "archive_url": "https://example.invalid/data.zip",
                            "member_basename": "missing.pth",
                        },
                    }
                )

    def test_umof_asset_defaults_to_figshare_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "umof-mh.model"
            downloads = []

            def fake_download(url: str, output_path: Path) -> None:
                downloads.append(url)
                output_path.write_bytes(b"umof-model")

            with patch(
                "modules.module_c_mlips.model_assets._figshare_files",
                return_value=[
                    {
                        "id": 11,
                        "name": "uMOF-MH.model",
                        "download_url": "https://figshare.test/umof-mh.model",
                    },
                    {
                        "id": 12,
                        "name": "uMOF-POLAR.model",
                        "download_url": "https://figshare.test/umof-polar.model",
                    },
                ],
            ), patch(
                "modules.module_c_mlips.model_assets._download",
                side_effect=fake_download,
            ):
                result = ensure_umof_model_asset(
                    {
                        "name": "umof_mh",
                        "model_family": "umof",
                        "model": str(target),
                    },
                    download_missing=True,
                )

            self.assertEqual(target.read_bytes(), b"umof-model")
            self.assertEqual(downloads, ["https://figshare.test/umof-mh.model"])
            self.assertEqual(result["transfer_mode"], "direct_download")
            self.assertTrue(result["downloaded"])

    def test_umof_download_requires_install_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "umof-polar.model"
            with self.assertRaisesRegex(FileNotFoundError, "--install-missing"):
                ensure_umof_model_asset(
                    {
                        "name": "umof_polar",
                        "model_family": "umof",
                        "model": str(target),
                    }
                )


if __name__ == "__main__":
    unittest.main()
