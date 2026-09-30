"""GoldDAC test-set setup and loading for independent DFT validation."""

from __future__ import annotations

from hashlib import md5, sha256
import json
from pathlib import Path
import random
import shutil
from typing import Any, Iterable
from urllib.request import urlopen
from zipfile import ZipFile

from modules.module_c_mlips.models import InteractionConfiguration


GOLDDAC_DOI = "10.6084/m9.figshare.27978474.v3"
GOLDDAC_LICENSE = "CC BY 4.0"
GOLDDAC_ARCHIVE_URL = "https://ndownloader.figshare.com/files/51021315"
GOLDDAC_ARCHIVE_MD5 = "695824c1088bb0b3b9a7018bac1165a1"
DEFAULT_GOLDDAC_ROOT = Path("external/datasets/golddac")
_SPLITS = ("train", "val", "test")
_REGIONS = {
    "R": "repulsive",
    "E": "equilibrium",
    "P": "weak_attraction",
}


def ensure_golddac_dataset(
    root: str | Path = DEFAULT_GOLDDAC_ROOT,
    *,
    download_missing: bool = False,
) -> dict[str, Any]:
    """Verify GoldDAC locally, downloading its versioned archive when asked."""
    target = Path(root).expanduser()
    split_paths = {split: target / f"{split}.xyz" for split in _SPLITS}
    missing = [path for path in split_paths.values() if not path.is_file()]
    downloaded = False

    if missing and not download_missing:
        raise FileNotFoundError(
            "GoldDAC is not installed at "
            f"{target}. Run benchmark.py with --setup-golddac first."
        )
    if missing:
        target.mkdir(parents=True, exist_ok=True)
        archive_path = target / "GoldDAC.zip"
        _download(GOLDDAC_ARCHIVE_URL, archive_path)
        archive_digest = _digest(archive_path, "md5")
        if archive_digest != GOLDDAC_ARCHIVE_MD5:
            archive_path.unlink(missing_ok=True)
            raise ValueError(
                "GoldDAC archive checksum mismatch: expected "
                f"{GOLDDAC_ARCHIVE_MD5}, found {archive_digest}."
            )
        with ZipFile(archive_path) as archive:
            members = {Path(name).name: name for name in archive.namelist()}
            filenames = ["README.md", *[f"{split}.xyz" for split in _SPLITS]]
            for filename in filenames:
                member = members.get(filename)
                if member is None:
                    raise FileNotFoundError(
                        f"GoldDAC archive does not contain {filename}."
                    )
                with archive.open(member) as source, (target / filename).open(
                    "wb"
                ) as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        archive_path.unlink(missing_ok=True)
        downloaded = True

    files = {
        split: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _digest(path, "sha256"),
        }
        for split, path in split_paths.items()
    }
    manifest = {
        "status": "ready",
        "dataset": "GoldDAC",
        "version": "v3",
        "doi": GOLDDAC_DOI,
        "license": GOLDDAC_LICENSE,
        "source_url": GOLDDAC_ARCHIVE_URL,
        "archive_md5": GOLDDAC_ARCHIVE_MD5,
        "root": str(target),
        "downloaded": downloaded,
        "files": files,
        "benchmark_split": "test",
        "training_splits": ["train", "val"],
    }
    (target / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_golddac_configurations(
    root_or_file: str | Path,
    *,
    split: str = "test",
    adsorbates: Iterable[str] | None = None,
    materials: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
    max_configurations: int | None = None,
    seed: int = 12345,
) -> list[InteractionConfiguration]:
    """Load the held-out GoldDAC test split as interaction configurations."""
    normalized_split = str(split).strip().casefold()
    if normalized_split != "test":
        raise ValueError(
            "GoldDAC benchmark evaluation must use split='test'. The train and "
            "validation splits are reserved for model development."
        )
    path = Path(root_or_file).expanduser()
    if path.is_dir() or path.suffix.casefold() != ".xyz":
        path = path / "test.xyz"
    if not path.is_file():
        raise FileNotFoundError(f"GoldDAC test split does not exist: {path}")
    if max_configurations is not None:
        if (
            isinstance(max_configurations, bool)
            or not isinstance(max_configurations, int)
            or max_configurations <= 0
        ):
            raise ValueError("max_configurations must be a positive integer.")

    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required to load GoldDAC extxyz data.") from exc

    adsorbate_filter = _normalized_filter(adsorbates)
    material_filter = _normalized_filter(materials)
    region_filter = _normalized_filter(regions)
    configurations = []
    for atoms in read(path, index=":", format="extxyz"):
        parsed = _parse_identity(atoms.info)
        if (
            adsorbate_filter
            and parsed["adsorbate"].casefold() not in adsorbate_filter
        ):
            continue
        if material_filter and parsed["material"].casefold() not in material_filter:
            continue
        if region_filter and parsed["region"].casefold() not in region_filter:
            continue
        configurations.append(_to_configuration(atoms, parsed, path))

    if not configurations:
        raise ValueError("No GoldDAC configurations match the configured filters.")
    if max_configurations is not None and len(configurations) > max_configurations:
        configurations = _balanced_sample(configurations, max_configurations, seed)
    return configurations


def _parse_identity(info: dict[str, Any]) -> dict[str, str]:
    name = str(info.get("name", "")).strip()
    material = str(info.get("group", "")).strip()
    if not name or not material or not name.startswith(material + "_"):
        raise ValueError(
            "GoldDAC frame requires consistent 'name' and 'group' metadata."
        )
    tokens = name[len(material) + 1 :].split("_")
    if len(tokens) < 2 or tokens[1] not in _REGIONS:
        raise ValueError(f"Cannot parse GoldDAC adsorbate/region from {name!r}.")
    return {
        "name": name,
        "material": material,
        "adsorbate": tokens[0],
        "raw_region": tokens[1],
        "region": _REGIONS[tokens[1]],
    }


def _to_configuration(
    atoms: Any,
    parsed: dict[str, str],
    path: Path,
) -> InteractionConfiguration:
    tags = atoms.get_tags().tolist()
    framework_indices = [index for index, tag in enumerate(tags) if int(tag) == 0]
    adsorbate_indices = [index for index, tag in enumerate(tags) if int(tag) == 1]
    if not framework_indices or not adsorbate_indices:
        raise ValueError(
            f"GoldDAC frame {parsed['name']!r} must contain tag 0 framework "
            "atoms and tag 1 adsorbate atoms."
        )
    if len(framework_indices) + len(adsorbate_indices) != len(atoms):
        raise ValueError(
            f"GoldDAC frame {parsed['name']!r} contains unsupported atom tags."
        )

    required = ("DFT_E_total", "DFT_E_mof", "DFT_E_gas", "DFT_E_int")
    missing = [key for key in required if key not in atoms.info]
    if missing or "REF_forces" not in atoms.arrays:
        raise ValueError(
            f"GoldDAC frame {parsed['name']!r} lacks reference fields: "
            + ", ".join(
                [
                    *missing,
                    *(
                        ["REF_forces"]
                        if "REF_forces" not in atoms.arrays
                        else []
                    ),
                ]
            )
        )
    total = float(atoms.info["DFT_E_total"])
    framework = float(atoms.info["DFT_E_mof"])
    adsorbate = float(atoms.info["DFT_E_gas"])
    interaction = float(atoms.info["DFT_E_int"])
    reconstructed = total - framework - adsorbate
    if abs(interaction - reconstructed) > 1.0e-6:
        raise ValueError(
            f"GoldDAC interaction-energy consistency check failed for "
            f"{parsed['name']!r}: stored {interaction}, reconstructed "
            f"{reconstructed}."
        )

    metadata = {
        "dataset": "GoldDAC",
        "dataset_version": "v3",
        "dataset_doi": GOLDDAC_DOI,
        "dataset_license": GOLDDAC_LICENSE,
        "dataset_split": "test",
        "dataset_path": str(path),
        "raw_name": parsed["name"],
        "raw_region": parsed["raw_region"],
        "metal": str(atoms.info.get("metal", "")),
        "reference_total_energy_ev": total,
        "reference_framework_energy_ev": framework,
        "reference_adsorbate_energy_ev": adsorbate,
        "reference_force_mode": "combined_total",
        "reference_force_definition": (
            "DFT forces on the combined MOF-guest system"
        ),
        "training_data": False,
    }
    return InteractionConfiguration(
        configuration_id=parsed["name"],
        material=parsed["material"],
        adsorbate=parsed["adsorbate"],
        atoms=atoms.copy(),
        framework_indices=framework_indices,
        adsorbate_indices=adsorbate_indices,
        reference_interaction_energy_ev=interaction,
        reference_forces_ev_per_angstrom=(
            atoms.arrays["REF_forces"].astype(float).tolist()
        ),
        region=parsed["region"],
        source="golddac_v3_test",
        metadata=metadata,
    )


def _balanced_sample(
    configurations: list[InteractionConfiguration],
    count: int,
    seed: int,
) -> list[InteractionConfiguration]:
    groups: dict[str, list[InteractionConfiguration]] = {}
    for configuration in configurations:
        groups.setdefault(
            configuration.region or "unclassified",
            [],
        ).append(configuration)
    generator = random.Random(seed)
    for values in groups.values():
        generator.shuffle(values)
    selected = []
    while len(selected) < count and groups:
        for region in sorted(list(groups)):
            values = groups[region]
            if values:
                selected.append(values.pop())
                if len(selected) == count:
                    break
            if not values:
                groups.pop(region)
    return selected


def _normalized_filter(values: Iterable[str] | None) -> set[str]:
    if isinstance(values, str):
        values = [values]
    return {
        str(value).strip().casefold()
        for value in (values or [])
        if str(value).strip()
    }


def _download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _digest(path: Path, algorithm: str) -> str:
    digest = md5() if algorithm == "md5" else sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
