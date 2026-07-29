from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
import json
from pathlib import Path
import re


DEFAULT_ALIAS_MAP = {
    "MOF-5": ["IRMOF-1", "IRMOF1", "MOF5", "Zn4O(BDC)3"],
    "IRMOF-1": ["MOF-5", "IRMOF1", "MOF5", "Zn4O(BDC)3"],
    "UiO-66": ["UIO66", "UiO66"],
    "HKUST-1": ["Cu-BTC", "CuBTC", "Basolite C300"],
    "MOF-74": ["CPO-27", "M-MOF-74"],
    "ZIF-8": ["ZIF8"],
    "ZIF-4": ["ZIF4"],
}


CHARGE_SCHEME_PRIORITY = ("DDEC", "EQeq", "Qeq", "PACMOF", "MPNN", "NEUTRAL")


@dataclass(frozen=True)
class MaterialMatch:
    query: str
    material_id: str
    cif_path: Path
    charge_scheme: str
    matched_by: str
    aliases: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()


class MaterialResolver:
    def __init__(
        self,
        crafted_root: str | Path = "CRAFTED-2.0.0",
        alias_file: str | Path = "data/material_aliases.json",
        nist_root: str | Path = "isodb-library",
        use_nist_aliases: bool = True,
    ) -> None:
        self.crafted_root = Path(crafted_root)
        self.cif_root = self.crafted_root / "CIF_FILES"
        self.alias_file = Path(alias_file)
        self.nist_root = Path(nist_root)
        self.use_nist_aliases = use_nist_aliases
        self.alias_map = self._load_aliases()
        self._cif_index = self._build_cif_index()
        if self.use_nist_aliases:
            self.alias_map = self._merge_nist_aliases(self.alias_map)
        self._name_index = self._build_name_index()

    def resolve(self, name: str, charge_scheme: str = "auto") -> MaterialMatch:
        if not name or not name.strip():
            raise ValueError("Material name must not be empty.")

        candidates = self._candidate_names(name)
        for candidate in candidates:
            material_id = self._name_index.get(_normalize_name(candidate))
            if material_id:
                return self._match(name, material_id, charge_scheme, "alias" if candidate != name else "exact")

        suggestions = self.suggest(name)
        raise LookupError(
            f"Could not resolve material {name!r}. "
            f"Closest known names: {', '.join(suggestions) if suggestions else 'none'}"
        )

    def suggest(self, name: str, limit: int = 5) -> tuple[str, ...]:
        normalized_names = list(self._name_index)
        alias_index = self._alias_index()
        normalized_names.extend(alias_index)
        matches = get_close_matches(_normalize_name(name), normalized_names, n=limit, cutoff=0.55)
        suggestions: list[str] = []
        for match in matches:
            suggestions.append(self._name_index.get(match) or alias_index[match])
        return tuple(_dedupe(suggestions))

    def _match(self, query: str, material_id: str, charge_scheme: str, matched_by: str) -> MaterialMatch:
        scheme = self._select_charge_scheme(material_id, charge_scheme)
        return MaterialMatch(
            query=query,
            material_id=material_id,
            cif_path=self._cif_index[material_id][scheme],
            charge_scheme=scheme,
            matched_by=matched_by,
            aliases=tuple(self.alias_map.get(material_id, ())),
            suggestions=(),
        )

    def _select_charge_scheme(self, material_id: str, requested: str) -> str:
        available = self._cif_index[material_id]
        if requested and requested.lower() != "auto":
            scheme = _lookup_case_insensitive(requested, available)
            if scheme is None:
                raise LookupError(
                    f"Material {material_id!r} has no CIF for charge scheme {requested!r}. "
                    f"Available: {', '.join(sorted(available))}"
                )
            return scheme

        for scheme in CHARGE_SCHEME_PRIORITY:
            if scheme in available:
                return scheme
        return sorted(available)[0]

    def _candidate_names(self, name: str) -> list[str]:
        candidates = [name]
        normalized_query = _normalize_name(name)

        for canonical, aliases in self.alias_map.items():
            all_names = [canonical, *aliases]
            if normalized_query in {_normalize_name(item) for item in all_names}:
                candidates.extend(all_names)
                candidates.append(canonical)

        return _dedupe(candidates)

    def _build_cif_index(self) -> dict[str, dict[str, Path]]:
        if not self.cif_root.exists():
            raise FileNotFoundError(f"CRAFTED CIF directory not found: {self.cif_root}")

        index: dict[str, dict[str, Path]] = {}
        for cif_path in self.cif_root.glob("*/*.cif"):
            # CRAFTED archives created on macOS can contain AppleDouble
            # metadata files such as ``._IRMOF-1.cif``. They are not CIF
            # structures and must not participate in material resolution.
            if cif_path.name.startswith("._") or cif_path.parent.name.startswith("._"):
                continue
            charge_scheme = cif_path.parent.name
            material_id = cif_path.stem
            index.setdefault(material_id, {})[charge_scheme] = cif_path

        if not index:
            raise LookupError(f"No CIF files found below {self.cif_root}")
        return index

    def _build_name_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for material_id in self._cif_index:
            index[_normalize_name(material_id)] = material_id
        return index

    def _load_aliases(self) -> dict[str, list[str]]:
        aliases = {key: list(value) for key, value in DEFAULT_ALIAS_MAP.items()}
        if self.alias_file.exists():
            with self.alias_file.open("r", encoding="utf-8") as handle:
                user_aliases = json.load(handle)
            for canonical, values in user_aliases.items():
                aliases.setdefault(canonical, [])
                aliases[canonical].extend(values)
        return {canonical: _dedupe(values) for canonical, values in aliases.items()}

    def _merge_nist_aliases(self, aliases: dict[str, list[str]]) -> dict[str, list[str]]:
        names = _load_nist_adsorbent_names(self.nist_root)
        if not names:
            return aliases

        known = self._known_alias_targets(aliases)
        for nist_name in names:
            if not _looks_like_base_material_name(nist_name):
                continue
            for candidate in _nist_name_candidates(nist_name):
                material_id = known.get(_normalize_name(candidate))
                if material_id is not None:
                    aliases.setdefault(material_id, [])
                    if _normalize_name(nist_name) != _normalize_name(material_id):
                        aliases[material_id].append(nist_name)
                    if _normalize_name(candidate) != _normalize_name(material_id):
                        aliases[material_id].append(candidate)

        return {canonical: _dedupe(values) for canonical, values in aliases.items()}

    def _known_alias_targets(self, aliases: dict[str, list[str]]) -> dict[str, str]:
        targets: dict[str, str] = {}
        for material_id in self._cif_index:
            targets[_normalize_name(material_id)] = material_id
        for canonical, values in aliases.items():
            if canonical in self._cif_index:
                target = canonical
            else:
                target = targets.get(_normalize_name(canonical))
                if target is None:
                    continue
            for value in [canonical, *values]:
                targets[_normalize_name(value)] = target
        return targets

    def _alias_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for canonical, aliases in self.alias_map.items():
            material_id = self._name_index.get(_normalize_name(canonical), canonical)
            for alias in aliases:
                index[_normalize_name(alias)] = material_id
        return index


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = _normalize_name(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _lookup_case_insensitive(value: str, mapping: dict[str, Path]) -> str | None:
    normalized = value.casefold()
    for key in mapping:
        if key.casefold() == normalized:
            return key
    return None


def _load_nist_adsorbent_names(nist_root: Path) -> tuple[str, ...]:
    return _load_nist_adsorbent_names_cached(str(nist_root))


@lru_cache(maxsize=8)
def _load_nist_adsorbent_names_cached(nist_root: str) -> tuple[str, ...]:
    nist_root_path = Path(nist_root)
    search_root = nist_root_path / "Library" if (nist_root_path / "Library").exists() else nist_root_path
    if not search_root.exists():
        return ()

    names: list[str] = []
    for path in search_root.glob("*/*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        name = str((data.get("adsorbent") or {}).get("name", "")).strip()
        if name:
            names.append(name)
    return tuple(_dedupe(names))


def _nist_name_candidates(name: str) -> list[str]:
    candidates = [name.strip()]
    candidates.extend(re.findall(r"\(([^)]+)\)", name))
    candidates.extend(re.findall(r"\[([^\]]+)\]", name))
    return _dedupe([candidate.strip() for candidate in candidates if candidate.strip()])


def _looks_like_base_material_name(name: str) -> bool:
    normalized = name.casefold()
    unsafe_fragments = (
        "/",
        "@",
        " wt",
        "wt%",
        "composite",
        "nanoparticle",
        "graphene",
        "oxide",
        "functionalized",
        "impregnated",
        "loaded",
        "activated carbon",
        "carbon black",
        "mwcnt",
        "cnt",
    )
    if any(fragment in normalized for fragment in unsafe_fragments):
        return False
    if re.search(r"^\d+\s*(pt|pd|cu|zn|mg|co|ni|fe|ag|au)[/@-]", normalized):
        return False
    return True
