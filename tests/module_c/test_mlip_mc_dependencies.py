from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock, patch

from modules.module_c_mlips.dependencies import (
    DependencyStatus,
    ensure_mlip_mc_dependencies,
    normalize_mlip_backend,
)


class MlipMcDependencyTests(unittest.TestCase):
    def test_backend_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_mlip_backend("mace_mp"), "mace-torch")
        self.assertEqual(normalize_mlip_backend("ORB"), "orb-models")
        self.assertEqual(normalize_mlip_backend("ODAC"), "fairchem")

    def test_missing_dependency_does_not_install_without_opt_in(self) -> None:
        missing = _status(ready=False)
        with patch(
            "modules.module_c_mlips.dependencies.inspect_mlip_mc_dependencies",
            return_value=missing,
        ), patch("modules.module_c_mlips.dependencies.subprocess.run") as run:
            with self.assertRaisesRegex(ImportError, "--install-missing"):
                ensure_mlip_mc_dependencies("mace-torch")
        run.assert_not_called()

    def test_auto_install_uses_active_python_and_pinned_package(self) -> None:
        missing = _status(ready=False)
        ready = _status(ready=True, installed_version="0.1.3")
        completed = Mock(returncode=0, stdout="", stderr="")
        with patch(
            "modules.module_c_mlips.dependencies.inspect_mlip_mc_dependencies",
            side_effect=[missing, ready],
        ), patch(
            "modules.module_c_mlips.dependencies.subprocess.run",
            return_value=completed,
        ) as run:
            result = ensure_mlip_mc_dependencies(
                "mace-torch",
                install_missing=True,
            )

        self.assertTrue(result.ready)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        self.assertEqual(command[-1], "mlip-mc==0.1.3")


def _status(
    *,
    ready: bool,
    installed_version: str | None = None,
) -> DependencyStatus:
    return DependencyStatus(
        backend="mace-torch",
        python_executable=sys.executable,
        mlip_mc_required_version="0.1.3",
        mlip_mc_installed_version=installed_version,
        backend_module="mace",
        mlip_mc_available=ready,
        backend_available=True,
        ready=ready,
        install_specification="mlip-mc==0.1.3",
    )


if __name__ == "__main__":
    unittest.main()
