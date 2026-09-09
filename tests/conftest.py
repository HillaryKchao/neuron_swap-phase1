import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Location of the BBP L5PC mechanisms (override with NEURON_SWAP_MOD_DIR).
DEFAULT_MOD_DIR = Path("/Users/hillary/dev/bouchardlabfiles/l5pc_model/mechanisms")
MOD_FILES = [
    "NaTa_t",
    "NaTs2_t",
    "Nap_Et2",
    "K_Tst",
    "K_Pst",
    "SKv3_1",
    "SK_E2",
    "Im",
    "Ih",
    "Ca_HVA",
    "Ca_LVAst",
    "CaDynamics_E2",
]


def mod_dir() -> Path:
    return Path(os.environ.get("NEURON_SWAP_MOD_DIR", DEFAULT_MOD_DIR))


def nrnivmodl_path() -> str | None:
    """``nrnivmodl`` from PATH or from the bin directory of this interpreter
    (the env need not be activated to run the tests)."""
    found = shutil.which("nrnivmodl")
    if found:
        return found
    candidate = Path(sys.executable).parent / "nrnivmodl"
    return str(candidate) if candidate.exists() else None


def have_neuron() -> bool:
    try:
        import neuron  # noqa: F401
        from neuron import h  # noqa: F401
    except Exception:
        return False
    return nrnivmodl_path() is not None


def have_dca() -> bool:
    try:
        import dca  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def mod_text() -> dict[str, str]:
    """Raw text of each L5PC .mod file (skips if the files are absent)."""
    d = mod_dir()
    if not d.is_dir():
        pytest.skip(f"L5PC .mod files not found at {d}")
    return {name: (d / f"{name}.mod").read_text() for name in MOD_FILES}


@pytest.fixture(scope="session")
def nrn(tmp_path_factory):
    """NEURON ``h`` with the 12 L5PC mechanisms compiled in a scratch dir."""
    if not have_neuron():
        pytest.skip("NEURON (with nrnivmodl) is not available")
    d = mod_dir()
    if not d.is_dir():
        pytest.skip(f"L5PC .mod files not found at {d}")
    build = tmp_path_factory.mktemp("l5pc_mods")
    for name in MOD_FILES:
        shutil.copy(d / f"{name}.mod", build / f"{name}.mod")
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    res = subprocess.run([nrnivmodl_path(), "."], cwd=build, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        pytest.skip(f"nrnivmodl failed:\n{res.stdout}\n{res.stderr}")
    import neuron
    from neuron import h

    neuron.load_mechanisms(str(build))
    h.load_file("stdrun.hoc")
    return h
