"""SeedFormer loader - isolated import shim.

SeedFormer's model.py uses `from models.utils import ...` (top-level relative
style). PoinTr's `baselines/PoinTr` is also on sys.path and registers its own
`models` package (which has no `utils.py`). Naive `import model` after PoinTr
load -> ImportError because `sys.modules['models']` is PoinTr's package.

This shim solves the collision by:
1. Pop the `'models'` (and submodule) entries from sys.modules, save them.
2. Prepend SeedFormer's path to sys.path so its `models/utils.py` resolves first.
3. Force re-import of `model` and let SeedFormer load its own `models.utils`.
4. After instantiation, restore PoinTr's `models` namespace so other baselines
   (PoinTr / AdaPoinTr / SnowFlakeNet) continue to work - the already-built
   SeedFormer instance does not depend on `sys.modules['models']` at runtime
   because imports were resolved at module load time and bound into closures.

Usage:
    from src.seedformer_loader import load_seedformer_dim128
    model = load_seedformer_dim128(up_factors=[1, 4, 8])
    model.load_state_dict(...)
    model.eval().to(device)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SF_ROOT = REPO_ROOT / "baselines" / "SeedFormer"


def load_seedformer_dim128(up_factors):
    """Build a fresh SeedFormer model with given up_factors, isolated from PoinTr."""
    if not (SF_ROOT / "model.py").is_file():
        raise FileNotFoundError(
            f"SeedFormer model.py not found at {SF_ROOT/'model.py'}. "
            f"Copy or symlink the official SeedFormer `codes/` directory to "
            f"`baselines/SeedFormer` (see README)."
        )

    sf_root_str = str(SF_ROOT)
    saved_models = {}
    for k in list(sys.modules.keys()):
        if k == "models" or k.startswith("models."):
            saved_models[k] = sys.modules.pop(k)
    # Save and restore the top-level `model` too, otherwise a pre-existing module
    # named 'model' from another caller would be clobbered.
    saved_model_top = sys.modules.pop("model", None)

    inserted = False
    if sf_root_str not in sys.path:
        sys.path.insert(0, sf_root_str)
        inserted = True

    try:
        import model as _sf_main  # SeedFormer codes/model.py
        return _sf_main.seedformer_dim128(up_factors=up_factors)
    finally:
        sys.modules.pop("model", None)
        for k in list(sys.modules.keys()):
            if k == "models" or k.startswith("models."):
                sys.modules.pop(k, None)
        for k, v in saved_models.items():
            sys.modules[k] = v
        if saved_model_top is not None:
            sys.modules["model"] = saved_model_top
        if inserted:
            try:
                sys.path.remove(sf_root_str)
            except ValueError:
                pass
