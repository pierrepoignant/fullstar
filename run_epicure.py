#!/usr/bin/env python3
"""Work with the Epicure food-ingredient embeddings (arXiv:2605.22391).

The model ships its own loader helper (`epicure.py`) inside the HF repo, so we
download that first, put it on the import path, then drive the `Epicure` class.

    ./venv/bin/python run_epicure.py
"""
import os
import sys

from huggingface_hub import hf_hub_download

REPO = "Kaikaku/epicure-core"  # siblings: epicure-cooc, epicure-chem


def load_model(repo: str = REPO):
    """Download the shipped helper, import it, and load the embeddings."""
    epicure_py = hf_hub_download(repo_id=repo, filename="epicure.py")
    sys.path.append(os.path.dirname(epicure_py))
    from epicure import Epicure  # noqa: E402  (path set above)

    return Epicure.from_pretrained(repo)


def main():
    m = load_model()
    print(m, "\n")

    # 1. Nearest neighbours in the 300-D space.
    print("neighbors('chicken', k=5):")
    for name, sim in m.neighbors("chicken", k=5):
        print(f"  {name:<28} {sim:.3f}")

    # 2. SLERP: rotate a seed toward a supervised pole (cuisine / food_group / nova_level...).
    print("\nslerp('rice' -> cuisine:South_Asian, 30deg, k=5):")
    for name, sim in m.slerp("rice", "cuisine:South_Asian", theta_deg=30, k=5):
        print(f"  {name:<28} {sim:.3f}")

    # 3. Closest emergent factor modes for an ingredient.
    print("\nclosest_mode('chocolate', kind='factor', k=3):")
    for mode_id, label, sim in m.closest_mode("chocolate", kind="factor", k=3):
        print(f"  {mode_id:<14} {label:<32} {sim:.3f}")

    # Bonus: introspect what directions/modes are available to steer toward.
    cuisines = m.list_supervised_poles(prefix="cuisine:")
    print(f"\n{len(cuisines)} cuisine poles available, e.g. {cuisines[:5]}")
    factors = m.list_modes(kind="factor")
    print(f"{len(factors)} factor modes available, e.g. {[f[1] for f in factors[:5]]}")


if __name__ == "__main__":
    main()
