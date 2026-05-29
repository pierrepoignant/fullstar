#!/usr/bin/env python3
"""Vegetable-chopper recipe builder on top of Epicure embeddings (arXiv:2605.22391).

Epicure is an *embedding* model, not a recipe generator. So we use its geometry:
  - neighbours of a seed vegetable  -> 3-5 coherent ingredients ("chop these")
  - sauce whose vector best fits the ingredient centroid -> "+ a sauce"
  - the sauce's type (dressing vs cooking) decides chop-&-salad vs chop-&-cook

Usage:
    ./venv/bin/python chopper_recipes.py                 # a few demo recipes
    ./venv/bin/python chopper_recipes.py carrot
    ./venv/bin/python chopper_recipes.py cabbage cuisine:East_Asian
"""
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "Kaikaku/epicure-core"

# Sauces we treat as raw/cold dressings -> "chop & salad"; everything else cooks.
RAW_MARKERS = ("dressing", "vinegar", "vinaigrette", "tahini", "yogurt",
               "mayo", "mayonnaise", "pesto", "olive_oil", "lemon", "lime")
# Plain neutral fats make boring "sauces" -> skip them as the headline sauce.
SKIP_SAUCE = {"canola_oil", "corn_oil", "vegetable_oil", "grapeseed_oil",
              "soybean_oil", "sunflower_oil", "gum_paste", "almond_paste"}


def load_model(repo=REPO):
    p = hf_hub_download(repo_id=repo, filename="epicure.py")
    sys.path.append(os.path.dirname(p))
    from epicure import Epicure  # noqa: E402
    return Epicure.from_pretrained(repo)


# --- Food-group classifier (nearest exemplar prototype) -------------------
# The model's fg_* poles are direction vectors, not class centroids, so they
# classify poorly. Small exemplar sets per category give far cleaner results.
ANCHORS = {
    "Vegetable": ["carrot", "cabbage", "broccoli", "spinach", "zucchini",
                  "cucumber", "cauliflower", "celery", "kale", "bell_pepper",
                  "onion", "tomato", "scallion", "bean_sprout", "mushroom",
                  "leek", "eggplant", "swiss_chard"],
    "Feculent": ["rice", "pasta", "bread", "quinoa", "potato", "couscous",
                 "barley", "oat", "sweet_potato", "tortilla", "noodle"],
    "Protein": ["chicken", "beef", "pork", "tofu", "egg", "shrimp", "salmon",
                "lentil", "chickpea", "black_bean"],
    "Dairy": ["cheese", "milk", "butter", "cream", "yogurt"],
    "Spice": ["turmeric", "cumin", "cinnamon", "paprika", "black_pepper",
              "ginger", "coriander"],
    "Fruit": ["apple", "banana", "orange", "lemon", "strawberry", "mango"],
}
_PROTOS = {}  # cached category -> unit prototype vector


def _protos(m):
    if not _PROTOS:
        for cat, xs in ANCHORS.items():
            vs = [m.vec(x) for x in xs if x in m.vocab]
            c = np.mean(vs, axis=0)
            _PROTOS[cat] = c / np.linalg.norm(c)
    return _PROTOS


def classify(m, name):
    """Nearest food-group category for an ingredient (Vegetable/Feculent/...)."""
    if name not in m.vocab:
        return None
    v = m.vec(name)
    return max(_protos(m), key=lambda c: float(v @ _PROTOS[c]))


def _is_sauce(name):
    # Real sauces/pastes/dressings only — bare oils sit near every veg centroid
    # and would otherwise win every time, so they're excluded as the headline sauce.
    return any(t in name for t in ("sauce", "vinegar", "dressing", "paste",
                                   "miso", "tahini", "yogurt", "pesto",
                                   "chutney", "salsa", "hummus", "aioli")) \
        and name not in SKIP_SAUCE


def pick_ingredients(m, seed, n=4, cuisine=None, veg_only=False):
    """Seed veg + n coherent companions. Optionally steer toward a cuisine pole
    and/or restrict companions to items classified as Vegetable."""
    # Pull a wider candidate pool when filtering, so we can still reach n.
    k = n * (8 if veg_only else 4)
    if cuisine:
        # Rotate the seed toward the cuisine, then read off neighbours of that point.
        cand = m.slerp(seed, cuisine, theta_deg=25, k=k)
    else:
        cand = m.neighbors(seed, k=k)
    chosen = [seed]
    for name, _ in cand:
        if _is_sauce(name) or name in chosen:
            continue
        # skip non-choppable liquids/bases (stock, broth, water — but keep water_chestnut)
        if any(t in name for t in ("stock", "broth")) or name == "water":
            continue
        # de-dupe near-identical items (e.g. onion / red_onion / yellow_onion)
        if any(name in c or c in name for c in chosen):
            continue
        if veg_only and classify(m, name) != "Vegetable":
            continue
        chosen.append(name)
        if len(chosen) > n:
            break
    return chosen


# Curated starch shortlist — classify() is too noisy to mine these reliably
# (it tags nori/red_onion as "Feculent"), so we pick the best geometric fit
# from a known-good list of real starches instead.
FECULENTS = ["rice", "brown_rice", "basmati_rice", "pasta", "noodle",
             "rice_noodle", "egg_noodle", "potato", "sweet_potato", "bread",
             "naan", "tortilla", "quinoa", "couscous", "barley", "bulgur",
             "polenta", "gnocchi", "farro", "millet", "vermicelli"]


# Curated protein shortlist (same rationale as FECULENTS).
PROTEINS = ["chicken", "beef", "pork", "lamb", "turkey", "duck", "bacon",
            "ham", "sausage", "tofu", "tempeh", "paneer", "egg", "shrimp",
            "salmon", "tuna", "cod", "fish", "crab", "squid", "scallop",
            "lentil", "chickpea", "black_bean", "kidney_bean", "edamame"]


def _best_fit(m, ingredients, pool):
    """Item from pool whose vector best fits the ingredient centroid."""
    vecs = [m.vec(i) for i in ingredients if i in m.vocab]
    centroid = np.mean(vecs, axis=0)
    centroid /= np.linalg.norm(centroid)
    pool = [x for x in pool if x in m.vocab and x not in ingredients]
    return max(pool, key=lambda x: float(centroid @ m.vec(x))) if pool else None


def best_feculent(m, ingredients):
    """Real starch from FECULENTS whose vector best fits the ingredient centroid."""
    return _best_fit(m, ingredients, FECULENTS)


def best_protein(m, ingredients):
    """Real protein from PROTEINS whose vector best fits the ingredient centroid."""
    return _best_fit(m, ingredients, PROTEINS)


def best_sauce(m, ingredients):
    """Sauce vector closest to the centroid of the chosen ingredients."""
    vecs = [m.vec(i) for i in ingredients if i in m.vocab]
    centroid = np.mean(vecs, axis=0)
    centroid /= np.linalg.norm(centroid)
    best, best_sim = None, -1.0
    for name in m.vocab:
        if not _is_sauce(name):
            continue
        sim = float(centroid @ m.vec(name))
        if sim > best_sim:
            best, best_sim = name, sim
    return best, best_sim


def method_for(sauce):
    if any(t in sauce for t in RAW_MARKERS):
        return "chop & salad", "Toss raw, dress, rest 10 min so flavours meld."
    return "chop & cook", "Saute the chopped veg hot 6-8 min, stir the sauce in to finish."


def pretty(name):
    return name.replace("_", " ")


def make_recipe(m, seed, cuisine=None, n=4, veg_only=False, feculent=False,
                protein=False):
    """Return a recipe as a plain dict (shared by the CLI and the Flask app).

    veg_only  -> chopped companions restricted to Vegetable-class items.
    feculent  -> add one best-fitting starch (rice/pasta/potato/...) on the side.
    protein   -> add one best-fitting protein (chicken/tofu/shrimp/...).
    """
    ings = pick_ingredients(m, seed, n=n, cuisine=cuisine, veg_only=veg_only)
    fec = best_feculent(m, ings) if feculent else None
    pro = best_protein(m, ings) if protein else None
    sauce, sim = best_sauce(m, ings)
    method, how = method_for(sauce)
    title = f"{pretty(seed).title()} {method.split(' & ')[1]}"
    if cuisine:
        title += f" ({cuisine.split(':')[1].replace('_', ' ')})"
    extra = ""
    if pro:
        extra += f" Add {pretty(pro)}."
    if fec:
        extra += f" Serve over {pretty(fec)}."
    return {
        "title": title,
        "method": method,
        "ingredients": [pretty(i) for i in ings],
        "protein": pretty(pro) if pro else None,
        "feculent": pretty(fec) if fec else None,
        "sauce": pretty(sauce),
        "fit": round(sim, 2),
        "steps": f"Chop everything to even pieces. {how}{extra}",
        "cuisine": cuisine,
    }


def build(m, seed, cuisine=None):
    r = make_recipe(m, seed, cuisine=cuisine)
    print(f"\n=== {r['title']} ===")
    print(f"  Method : {r['method']}")
    print(f"  Chop   : {', '.join(r['ingredients'])}  ({len(r['ingredients'])} ingredients)")
    if r.get("protein"):
        print(f"  Protein: {r['protein']}")
    if r.get("feculent"):
        print(f"  Serve  : over {r['feculent']}")
    print(f"  Sauce  : {r['sauce']}  (fit {r['fit']})")
    print(f"  Steps  : {r['steps']}")


def main():
    m = load_model()
    args = sys.argv[1:]
    if args:
        seed = args[0]
        cuisine = args[1] if len(args) > 1 else None
        if seed not in m.vocab:
            sys.exit(f"'{seed}' not in vocab. Try e.g. carrot, cabbage, zucchini, beet, fennel.")
        build(m, seed, cuisine)
    else:
        for seed in ("carrot", "cabbage", "zucchini", "beet", "broccoli"):
            build(m, seed)
        build(m, "cabbage", cuisine="cuisine:East_Asian")
        build(m, "carrot", cuisine="cuisine:South_Asian")


if __name__ == "__main__":
    main()
