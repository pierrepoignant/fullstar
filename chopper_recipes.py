#!/usr/bin/env python3
"""Vegetable-chopper recipe builder on top of Epicure embeddings (arXiv:2605.22391).

Epicure is an *embedding* model, not a recipe generator. We drive its geometry:
  - a query vector (seed, optionally blended with a 2nd seed, optionally rotated
    toward one or more steer directions) -> 3-5 coherent ingredients
  - a sauce / protein / feculent whose vector best fits the ingredient centroid
  - the sauce type (or an explicit override) decides chop-&-salad vs chop-&-cook

Steer directions come from three sources:
  - the model's supervised poles      -> cuisine, processing level (nova)
  - the model's emergent factor modes -> "factor:<mode_id>"
  - exemplar contrastive axes (below) -> flavor + aroma + nutrition
    (the model's taste/aroma score poles are cuisine-entangled and cancel out
    when averaged, so we define those axes from small exemplar sets instead.)

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
SIBLINGS = {                       # the chemistry-vs-recipe-context design axis
    "core": "Kaikaku/epicure-core",   # FlavorDB chemistry + recipe co-occurrence
    "cooc": "Kaikaku/epicure-cooc",   # recipe-context only
    "chem": "Kaikaku/epicure-chem",   # chemistry only
}

# Sauces we treat as raw/cold dressings -> "chop & salad"; everything else cooks.
RAW_MARKERS = ("dressing", "vinegar", "vinaigrette", "tahini", "yogurt",
               "mayo", "mayonnaise", "pesto", "olive_oil", "lemon", "lime")
SKIP_SAUCE = {"canola_oil", "corn_oil", "vegetable_oil", "grapeseed_oil",
              "soybean_oil", "sunflower_oil", "gum_paste", "almond_paste"}


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


_MODELS = {}  # sibling key -> loaded Epicure


def load_model(repo=REPO):
    p = hf_hub_download(repo_id=repo, filename="epicure.py")
    sys.path.append(os.path.dirname(p))
    from epicure import Epicure  # noqa: E402
    return Epicure.from_pretrained(repo)


def get_model(sibling="core"):
    """Lazily load and cache a sibling model (core / cooc / chem)."""
    if sibling not in _MODELS:
        _MODELS[sibling] = load_model(SIBLINGS.get(sibling, REPO))
    return _MODELS[sibling]


# --- Food-group classifier (nearest exemplar prototype) -------------------
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
_PROTOS = {}  # id(model) -> {category: unit prototype}


def _protos(m):
    key = id(m)
    if key not in _PROTOS:
        _PROTOS[key] = {cat: _unit(np.mean([m.vec(x) for x in xs if x in m.vocab],
                                           axis=0))
                        for cat, xs in ANCHORS.items()}
    return _PROTOS[key]


def classify(m, name):
    """Nearest food-group category for an ingredient (Vegetable/Feculent/...)."""
    if name not in m.vocab:
        return None
    v = m.vec(name)
    p = _protos(m)
    return max(p, key=lambda c: float(v @ p[c]))


# --- Steer axes (exemplar contrastive: unit(mean(high) - mean(low))) ------
FLAVOR_AXES = {
    "sweeter":     (["honey", "sugar", "maple_syrup", "brown_sugar", "date", "raisin"], ["garlic", "onion", "soy_sauce"]),
    "more sour":   (["lemon", "lime", "vinegar", "tamarind", "rice_vinegar"], ["sugar", "honey"]),
    "more bitter": (["kale", "radicchio", "coffee", "endive", "arugula"], ["sugar", "honey"]),
    "more umami":  (["soy_sauce", "miso", "tomato", "anchovy", "fish_sauce", "mushroom"], ["sugar", "lettuce"]),
    "spicier":     (["chili_pepper", "cayenne_pepper", "black_pepper", "ginger", "mustard", "jalapeno"], ["lettuce", "cucumber"]),
    "richer":      (["butter", "olive_oil", "cream", "cheese", "avocado", "coconut_milk"], ["lettuce", "cucumber"]),
}
AROMA_AXES = {
    "citrusy":    (["lemon", "lime", "orange", "lemongrass", "yuzu"], ["beef", "potato"]),
    "herbal":     (["mint", "basil", "coriander", "parsley", "dill"], ["sugar", "beef"]),
    "woody/warm": (["rosemary", "thyme", "sage", "cinnamon", "clove"], ["lettuce"]),
    "earthy":     (["mushroom", "beet", "cumin", "turmeric", "potato"], ["lemon", "sugar"]),
}
NUTRITION_AXES = {
    "high-protein": (["chicken", "tofu", "lentil", "chickpea", "egg", "edamame"], ["lettuce", "cucumber"]),
    "high-fiber":   (["lentil", "black_bean", "oat", "broccoli", "chickpea", "barley"], ["cream", "butter"]),
    "lighter":      (["lettuce", "cucumber", "celery", "spinach", "zucchini"], ["butter", "cream", "cheese", "bacon"]),
}
PROCESSING = ["whole & fresh", "more processed"]  # from the nova_level poles
_AXIS_CACHE = {}  # (id(model), spec) -> unit direction


def _exemplar_axis(m, hi, lo):
    h = _unit(np.mean([m.vec(x) for x in hi if x in m.vocab], axis=0))
    los = [m.vec(x) for x in lo if x in m.vocab]
    return _unit(h - _unit(np.mean(los, axis=0))) if los else h


def direction_for(m, spec):
    """Resolve a steer spec string to a unit direction vector (or None)."""
    if not spec:
        return None
    ck = (id(m), spec)
    if ck in _AXIS_CACHE:
        return _AXIS_CACHE[ck]
    kind, _, val = spec.partition(":")
    d = None
    if kind == "cuisine" and spec in m.supervised_poles:
        d = _unit(m.supervised_poles[spec])
    elif kind == "flavor" and val in FLAVOR_AXES:
        d = _exemplar_axis(m, *FLAVOR_AXES[val])
    elif kind == "aroma" and val in AROMA_AXES:
        d = _exemplar_axis(m, *AROMA_AXES[val])
    elif kind == "nutrition" and val in NUTRITION_AXES:
        d = _exemplar_axis(m, *NUTRITION_AXES[val])
    elif kind == "processing":
        poles = [_unit(m.supervised_poles[k]) for k in m.supervised_poles
                 if k.startswith("nova_level/")]
        if poles:
            d = _unit(np.mean(poles, axis=0))
            if val == "whole":          # steer AWAY from processed
                d = -d
    elif kind == "factor":
        for md in m.modes:
            if md.mode_id == val:
                d = _unit(md.pole)
                break
    _AXIS_CACHE[ck] = d
    return d


def _rotate(v, d, theta_deg):
    """Rotate unit vector v toward unit direction d by theta degrees on the sphere."""
    d_perp = d - (d @ v) * v
    n = np.linalg.norm(d_perp)
    if n < 1e-9:
        return v
    d_perp /= n
    t = np.deg2rad(float(theta_deg))
    return _unit(np.cos(t) * v + np.sin(t) * d_perp)


def query_vector(m, seeds, steers=(), theta_deg=30):
    """Blend seed(s) into a base point, then rotate toward the combined steer."""
    base = _unit(np.mean([m.vec(s) for s in seeds if s in m.vocab], axis=0))
    dirs = [d for d in (direction_for(m, s) for s in steers) if d is not None]
    if dirs and theta_deg > 0:
        return _rotate(base, _unit(np.mean(dirs, axis=0)), theta_deg)
    return base


# --- Diet excludes (token-level so 'egg' never matches 'eggplant') --------
ANIMAL = {"chicken", "beef", "pork", "lamb", "turkey", "duck", "bacon", "ham",
          "sausage", "prosciutto", "pancetta", "veal", "meat", "shrimp", "prawn",
          "salmon", "tuna", "cod", "fish", "crab", "squid", "octopus", "clam",
          "mussel", "scallop", "anchovy", "oyster", "seafood", "egg", "milk",
          "cheese", "butter", "cream", "yogurt", "paneer", "ghee", "honey", "lard"}
DAIRY = {"milk", "cheese", "butter", "cream", "yogurt", "paneer", "ghee"}
NUTS = {"peanut", "almond", "cashew", "walnut", "pecan", "hazelnut", "pistachio",
        "macadamia", "nut", "pine"}
PLANT_PROTEIN = ["tofu", "tempeh", "lentil", "chickpea", "black_bean",
                 "kidney_bean", "edamame"]


def _excluded(name, excludes):
    toks = set(name.split("_"))
    if "vegan" in excludes and toks & ANIMAL:
        return True
    if "no_dairy" in excludes and toks & DAIRY:
        return True
    if "no_nuts" in excludes and toks & NUTS:
        return True
    return False


def _is_sauce(name):
    return any(t in name for t in ("sauce", "vinegar", "dressing", "paste",
                                   "miso", "tahini", "yogurt", "pesto",
                                   "chutney", "salsa", "hummus", "aioli")) \
        and name not in SKIP_SAUCE


def pick_ingredients(m, seeds, n=4, steers=(), theta_deg=30, veg_only=False,
                     excludes=()):
    """Seed(s) + companions nearest the (steered) query vector, with filters."""
    if isinstance(seeds, str):
        seeds = [seeds]
    q = query_vector(m, seeds, steers=steers, theta_deg=theta_deg)
    sims = m.E @ q
    order = np.argsort(-sims)
    chosen = list(seeds)
    for i in order:
        name = m.itos[int(i)]
        if name in chosen or _is_sauce(name):
            continue
        if any(t in name for t in ("stock", "broth")) or name == "water":
            continue
        if any(name in c or c in name for c in chosen):       # de-dupe variants
            continue
        if veg_only and classify(m, name) != "Vegetable":
            continue
        if _excluded(name, excludes):
            continue
        chosen.append(name)
        if len(chosen) >= n:
            break
    return chosen


FECULENTS = ["rice", "brown_rice", "basmati_rice", "pasta", "noodle",
             "rice_noodle", "egg_noodle", "potato", "sweet_potato", "bread",
             "naan", "tortilla", "quinoa", "couscous", "barley", "bulgur",
             "polenta", "gnocchi", "farro", "millet", "vermicelli"]
PROTEINS = ["chicken", "beef", "pork", "lamb", "turkey", "duck", "bacon",
            "ham", "sausage", "tofu", "tempeh", "paneer", "egg", "shrimp",
            "salmon", "tuna", "cod", "fish", "crab", "squid", "scallop",
            "lentil", "chickpea", "black_bean", "kidney_bean", "edamame"]


def _best_fit(m, ingredients, pool, excludes=()):
    vecs = [m.vec(i) for i in ingredients if i in m.vocab]
    centroid = _unit(np.mean(vecs, axis=0))
    pool = [x for x in pool if x in m.vocab and x not in ingredients
            and not _excluded(x, excludes)]
    return max(pool, key=lambda x: float(centroid @ m.vec(x))) if pool else None


def best_feculent(m, ingredients, excludes=()):
    return _best_fit(m, ingredients, FECULENTS, excludes)


def best_protein(m, ingredients, excludes=()):
    pool = PLANT_PROTEIN if "vegan" in excludes else PROTEINS
    return _best_fit(m, ingredients, pool, excludes)


def best_sauce(m, ingredients, want=None, excludes=()):
    """Sauce closest to the centroid. want='raw'|'cook'|None filters the pool."""
    centroid = _unit(np.mean([m.vec(i) for i in ingredients if i in m.vocab], axis=0))
    best, best_sim = None, -1.0
    for name in m.vocab:
        if not _is_sauce(name) or _excluded(name, excludes):
            continue
        raw = any(t in name for t in RAW_MARKERS)
        if want == "raw" and not raw:
            continue
        if want == "cook" and raw:
            continue
        sim = float(centroid @ m.vec(name))
        if sim > best_sim:
            best, best_sim = name, sim
    return best, best_sim


def method_for(sauce):
    if sauce and any(t in sauce for t in RAW_MARKERS):
        return "chop & salad", "Toss raw, dress, rest 10 min so flavours meld."
    return "chop & cook", "Saute the chopped veg hot 6-8 min, stir the sauce in to finish."


def pretty(name):
    return name.replace("_", " ")


def make_recipe(m, seed, seed2=None, n=4, cuisine=None, flavor=None, aroma=None,
                nutrition=None, processing=None, factor=None, intensity=30,
                method="auto", veg_only=False, protein=False, feculent=False,
                excludes=()):
    """Return a recipe dict. Steer params take human values (e.g. flavor='spicier',
    processing='whole & fresh'); intensity is the SLERP angle in degrees."""
    seeds = [seed] + ([seed2] if seed2 and seed2 in m.vocab else [])
    steers = []
    if cuisine:
        steers.append(cuisine if cuisine.startswith("cuisine:") else f"cuisine:{cuisine}")
    if flavor:
        steers.append(f"flavor:{flavor}")
    if aroma:
        steers.append(f"aroma:{aroma}")
    if nutrition:
        steers.append(f"nutrition:{nutrition}")
    if processing:
        steers.append("processing:" + ("whole" if processing.startswith("whole") else "processed"))
    if factor:
        steers.append(factor if factor.startswith("factor:") else f"factor:{factor}")

    ings = pick_ingredients(m, seeds, n=n, steers=steers, theta_deg=intensity,
                            veg_only=veg_only, excludes=excludes)
    fec = best_feculent(m, ings, excludes) if feculent else None
    pro = best_protein(m, ings, excludes) if protein else None
    want = {"salad": "raw", "cook": "cook"}.get(method)
    sauce, sim = best_sauce(m, ings, want=want, excludes=excludes)
    if method == "salad":
        mlabel, how = "chop & salad", "Toss raw, dress, rest 10 min so flavours meld."
    elif method == "cook":
        mlabel, how = "chop & cook", "Saute the chopped veg hot 6-8 min, stir the sauce in to finish."
    else:
        mlabel, how = method_for(sauce)

    title = f"{pretty(seed).title()}"
    if len(seeds) > 1:
        title += f" & {pretty(seed2).title()}"
    title += f" {mlabel.split(' & ')[1]}"
    tags = [s.split(":")[1].replace("_", " ") for s in (cuisine,) if s] \
        + [x for x in (flavor, aroma, nutrition, processing) if x]
    if tags:
        title += f" ({', '.join(tags)})"

    extra = ""
    if pro:
        extra += f" Add {pretty(pro)}."
    if fec:
        extra += f" Serve over {pretty(fec)}."
    return {
        "title": title,
        "method": mlabel,
        "ingredients": [pretty(i) for i in ings],
        "protein": pretty(pro) if pro else None,
        "feculent": pretty(fec) if fec else None,
        "sauce": pretty(sauce) if sauce else "—",
        "fit": round(sim, 2),
        "steps": f"Chop everything to even pieces. {how}{extra}",
        "cuisine": cuisine,
    }


def build(m, seed, **kw):
    r = make_recipe(m, seed, **kw)
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
            sys.exit(f"'{seed}' not in vocab. Try carrot, cabbage, zucchini, beet, fennel.")
        build(m, seed, cuisine=cuisine)
    else:
        build(m, "carrot")
        build(m, "carrot", flavor="spicier", intensity=40)
        build(m, "zucchini", aroma="herbal", cuisine="cuisine:Mediterranean")
        build(m, "beet", nutrition="high-fiber", veg_only=True, protein=True, feculent=True)
        build(m, "cabbage", seed2="ginger", cuisine="cuisine:East_Asian", excludes=["vegan"], protein=True)
        build(m, "carrot", processing="whole & fresh", method="salad")


if __name__ == "__main__":
    main()
