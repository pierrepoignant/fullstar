#!/usr/bin/env python3
"""Mini Flask app for Epicure vegetable-chopper recipes (arXiv:2605.22391).

Loads the embedding model once at startup, then serves a small web UI plus a
JSON API on top of the geometry helpers in chopper_recipes.py.

    ./venv/bin/python app.py      # -> http://127.0.0.1:5001
                                  # (5000 is taken by macOS AirPlay Receiver)
"""
from flask import Flask, jsonify, render_template_string, request

from chopper_recipes import load_model, make_recipe

app = Flask(__name__)

print("Loading Epicure model (first run downloads ~a few MB from HF)...")
MODEL = load_model()
VEG = sorted(v for v in MODEL.vocab if not any(
    t in v for t in ("sauce", "vinegar", "dressing", "paste", "_oil")))
CUISINES = [""] + MODEL.list_supervised_poles(prefix="cuisine:")
print(f"Ready: {MODEL} | {len(VEG)} selectable items, {len(CUISINES)-1} cuisines")

PAGE = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Fullstar Recipe Studio</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&display=swap" rel="stylesheet">
<style>
 :root{
   --coral:#ff585d; --coral-dark:#e0454a; --ink:#1f1e1e; --muted:#6b6b70;
   --bg:#f9f9fc; --line:#eceef3; --white:#fff;
 }
 *{box-sizing:border-box}
 body{font-family:"DM Sans",system-ui,sans-serif;margin:0;background:var(--bg);color:var(--ink);
   -webkit-font-smoothing:antialiased;line-height:1.5}
 .wrap{max-width:760px;margin:0 auto;padding:0 20px}
 /* top bar */
 .nav{background:var(--white);border-bottom:1px solid var(--line)}
 .nav .wrap{display:flex;align-items:center;gap:10px;height:60px}
 .brand{font-weight:700;font-size:1.15rem;letter-spacing:-.02em}
 .brand .dot{color:var(--coral)}
 .nav .pill{margin-left:auto;font-size:.72rem;color:var(--muted);background:var(--bg);
   border:1px solid var(--line);border-radius:999px;padding:5px 12px}
 /* hero */
 .hero{background:var(--white);border-bottom:1px solid var(--line);text-align:center;padding:48px 0 40px}
 .hero h1{margin:0 0 10px;font-size:2.3rem;line-height:1.1;letter-spacing:-.03em;font-weight:700}
 .hero h1 .accent{color:var(--coral)}
 .hero p{margin:0 auto;max-width:460px;color:var(--muted);font-size:1.02rem}
 .trust{margin-top:14px;font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
 /* form card */
 .panel{background:var(--white);border:1px solid var(--line);border-radius:18px;
   padding:26px;margin:28px 0;box-shadow:0 6px 24px rgba(31,30,30,.05)}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 .field{flex:1;min-width:150px}
 label.lbl{display:block;font-size:.74rem;font-weight:500;text-transform:uppercase;
   letter-spacing:.06em;color:var(--muted);margin-bottom:7px}
 input[type=text],select{width:100%;padding:12px 14px;font:inherit;font-size:.98rem;color:var(--ink);
   background:var(--bg);border:1.5px solid var(--line);border-radius:11px;outline:none;transition:.15s}
 input[type=text]:focus,select:focus{border-color:var(--coral);background:var(--white)}
 .toggles{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0 4px}
 .chip{display:inline-flex;align-items:center;gap:8px;cursor:pointer;user-select:none;
   background:var(--bg);border:1.5px solid var(--line);border-radius:999px;
   padding:9px 16px;font-size:.9rem;font-weight:500;transition:.15s}
 .chip:hover{border-color:#d9dbe3}
 .chip input{accent-color:var(--coral);width:16px;height:16px;margin:0}
 .chip.on{background:#fff0f0;border-color:var(--coral);color:var(--coral-dark)}
 .btn{margin-top:22px;width:100%;padding:15px;font:inherit;font-weight:700;font-size:1.02rem;
   color:var(--white);background:var(--coral);border:none;border-radius:12px;cursor:pointer;
   transition:.15s;letter-spacing:.01em}
 .btn:hover{background:var(--coral-dark)}
 /* recipe card */
 .recipe{background:var(--white);border:1px solid var(--line);border-radius:18px;overflow:hidden;
   margin-bottom:28px;box-shadow:0 6px 24px rgba(31,30,30,.05)}
 .recipe .head{padding:24px 26px;border-bottom:1px solid var(--line);
   display:flex;align-items:center;justify-content:space-between;gap:12px}
 .recipe h2{margin:0;font-size:1.45rem;letter-spacing:-.02em}
 .badge{flex:none;background:var(--coral);color:#fff;font-size:.74rem;font-weight:700;
   text-transform:uppercase;letter-spacing:.05em;border-radius:999px;padding:6px 13px}
 .recipe .body{padding:8px 26px 24px}
 .line{display:flex;gap:14px;padding:14px 0;border-bottom:1px solid var(--line)}
 .line:last-child{border-bottom:none}
 .line .k{flex:none;width:96px;font-size:.74rem;font-weight:700;text-transform:uppercase;
   letter-spacing:.06em;color:var(--muted);padding-top:2px}
 .line .v{font-size:1.05rem}
 .v .sauce{color:var(--coral-dark);font-weight:700}
 .fit{color:var(--muted);font-size:.8rem;font-weight:400}
 .steps{color:var(--muted);font-size:.97rem}
 .err{background:#fff0f0;border:1px solid var(--coral);color:var(--coral-dark);
   border-radius:12px;padding:14px 16px;margin-bottom:24px;font-size:.95rem}
 .foot{color:var(--muted);font-size:.82rem;text-align:center;padding:8px 0 40px}
 .foot code{background:var(--white);border:1px solid var(--line);border-radius:6px;padding:2px 7px;font-size:.8rem}
 .foot .repr{display:block;margin-top:6px;font-size:.72rem;opacity:.7}
</style></head><body>

<div class="nav"><div class="wrap">
  <span class="brand">Full<span class="dot">star</span> Recipe Studio</span>
  <span class="pill">powered by Epicure embeddings</span>
</div></div>

<div class="hero"><div class="wrap">
  <h1>Cooking Made <span class="accent">Simple</span></h1>
  <p>Chop a few ingredients, add a sauce, and cook. Pairings picked by a food-flavor AI &mdash; built for your vegetable chopper.</p>
  <div class="trust">Designed for real home cooks</div>
</div></div>

<div class="wrap">
<form class="panel" method="get" action="/">
  <div class="row">
    <div class="field" style="flex:2">
      <label class="lbl" for="seed">Star ingredient</label>
      <input type="text" id="seed" name="seed" list="veg" value="{{seed}}" placeholder="e.g. carrot, cabbage, beet" autocomplete="off">
      <datalist id="veg">{% for v in veg %}<option value="{{v}}">{% endfor %}</datalist>
    </div>
    <div class="field">
      <label class="lbl" for="n">Ingredients</label>
      <select id="n" name="n">{% for i in [3,4,5] %}<option value="{{i}}" {{'selected' if i==n}}>{{i}}</option>{% endfor %}</select>
    </div>
    <div class="field">
      <label class="lbl" for="cuisine">Cuisine</label>
      <select id="cuisine" name="cuisine"><option value="">Any</option>
        {% for c in cuisines if c %}<option value="{{c}}" {{'selected' if c==cuisine}}>{{c.split(':')[1].replace('_',' ')}}</option>{% endfor %}
      </select>
    </div>
  </div>
  <div class="toggles">
    <label class="chip {{'on' if veg_only}}"><input type="checkbox" name="veg_only" value="1" {{'checked' if veg_only}}> Vegetables only</label>
    <label class="chip {{'on' if protein}}"><input type="checkbox" name="protein" value="1" {{'checked' if protein}}> Add protein</label>
    <label class="chip {{'on' if feculent}}"><input type="checkbox" name="feculent" value="1" {{'checked' if feculent}}> Add a feculent</label>
  </div>
  <button class="btn" type="submit">Generate recipe</button>
</form>

{% if error %}<div class="err">{{error}}</div>{% endif %}

{% if recipe %}
<div class="recipe">
  <div class="head">
    <h2>{{recipe.title}}</h2>
    <span class="badge">{{recipe.method}}</span>
  </div>
  <div class="body">
    <div class="line"><div class="k">Chop</div><div class="v">{{recipe.ingredients|join(', ')}}</div></div>
    {% if recipe.protein %}<div class="line"><div class="k">Protein</div><div class="v">{{recipe.protein}}</div></div>{% endif %}
    {% if recipe.feculent %}<div class="line"><div class="k">Serve over</div><div class="v">{{recipe.feculent}}</div></div>{% endif %}
    <div class="line"><div class="k">Sauce</div><div class="v"><span class="sauce">{{recipe.sauce}}</span> <span class="fit">fit {{recipe.fit}}</span></div></div>
    <div class="line"><div class="k">Method</div><div class="v steps">{{recipe.steps}}</div></div>
  </div>
</div>
{% endif %}

<div class="foot">
  JSON API &middot; <code>/api/recipe?seed=carrot&amp;n=4&amp;protein=1&amp;feculent=1</code>
  <span class="repr">{{repr}}</span>
</div>
</div>
</body></html>
"""


def _recipe_from_args(args):
    seed = (args.get("seed") or "").strip().lower().replace(" ", "_")
    n = max(2, min(4, int(args.get("n", 4)) - 1))  # companions = n-1; total stays 3..5
    cuisine = args.get("cuisine") or None
    veg_only = args.get("veg_only") in ("1", "true", "on")
    feculent = args.get("feculent") in ("1", "true", "on")
    protein = args.get("protein") in ("1", "true", "on")
    if not seed:
        return None, None
    if seed not in MODEL.vocab:
        return None, f"'{seed.replace('_',' ')}' is not in the vocabulary. Try one from the list."
    return make_recipe(MODEL, seed, cuisine=cuisine, n=n,
                       veg_only=veg_only, feculent=feculent, protein=protein), None


@app.route("/")
def home():
    recipe, error = _recipe_from_args(request.args)
    return render_template_string(
        PAGE, recipe=recipe, error=error, veg=VEG, cuisines=CUISINES,
        seed=request.args.get("seed", ""), n=int(request.args.get("n", 4)),
        cuisine=request.args.get("cuisine", ""), repr=str(MODEL),
        veg_only=request.args.get("veg_only") in ("1", "true", "on"),
        feculent=request.args.get("feculent") in ("1", "true", "on"),
        protein=request.args.get("protein") in ("1", "true", "on"))


@app.route("/api/recipe")
def api_recipe():
    recipe, error = _recipe_from_args(request.args)
    if error:
        return jsonify(error=error), 400
    if not recipe:
        return jsonify(error="missing ?seed="), 400
    return jsonify(recipe)


if __name__ == "__main__":
    app.run(debug=False, port=5001)
