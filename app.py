#!/usr/bin/env python3
"""Fullstar Recipe Studio — Flask UI over Epicure embeddings (arXiv:2605.22391).

Loads the model(s) and serves a Fullstar-styled web UI plus a JSON API exposing
the full steering surface: seed blending, sibling model, cuisine / flavor / aroma
/ nutrition / processing / factor-mode steers, intensity, method, diet excludes.

    ./venv/bin/python app.py      # -> http://127.0.0.1:5001
                                  # (5000 is taken by macOS AirPlay Receiver)
"""
from flask import Flask, jsonify, render_template_string, request

import chopper_recipes as cr

app = Flask(__name__)

print("Loading Epicure (core)...")
MODEL = cr.get_model("core")
VEG = sorted(v for v in MODEL.vocab if not any(
    t in v for t in ("sauce", "vinegar", "dressing", "paste", "_oil")))
CUISINES = MODEL.list_supervised_poles(prefix="cuisine:")
FACTORS = MODEL.list_modes(kind="factor")          # [(mode_id, label), ...]
FLAVORS = list(cr.FLAVOR_AXES)
AROMAS = list(cr.AROMA_AXES)
NUTRITION = list(cr.NUTRITION_AXES)
PROCESSING = cr.PROCESSING
print(f"Ready: {MODEL} | {len(VEG)} items, {len(CUISINES)} cuisines, {len(FACTORS)} factor modes")

PAGE = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Fullstar Recipe Studio</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&display=swap" rel="stylesheet">
<style>
 :root{--coral:#ff585d;--coral-dark:#e0454a;--ink:#1f1e1e;--muted:#6b6b70;
   --bg:#f9f9fc;--line:#eceef3;--white:#fff}
 *{box-sizing:border-box}
 body{font-family:"DM Sans",system-ui,sans-serif;margin:0;background:var(--bg);color:var(--ink);
   -webkit-font-smoothing:antialiased;line-height:1.5}
 .wrap{max-width:820px;margin:0 auto;padding:0 20px}
 .nav{background:var(--white);border-bottom:1px solid var(--line)}
 .nav .wrap{display:flex;align-items:center;gap:12px;height:62px}
 .nav .logo{height:26px;width:auto;display:block}
 .brand{font-weight:500;font-size:.95rem;letter-spacing:-.01em;color:var(--muted)}
 .nav .pill{margin-left:auto;font-size:.72rem;color:var(--muted);background:var(--bg);
   border:1px solid var(--line);border-radius:999px;padding:5px 12px}
 .hero{position:relative;text-align:center;padding:72px 0 64px;color:var(--white);
   border-bottom:1px solid var(--line);
   background:linear-gradient(rgba(31,30,30,.42),rgba(31,30,30,.58)),
     url('/static/background.webp') center/cover no-repeat}
 .hero h1{margin:0 0 12px;font-size:2.5rem;line-height:1.1;letter-spacing:-.03em;font-weight:700;
   text-shadow:0 2px 14px rgba(0,0,0,.35)}
 .hero h1 .accent{color:var(--coral)}
 .hero p{margin:0 auto;max-width:500px;color:rgba(255,255,255,.94);font-size:1.05rem;
   text-shadow:0 1px 10px rgba(0,0,0,.4)}
 .trust{margin-top:14px;font-size:.76rem;color:rgba(255,255,255,.82);
   text-transform:uppercase;letter-spacing:.08em;text-shadow:0 1px 8px rgba(0,0,0,.4)}
 .panel{background:var(--white);border:1px solid var(--line);border-radius:18px;
   padding:24px;margin:26px 0;box-shadow:0 6px 24px rgba(31,30,30,.05)}
 .sec{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
   color:var(--coral-dark);margin:6px 0 12px}
 .sec.t2{margin-top:24px;border-top:1px solid var(--line);padding-top:20px}
 .row{display:flex;gap:14px;flex-wrap:wrap}
 .field{flex:1;min-width:140px}
 label.lbl{display:block;font-size:.72rem;font-weight:500;text-transform:uppercase;
   letter-spacing:.05em;color:var(--muted);margin-bottom:6px}
 input[type=text],select{width:100%;padding:11px 13px;font:inherit;font-size:.95rem;color:var(--ink);
   background:var(--bg);border:1.5px solid var(--line);border-radius:11px;outline:none;transition:.15s}
 input[type=text]:focus,select:focus{border-color:var(--coral);background:var(--white)}
 input[type=range]{width:100%;accent-color:var(--coral)}
 .toggles{display:flex;gap:9px;flex-wrap:wrap;margin-top:4px}
 .chip{display:inline-flex;align-items:center;gap:8px;cursor:pointer;user-select:none;
   background:var(--bg);border:1.5px solid var(--line);border-radius:999px;
   padding:8px 15px;font-size:.88rem;font-weight:500;transition:.15s}
 .chip:hover{border-color:#d9dbe3}
 .chip input{accent-color:var(--coral);width:16px;height:16px;margin:0}
 .chip.on{background:#fff0f0;border-color:var(--coral);color:var(--coral-dark)}
 .seg{display:inline-flex;background:var(--bg);border:1.5px solid var(--line);border-radius:11px;overflow:hidden}
 .seg label{padding:9px 16px;font-size:.88rem;font-weight:500;cursor:pointer}
 .seg input{display:none}
 .seg input:checked+span{color:var(--white)}
 .seg label:has(input:checked){background:var(--coral)}
 .btnrow{display:flex;gap:12px;margin-top:24px}
 .btn{flex:1;padding:15px;font:inherit;font-weight:700;font-size:1rem;color:var(--white);
   background:var(--coral);border:none;border-radius:12px;cursor:pointer;transition:.15s}
 .btn:hover{background:var(--coral-dark)}
 .btn.alt{flex:none;background:var(--white);color:var(--ink);border:1.5px solid var(--line)}
 .btn.alt:hover{border-color:var(--coral);color:var(--coral-dark)}
 .recipe{background:var(--white);border:1px solid var(--line);border-radius:18px;overflow:hidden;
   margin-bottom:26px;box-shadow:0 6px 24px rgba(31,30,30,.05)}
 .recipe .head{padding:22px 26px;border-bottom:1px solid var(--line);
   display:flex;align-items:center;justify-content:space-between;gap:12px}
 .recipe h2{margin:0;font-size:1.4rem;letter-spacing:-.02em}
 .badge{flex:none;background:var(--coral);color:#fff;font-size:.72rem;font-weight:700;
   text-transform:uppercase;letter-spacing:.05em;border-radius:999px;padding:6px 13px}
 .recipe .body{padding:6px 26px 22px}
 .line{display:flex;gap:14px;padding:13px 0;border-bottom:1px solid var(--line)}
 .line:last-child{border-bottom:none}
 .line .k{flex:none;width:96px;font-size:.72rem;font-weight:700;text-transform:uppercase;
   letter-spacing:.05em;color:var(--muted);padding-top:2px}
 .line .v{font-size:1.04rem}
 .v .sauce{color:var(--coral-dark);font-weight:700}
 .fit{color:var(--muted);font-size:.8rem;font-weight:400}
 .steps{color:var(--muted);font-size:.96rem}
 .err{background:#fff0f0;border:1px solid var(--coral);color:var(--coral-dark);
   border-radius:12px;padding:14px 16px;margin-bottom:24px;font-size:.95rem}
</style></head><body>

<div class="nav"><div class="wrap">
  <img class="logo" src="/static/fullstar.svg" alt="Fullstar">
  <span class="brand">Recipe Studio</span>
  <span class="pill">powered by Epicure embeddings</span>
</div></div>

<div class="hero"><div class="wrap">
  <h1>Cooking Made <span class="accent">Simple</span></h1>
  <p>Chop a few ingredients, add a sauce, and cook. Pairings picked by a food-flavor AI &mdash; built for your vegetable chopper.</p>
  <div class="trust">Designed for real home cooks</div>
</div></div>

<div class="wrap">
<form class="panel" method="get" action="/" id="f">
  <div class="sec">The basics</div>
  <div class="row">
    <div class="field" style="flex:2">
      <label class="lbl" for="seed">Star ingredient</label>
      <input type="text" id="seed" name="seed" list="veg" value="{{seed}}" placeholder="e.g. carrot" autocomplete="off">
    </div>
    <div class="field" style="flex:2">
      <label class="lbl" for="seed2">Blend with (optional)</label>
      <input type="text" id="seed2" name="seed2" list="veg" value="{{seed2}}" placeholder="e.g. ginger" autocomplete="off">
    </div>
    <div class="field">
      <label class="lbl" for="n">Ingredients</label>
      <select id="n" name="n">{% for i in range(3,11) %}<option value="{{i}}" {{'selected' if i==n}}>{{i}}</option>{% endfor %}</select>
    </div>
    <div class="field">
      <label class="lbl" for="model">Model</label>
      <select id="model" name="model">
        {% for k,desc in models %}<option value="{{k}}" {{'selected' if k==model}}>{{desc}}</option>{% endfor %}
      </select>
    </div>
  </div>
  <datalist id="veg">{% for v in veg %}<option value="{{v}}">{% endfor %}</datalist>

  <div class="sec t2">Steer the flavor</div>
  <div class="row">
    <div class="field"><label class="lbl">Cuisine</label>
      <select name="cuisine"><option value="">Any</option>
      {% for c in cuisines %}<option value="{{c}}" {{'selected' if c==cuisine}}>{{c.split(':')[1].replace('_',' ')}}</option>{% endfor %}</select></div>
    <div class="field"><label class="lbl">Taste</label>
      <select name="flavor"><option value="">—</option>
      {% for x in flavors %}<option value="{{x}}" {{'selected' if x==flavor}}>{{x}}</option>{% endfor %}</select></div>
    <div class="field"><label class="lbl">Aroma</label>
      <select name="aroma"><option value="">—</option>
      {% for x in aromas %}<option value="{{x}}" {{'selected' if x==aroma}}>{{x}}</option>{% endfor %}</select></div>
  </div>
  <div class="row" style="margin-top:14px">
    <div class="field"><label class="lbl">Nutrition</label>
      <select name="nutrition"><option value="">—</option>
      {% for x in nutrition %}<option value="{{x}}" {{'selected' if x==nutrition_v}}>{{x}}</option>{% endfor %}</select></div>
    <div class="field"><label class="lbl">Processing</label>
      <select name="processing"><option value="">—</option>
      {% for x in processing %}<option value="{{x}}" {{'selected' if x==processing_v}}>{{x}}</option>{% endfor %}</select></div>
    <div class="field"><label class="lbl">Emergent factor mode</label>
      <select name="factor"><option value="">—</option>
      {% for mid,lab in factors %}<option value="factor:{{mid}}" {{'selected' if ('factor:'+mid)==factor}}>{{lab}}</option>{% endfor %}</select></div>
  </div>
  <div class="field" style="margin-top:16px">
    <label class="lbl" for="intensity">Steer intensity — {{intensity}}&deg; <span class="fit">(subtle &rarr; bold)</span></label>
    <input type="range" id="intensity" name="intensity" min="0" max="60" step="5" value="{{intensity}}"
      oninput="document.getElementById('iv').textContent=this.value">
  </div>

  <div class="sec t2">Add-ons &amp; diet</div>
  <div class="toggles">
    <label class="chip {{'on' if veg_only}}"><input type="checkbox" name="veg_only" value="1" {{'checked' if veg_only}}> Vegetables only</label>
    <label class="chip {{'on' if protein}}"><input type="checkbox" name="protein" value="1" {{'checked' if protein}}> Add protein</label>
    <label class="chip {{'on' if feculent}}"><input type="checkbox" name="feculent" value="1" {{'checked' if feculent}}> Add a feculent</label>
    <label class="chip {{'on' if 'vegan' in excludes}}"><input type="checkbox" name="vegan" value="1" {{'checked' if 'vegan' in excludes}}> Vegan</label>
    <label class="chip {{'on' if 'no_dairy' in excludes}}"><input type="checkbox" name="no_dairy" value="1" {{'checked' if 'no_dairy' in excludes}}> No dairy</label>
    <label class="chip {{'on' if 'no_nuts' in excludes}}"><input type="checkbox" name="no_nuts" value="1" {{'checked' if 'no_nuts' in excludes}}> No nuts</label>
  </div>
  <div style="margin-top:16px">
    <label class="lbl">Method</label>
    <div class="seg">
      {% for val,txt in [('auto','Auto'),('salad','Chop & salad'),('cook','Chop & cook')] %}
      <label><input type="radio" name="method" value="{{val}}" {{'checked' if val==method_v}}><span>{{txt}}</span></label>
      {% endfor %}
    </div>
  </div>

  <div class="btnrow">
    <button class="btn" type="submit">Generate recipe</button>
    <button class="btn alt" type="button" onclick="surprise()">🎲 Surprise me</button>
  </div>
</form>

{% if error %}<div class="err">{{error}}</div>{% endif %}

{% if recipe %}
<div class="recipe">
  <div class="head"><h2>{{recipe.title}}</h2><span class="badge">{{recipe.method}}</span></div>
  <div class="body">
    <div class="line"><div class="k">Chop</div><div class="v">{{recipe.ingredients|join(', ')}}</div></div>
    {% if recipe.protein %}<div class="line"><div class="k">Protein</div><div class="v">{{recipe.protein}}</div></div>{% endif %}
    {% if recipe.feculent %}<div class="line"><div class="k">Serve over</div><div class="v">{{recipe.feculent}}</div></div>{% endif %}
    <div class="line"><div class="k">Sauce</div><div class="v"><span class="sauce">{{recipe.sauce}}</span> <span class="fit">fit {{recipe.fit}}</span></div></div>
    <div class="line"><div class="k">Method</div><div class="v steps">{{recipe.steps}}</div></div>
  </div>
</div>
{% endif %}
</div>

<script>
const VEG={{veg_sample|tojson}}, CUIS={{cuisines|tojson}}, FLAV={{flavors|tojson}};
function pick(a){return a[Math.floor(Math.random()*a.length)]}
function surprise(){
  const f=document.getElementById('f');
  f.seed.value=pick(VEG);
  f.cuisine.value=Math.random()<.6?pick(CUIS):'';
  f.flavor.value=Math.random()<.5?pick(FLAV):'';
  f.intensity.value=[20,30,40,50][Math.floor(Math.random()*4)];
  f.veg_only.checked=Math.random()<.4;
  f.protein.checked=Math.random()<.5;
  f.feculent.checked=Math.random()<.5;
  f.submit();
}
</script>
</body></html>
"""

TRUE = ("1", "true", "on")


def _recipe_from_args(args):
    seed = (args.get("seed") or "").strip().lower().replace(" ", "_")
    seed2 = (args.get("seed2") or "").strip().lower().replace(" ", "_") or None
    sibling = args.get("model", "core")
    m = cr.get_model(sibling if sibling in cr.SIBLINGS else "core")
    try:
        n = max(3, min(10, int(args.get("n", 4))))
    except ValueError:
        n = 4
    try:
        intensity = max(0, min(60, int(args.get("intensity", 30))))
    except ValueError:
        intensity = 30
    excludes = [e for e in ("vegan", "no_dairy", "no_nuts") if args.get(e) in TRUE]
    if not seed:
        return None, None, m
    if seed not in m.vocab:
        return None, f"'{seed.replace('_',' ')}' is not in the vocabulary. Try one from the list.", m
    if seed2 and seed2 not in m.vocab:
        return None, f"'{seed2.replace('_',' ')}' is not in the vocabulary.", m
    recipe = cr.make_recipe(
        m, seed, seed2=seed2, n=n,
        cuisine=args.get("cuisine") or None,
        flavor=args.get("flavor") or None,
        aroma=args.get("aroma") or None,
        nutrition=args.get("nutrition") or None,
        processing=args.get("processing") or None,
        factor=args.get("factor") or None,
        intensity=intensity,
        method=args.get("method", "auto"),
        veg_only=args.get("veg_only") in TRUE,
        protein=args.get("protein") in TRUE,
        feculent=args.get("feculent") in TRUE,
        excludes=excludes,
    )
    return recipe, None, m


@app.route("/")
def home():
    recipe, error, _ = _recipe_from_args(request.args)
    a = request.args
    excludes = [e for e in ("vegan", "no_dairy", "no_nuts") if a.get(e) in TRUE]
    return render_template_string(
        PAGE, recipe=recipe, error=error,
        veg=VEG, veg_sample=VEG[::7][:120], cuisines=CUISINES, factors=FACTORS,
        flavors=FLAVORS, aromas=AROMAS, nutrition=NUTRITION, processing=PROCESSING,
        models=[(k, {"core": "Core (chem + recipe)", "cooc": "Cooc (recipe only)",
                     "chem": "Chem (chemistry only)"}[k]) for k in cr.SIBLINGS],
        seed=a.get("seed", ""), seed2=a.get("seed2", ""),
        n=int(a.get("n", 4)) if a.get("n", "4").isdigit() else 4,
        model=a.get("model", "core"),
        cuisine=a.get("cuisine", ""), flavor=a.get("flavor", ""), aroma=a.get("aroma", ""),
        nutrition_v=a.get("nutrition", ""), processing_v=a.get("processing", ""),
        factor=a.get("factor", ""),
        intensity=int(a.get("intensity", 30)) if a.get("intensity", "30").isdigit() else 30,
        method_v=a.get("method", "auto"),
        veg_only=a.get("veg_only") in TRUE, protein=a.get("protein") in TRUE,
        feculent=a.get("feculent") in TRUE, excludes=excludes)


@app.route("/healthz")
def healthz():
    return jsonify(status="ok", model=str(MODEL))


@app.route("/api/recipe")
def api_recipe():
    recipe, error, _ = _recipe_from_args(request.args)
    if error:
        return jsonify(error=error), 400
    if not recipe:
        return jsonify(error="missing ?seed="), 400
    return jsonify(recipe)


if __name__ == "__main__":
    app.run(debug=False, port=5001)
