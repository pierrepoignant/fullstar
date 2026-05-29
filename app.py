#!/usr/bin/env python3
"""Fullstar Recipe Studio — Flask UI over Epicure embeddings (arXiv:2605.22391).

Loads the model(s) and serves a Fullstar-styled web UI with three tabs:

  1. Design recipe  — steer the embeddings (seed blending, sibling model,
     cuisine / flavor / aroma / nutrition / processing / factor-mode steers,
     intensity, method, diet excludes) and save the result to a recipe book.
  2. View recipes   — the saved recipe book, newest first, paginated to 50.
  3. Fullstar tools — the Fullstar brand story and a link to buy the chopper.

A JSON API (``/api/recipe``) exposes the same steering surface.

    ./venv/bin/python app.py      # -> http://127.0.0.1:5001
                                  # (5000 is taken by macOS AirPlay Receiver)
"""
import os

from flask import (Flask, jsonify, redirect, render_template_string, request,
                   session, url_for)

import chopper_recipes as cr
import db

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-fullstar-not-secret")

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
try:
    db.init_db()
except Exception as e:  # MySQL may not be reachable yet; table is created lazily.
    print(f"DB init deferred ({e.__class__.__name__}): table will be created on first use")
print(f"Ready: {MODEL} | {len(VEG)} items, {len(CUISINES)} cuisines, {len(FACTORS)} factor modes")

AMAZON_URL = "https://www.amazon.com/Vegetable-Chopper-Spiralizer-Slicer-Choppers/dp/B0764HS4SL"
PER_PAGE = 50

# --- Shared chrome: head/styles, top nav, hero banner, tab bar -------------
HEAD = """
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
 a{color:inherit}
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
 .hero .powered{margin:10px auto 0;max-width:520px;color:rgba(255,255,255,.92);font-size:.92rem;
   text-shadow:0 1px 8px rgba(0,0,0,.4)}
 .hero .powered b{color:var(--coral);font-weight:700}
 .hero .powered a{color:var(--coral);font-weight:700;text-decoration:underline;text-underline-offset:2px}
 .trust{margin-top:14px;font-size:.76rem;color:rgba(255,255,255,.82);
   text-transform:uppercase;letter-spacing:.08em;text-shadow:0 1px 8px rgba(0,0,0,.4)}
 .tabs{background:var(--white);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
 .tabs .wrap{display:flex;gap:4px}
 .tabs a{padding:15px 18px;font-size:.9rem;font-weight:500;color:var(--muted);
   text-decoration:none;border-bottom:3px solid transparent;margin-bottom:-1px;transition:.15s}
 .tabs a:hover{color:var(--ink)}
 .tabs a.active{color:var(--coral-dark);border-bottom-color:var(--coral);font-weight:700}
 .panel{background:var(--white);border:1px solid var(--line);border-radius:18px;
   padding:24px;margin:26px 0;box-shadow:0 6px 24px rgba(31,30,30,.05)}
 .sec{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
   color:var(--coral-dark);margin:6px 0 12px}
 .sec.t2{margin-top:24px;border-top:1px solid var(--line);padding-top:20px}
 .row{display:flex;gap:14px;flex-wrap:wrap}
 .field{flex:1;min-width:140px}
 label.lbl{display:block;font-size:.72rem;font-weight:500;text-transform:uppercase;
   letter-spacing:.05em;color:var(--muted);margin-bottom:6px}
 input[type=text],input[type=email],select{width:100%;padding:11px 13px;font:inherit;font-size:.95rem;
   color:var(--ink);background:var(--bg);border:1.5px solid var(--line);border-radius:11px;outline:none;transition:.15s}
 input[type=text]:focus,input[type=email]:focus,select:focus{border-color:var(--coral);background:var(--white)}
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
 .ok{background:#eafaf0;border:1px solid #57c98a;color:#1c7a47;border-radius:12px;
   padding:14px 16px;margin:0 0 20px;font-size:.95rem;font-weight:500}
 /* saved recipe book */
 .saved{border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:16px;background:var(--white)}
 .saved-head{display:flex;align-items:center;justify-content:space-between;gap:12px}
 .saved-head h3{margin:0;font-size:1.18rem;letter-spacing:-.02em}
 .saved-meta{color:var(--muted);font-size:.78rem;margin:4px 0 6px}
 .empty{color:var(--muted);font-size:.95rem}
 .empty a{color:var(--coral-dark);font-weight:700}
 .filterbar{display:flex;gap:8px;margin-bottom:18px}
 .fbtn{font-size:.85rem;font-weight:600;color:var(--muted);text-decoration:none;
   background:var(--bg);border:1.5px solid var(--line);border-radius:999px;padding:8px 16px;transition:.15s}
 .fbtn:hover{border-color:#d9dbe3;color:var(--ink)}
 .fbtn.on{background:#fff0f0;border-color:var(--coral);color:var(--coral-dark)}
 .fbtn.disabled{opacity:.5;cursor:not-allowed}
 .pager{display:flex;align-items:center;justify-content:center;gap:14px;margin-top:20px;font-size:.9rem}
 .pager a{color:var(--coral-dark);text-decoration:none;font-weight:700;
   border:1.5px solid var(--line);border-radius:10px;padding:8px 14px}
 .pager a:hover{border-color:var(--coral)}
 .pager .muted{color:var(--muted)}
 /* about / fullstar tools */
 .about h2{font-size:1.7rem;letter-spacing:-.02em;margin:0 0 6px}
 .about .lede{color:var(--muted);font-size:1.08rem;margin:0 0 16px}
 .about h3{font-size:.95rem;text-transform:uppercase;letter-spacing:.06em;color:var(--coral-dark);margin:24px 0 8px}
 .about p{margin:0 0 12px;line-height:1.6}
 .about ul{margin:0 0 12px;padding-left:20px}
 .about li{margin:6px 0}
 .stats{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0 4px}
 .stat{flex:1;min-width:130px;background:var(--bg);border:1px solid var(--line);border-radius:14px;padding:16px;text-align:center}
 .stat b{display:block;font-size:1.5rem;color:var(--coral-dark)}
 .stat span{font-size:.74rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
 .cta{display:inline-block;margin-top:10px;padding:14px 22px;background:var(--coral);color:#fff;
   font-weight:700;border-radius:12px;text-decoration:none;transition:.15s}
 .cta:hover{background:var(--coral-dark)}
 .cta-note{display:block;margin-top:8px;color:var(--muted);font-size:.78rem}
</style></head><body>

<div class="nav"><div class="wrap">
  <img class="logo" src="/static/fullstar.svg" alt="Fullstar">
  <span class="brand">Recipe Studio</span>
  <span class="pill">powered by Epicure embeddings</span>
</div></div>

<div class="hero"><div class="wrap">
  <h1>Cooking Made <span class="accent">Simple</span></h1>
  <p>Chop a few ingredients, add a sauce, and cook. Pairings picked by a food-flavor AI &mdash; built for your vegetable chopper.</p>
  <p class="powered">Powered by <a href="https://huggingface.co/Kaikaku/epicure-core" target="_blank" rel="noopener noreferrer">Epicure</a>, an AI food model trained on <b>4.14M recipes</b>.</p>
  <div class="trust">Designed for real home cooks</div>
</div></div>
"""

TAIL = "</body></html>"

_TABS = [("design", "Design recipe", "/"),
         ("recipes", "View recipes", "/recipes"),
         ("about", "Fullstar tools", "/about")]


def render_page(active, content, **ctx):
    tabs = "".join(
        f'<a class="{"active" if key == active else ""}" href="{href}">{label}</a>'
        for key, label, href in _TABS)
    tabbar = f'<div class="tabs"><div class="wrap">{tabs}</div></div>'
    full = HEAD + tabbar + '<div class="wrap">' + content + "</div>" + TAIL
    return render_template_string(full, **ctx)


# --- Tab 1: Design recipe --------------------------------------------------
DESIGN_BODY = """
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
      <select name="flavor"><option value="">&mdash;</option>
      {% for x in flavors %}<option value="{{x}}" {{'selected' if x==flavor}}>{{x}}</option>{% endfor %}</select></div>
    <div class="field"><label class="lbl">Aroma</label>
      <select name="aroma"><option value="">&mdash;</option>
      {% for x in aromas %}<option value="{{x}}" {{'selected' if x==aroma}}>{{x}}</option>{% endfor %}</select></div>
  </div>
  <div class="row" style="margin-top:14px">
    <div class="field"><label class="lbl">Nutrition</label>
      <select name="nutrition"><option value="">&mdash;</option>
      {% for x in nutrition %}<option value="{{x}}" {{'selected' if x==nutrition_v}}>{{x}}</option>{% endfor %}</select></div>
    <div class="field"><label class="lbl">Processing</label>
      <select name="processing"><option value="">&mdash;</option>
      {% for x in processing %}<option value="{{x}}" {{'selected' if x==processing_v}}>{{x}}</option>{% endfor %}</select></div>
    <div class="field"><label class="lbl">Emergent factor mode</label>
      <select name="factor"><option value="">&mdash;</option>
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

<form class="panel" method="post" action="/save">
  <div class="sec">Save to your recipe book</div>
  {% if save_error %}<div class="err">{{save_error}}</div>{% endif %}
  <div class="row">
    <div class="field"><label class="lbl" for="author_name">Your name</label>
      <input type="text" id="author_name" name="author_name" value="{{author_name}}" placeholder="Jane Cook" required></div>
    <div class="field"><label class="lbl" for="author_email">Your email</label>
      <input type="email" id="author_email" name="author_email" value="{{author_email}}" placeholder="jane@example.com" required></div>
  </div>
  <div class="field" style="margin-top:14px"><label class="lbl" for="title">Recipe title</label>
    <input type="text" id="title" name="title" value="{{save_title}}" required></div>
  {% for k,v in gen_params.items() %}<input type="hidden" name="{{k}}" value="{{v}}">{% endfor %}
  <div class="btnrow"><button class="btn" type="submit">Save recipe</button></div>
</form>
{% endif %}

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
"""

# --- Tab 2: View recipes ---------------------------------------------------
RECIPES_BODY = """
<div class="panel">
  <div class="sec">{% if mine %}Your recipes{% else %}Recipe book{% endif %}{% if total %} &middot; {{total}} saved{% endif %}</div>
  {% if saved %}<div class="ok">Recipe saved to the book. 🎉</div>{% endif %}
  <div class="filterbar">
    <a class="fbtn {{'on' if not mine}}" href="/recipes">Everyone's recipes</a>
    {% if has_email %}
      <a class="fbtn {{'on' if mine}}" href="/recipes?mine=1">My recipes</a>
    {% else %}
      <span class="fbtn disabled" title="Save a recipe first to filter by yours">My recipes</span>
    {% endif %}
  </div>
  {% if total == 0 %}
    {% if mine %}
    <p class="empty">You haven't saved any recipes yet. Head to <a href="/">Design recipe</a> to create one.</p>
    {% else %}
    <p class="empty">No recipes saved yet. Head to <a href="/">Design recipe</a> to create your first one.</p>
    {% endif %}
  {% endif %}
  {% for r in recipes %}
  <article class="saved">
    <div class="saved-head"><h3>{{r.title}}</h3><span class="badge">{{r.method}}</span></div>
    <div class="saved-meta">by {{r.author_name}} &middot; {{r.created_at}}</div>
    <div class="line"><div class="k">Chop</div><div class="v">{{r.ingredients|join(', ')}}</div></div>
    {% if r.protein %}<div class="line"><div class="k">Protein</div><div class="v">{{r.protein}}</div></div>{% endif %}
    {% if r.feculent %}<div class="line"><div class="k">Serve over</div><div class="v">{{r.feculent}}</div></div>{% endif %}
    <div class="line"><div class="k">Sauce</div><div class="v"><span class="sauce">{{r.sauce}}</span> <span class="fit">fit {{r.fit}}</span></div></div>
    <div class="line"><div class="k">Method</div><div class="v steps">{{r.steps}}</div></div>
  </article>
  {% endfor %}
  {% if pages > 1 %}
  <div class="pager">
    {% if page > 1 %}<a href="/recipes?page={{page-1}}{% if mine %}&mine=1{% endif %}">&larr; Newer</a>{% endif %}
    <span class="muted">Page {{page}} of {{pages}}</span>
    {% if page < pages %}<a href="/recipes?page={{page+1}}{% if mine %}&mine=1{% endif %}">Older &rarr;</a>{% endif %}
  </div>
  {% endif %}
</div>
"""

# --- Tab 3: Fullstar tools (brand story) -----------------------------------
ABOUT_BODY = """
<div class="panel about">
  <h2>The Fullstar Story</h2>
  <p class="lede">Cooking made simple — that's the idea this Recipe Studio is built around,
    and it's the idea Fullstar has chased since day one.</p>

  <p>Founded in the United States in 2015, Fullstar is a kitchenware brand on a mission to
    revolutionize food preparation. Over the years the global Fullstar team has launched a
    steady stream of innovations aimed at one goal: solving the everyday problems of the
    kitchen so that cooking is faster, easier, and more enjoyable for everyone.</p>

  <p>From vegetable choppers and mandoline slicers to cookware, food-storage solutions and
    utensils, Fullstar empowers home cooks with high-quality tools that are thoughtfully
    designed and genuinely useful. Its 4-in-1 Vegetable Chopper became a viral sensation,
    with over a million units sold worldwide.</p>

  <div class="stats">
    <div class="stat"><b>2015</b><span>Founded in the USA</span></div>
    <div class="stat"><b>10M+</b><span>Home cooks</span></div>
    <div class="stat"><b>4.8&#9733;</b><span>Average rating</span></div>
  </div>

  <h3>Why a chopper pairs with this studio</h3>
  <p>The Recipe Studio picks coherent ingredient pairings with a food-flavor AI; the chopper
    turns those ingredients into evenly-cut pieces in seconds. Design a recipe in the first
    tab, prep it with the chopper, and dinner is done.</p>

  <h3>The Original Pro Chopper</h3>
  <ul>
    <li>Vegetable chopper, dicer, and spiralizer in one compact tool.</li>
    <li>Heavy-duty, rust-resistant 420 stainless-steel blades that stay razor-sharp.</li>
    <li>Large catch container so chopped veg lands ready to cook — no mess.</li>
    <li>Dishwasher safe (top rack) for fast cleanup.</li>
  </ul>

  <a class="cta" href="{{amazon_url}}" target="_blank" rel="noopener noreferrer">
    Get the Fullstar Vegetable Chopper on Amazon &rarr;</a>
  <span class="cta-note">Opens the Amazon listing in a new tab.</span>
</div>
"""

TRUE = ("1", "true", "on")
# Args that fully describe a generated recipe — carried as hidden fields on
# the save form so the recipe can be regenerated server-side at save time.
GEN_KEYS = ("seed", "seed2", "n", "model", "cuisine", "flavor", "aroma",
            "nutrition", "processing", "factor", "intensity", "method",
            "veg_only", "protein", "feculent", "vegan", "no_dairy", "no_nuts")


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


def _design_context(args, *, save_error=None, author_name=None,
                    author_email=None, save_title=None):
    """Build the full render context for the Design tab from a request mapping
    (request.args on GET, request.form when re-rendering after a failed save)."""
    recipe, error, _ = _recipe_from_args(args)
    excludes = [e for e in ("vegan", "no_dairy", "no_nuts") if args.get(e) in TRUE]
    gen_params = {k: args.get(k) for k in GEN_KEYS if (args.get(k) or "") != ""}
    if author_name is None:
        author_name = session.get("author_name", "")
    if author_email is None:
        author_email = session.get("author_email", "")
    if save_title is None:
        save_title = recipe["title"] if recipe else ""
    return dict(
        recipe=recipe, error=error,
        veg=VEG, veg_sample=VEG[::7][:120], cuisines=CUISINES, factors=FACTORS,
        flavors=FLAVORS, aromas=AROMAS, nutrition=NUTRITION, processing=PROCESSING,
        models=[(k, {"core": "Core (chem + recipe)", "cooc": "Cooc (recipe only)",
                     "chem": "Chem (chemistry only)"}[k]) for k in cr.SIBLINGS],
        seed=args.get("seed", ""), seed2=args.get("seed2", ""),
        n=int(args.get("n", 4)) if args.get("n", "4").isdigit() else 4,
        model=args.get("model", "core"),
        cuisine=args.get("cuisine", ""), flavor=args.get("flavor", ""),
        aroma=args.get("aroma", ""), nutrition_v=args.get("nutrition", ""),
        processing_v=args.get("processing", ""), factor=args.get("factor", ""),
        intensity=int(args.get("intensity", 30)) if args.get("intensity", "30").isdigit() else 30,
        method_v=args.get("method", "auto"),
        veg_only=args.get("veg_only") in TRUE, protein=args.get("protein") in TRUE,
        feculent=args.get("feculent") in TRUE, excludes=excludes,
        gen_params=gen_params, save_error=save_error,
        author_name=author_name, author_email=author_email, save_title=save_title)


@app.route("/")
def home():
    return render_page("design", DESIGN_BODY, **_design_context(request.args))


@app.route("/save", methods=["POST"])
def save():
    form = request.form
    recipe, error, _ = _recipe_from_args(form)
    if not recipe:
        # Can't regenerate the recipe (e.g. seed gone) — back to a clean form.
        return redirect(url_for("home"))

    name = (form.get("author_name") or "").strip()
    email = (form.get("author_email") or "").strip()
    title = (form.get("title") or "").strip()
    save_error = None
    if not name:
        save_error = "Please enter your name."
    elif not db.valid_email(email):
        save_error = "Please enter a valid email address."
    elif not title:
        save_error = "Please give the recipe a title."

    if save_error:
        ctx = _design_context(form, save_error=save_error, author_name=name,
                              author_email=email, save_title=title)
        return render_page("design", DESIGN_BODY, **ctx)

    # Remember the cook for next time, then persist and show the book.
    session["author_name"] = name
    session["author_email"] = email
    db.save_recipe(name, email, title, recipe)
    return redirect(url_for("recipes", saved=1))


@app.route("/recipes")
def recipes():
    my_email = session.get("author_email") or ""
    # "My recipes" filter only applies if we know who the cook is (they've
    # saved at least once this session); otherwise fall back to everyone's.
    mine = request.args.get("mine") in TRUE and bool(my_email)
    author_email = my_email if mine else None
    total = db.count_recipes(author_email=author_email)
    pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    page = max(1, min(page, pages))
    items = db.list_recipes(page=page, per_page=PER_PAGE, author_email=author_email)
    return render_page("recipes", RECIPES_BODY, recipes=items, total=total,
                       page=page, pages=pages, mine=mine, has_email=bool(my_email),
                       saved=request.args.get("saved") in TRUE)


@app.route("/about")
def about():
    return render_page("about", ABOUT_BODY, amazon_url=AMAZON_URL)


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
