"""IGNIS NOVA: fire safety evidence for human space exploration (NASA Space Apps Challenge 2026).

Run locally:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ignis_core as core

st.set_page_config(page_title="IGNIS NOVA", page_icon="🔥", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ design tokens
BG, SURFACE, RAISED, LINE = "#0C1220", "#131B2E", "#19233A", "#25324B"
TEXT, MUTED = "#E8EDF5", "#93A0B8"
FLAME, FLAME_SOFT = "#3987e5", "#6da7ec"      # evidence (blue flame)
EMBER, EMBER_TEXT = "#d95926", "#ec835a"      # gaps and risk, used sparingly
TEAL, TEAL_TEXT = "#199e70", "#3fc79a"
COMPARE = [FLAME, EMBER, TEAL]                # validated 3-series palette
FONT = "IBM Plex Sans, Segoe UI, Arial, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--bg:#0C1220;--surface:#131B2E;--raised:#19233A;--line:#25324B;--text:#E8EDF5;--muted:#93A0B8;--flame:#3987e5;--ember:#ec835a;--teal:#3fc79a;}
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {font-family:'IBM Plex Sans','Segoe UI',Arial,sans-serif;}
.stApp{background:var(--bg);color:var(--text);}
.block-container{max-width:1180px;padding-top:2.4rem;padding-bottom:5rem;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--line);}
.brand{display:flex;align-items:center;gap:14px;margin-bottom:22px;}
.wordmark{font-size:19px;font-weight:600;letter-spacing:.09em;line-height:1.1;}
.tagline{font-size:13px;color:var(--muted);margin-top:3px;}
.hint{font-size:13px;color:var(--muted);margin:2px 0 6px 0;}
.lead h2{font-size:30px;font-weight:600;line-height:1.28;letter-spacing:-.012em;max-width:30em;margin:26px 0 0 0;padding:0;}
.lead p{font-size:15px;line-height:1.6;color:var(--muted);max-width:62ch;margin:12px 0 0 0;}
.figs{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:30px 0 34px 0;}
.fig{padding:18px 22px;border-left:1px solid var(--line);}
.fig:first-child{border-left:none;padding-left:0;}
.fig b{display:block;font-size:30px;font-weight:600;line-height:1.15;font-variant-numeric:tabular-nums;}
.fig span{display:block;font-size:13px;color:var(--muted);margin-top:4px;line-height:1.45;}
.sec{font-size:17px;font-weight:600;margin:6px 0 14px 0;}
.note{font-size:13px;color:var(--muted);line-height:1.55;}
.item{display:flex;justify-content:space-between;gap:16px;padding:13px 0;border-bottom:1px solid var(--line);}
.item:first-of-type{border-top:1px solid var(--line);}
.item .t{font-weight:600;font-size:15px;}
.item .s{font-size:13px;color:var(--muted);margin-top:3px;line-height:1.45;}
.item .n{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap;}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;}
.facts{display:grid;grid-template-columns:150px 1fr;gap:11px 20px;font-size:14px;line-height:1.5;margin:14px 0 0 0;}
.facts dt{color:var(--muted);}
.facts dd{margin:0;}
.eri{width:100%;border-collapse:separate;border-spacing:4px;font-size:14px;}
.eri th{font-weight:500;color:var(--muted);text-align:left;padding:4px 8px;font-size:13px;}
.eri td.n{padding:8px 12px 8px 0;line-height:1.4;}
.eri td.n small{display:block;color:var(--muted);font-size:12px;}
.eri td.v{text-align:center;font-weight:600;font-size:20px;border-radius:6px;padding:16px 8px;font-variant-numeric:tabular-nums;width:15%;}
.eri td.m{color:var(--muted);font-size:13px;padding:8px 0 8px 12px;line-height:1.4;}
.gap{padding:14px 0;border-bottom:1px solid var(--line);}
.gap:first-of-type{border-top:1px solid var(--line);}
.gap b{display:block;font-size:14px;margin-bottom:4px;}
.gap span{font-size:13px;color:var(--muted);line-height:1.55;}
.stTabs [data-baseweb="tab-list"]{gap:30px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{padding:10px 0;height:auto;color:var(--muted);font-weight:500;background:transparent;}
.stTabs [aria-selected="true"]{color:var(--text);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--flame);height:2px;}
div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:8px;background:transparent;}
@media(max-width:800px){.figs{grid-template-columns:repeat(2,1fr);} .fig:nth-child(3){border-left:none;padding-left:0;} .facts{grid-template-columns:1fr;gap:2px 0;} .lead h2{font-size:24px;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LOGO = (
    '<svg viewBox="0 0 800 800" width="46" height="46" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M400 130C440 215 525 270 525 385C525 475 468 535 400 535C332 535 275 475 275 385C275 320 318 290 340 240C352 275 368 292 385 298C372 240 382 185 400 130Z" fill="#6da7ec"/>'
    '<path d="M400 290C425 340 462 365 462 425C462 472 435 500 400 500C365 500 338 472 338 425C338 385 365 365 380 335C388 352 396 360 404 362C398 335 396 312 400 290Z" fill="#0C1220"/>'
    '<ellipse cx="400" cy="380" rx="290" ry="85" transform="rotate(-20 400 380)" fill="none" stroke="#3987e5" stroke-width="22"/></svg>'
)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lerp(c1, c2, t):
    a, b = _hex(c1), _hex(c2)
    return "#{:02x}{:02x}{:02x}".format(*[int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3)])


def _lum(c):
    r, g, b = [x / 255.0 for x in _hex(c)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def rgba(h, a):
    r, g, b = _hex(h)
    return "rgba({},{},{},{})".format(r, g, b, a)


def dot(status):
    s = core.STATUS[status]
    return '<span class="dot" style="background:{}"></span>{}'.format(s["color"], s["label"])


def show(fig, height=380):
    """Render a Plotly figure in the dashboard's quiet chart style."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=8, r=8, t=16, b=8),
                      font=dict(family=FONT, size=12, color=TEXT),
                      hoverlabel=dict(bgcolor=RAISED, bordercolor=LINE, font=dict(family=FONT, color=TEXT)))
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def radar(items):
    """items: list of (name, [7 ratings], color)."""
    labels = [core.CRITERIA[k][0] for k in core.CRIT_KEYS]
    fig = go.Figure()
    for name, vals, colr in items:
        fig.add_trace(go.Scatterpolar(r=list(vals) + [vals[0]], theta=labels + [labels[0]], fill="toself", name=name,
                                      line=dict(color=colr, width=2), fillcolor=rgba(colr, 0.22)))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5], gridcolor=LINE,
                                                 linecolor=LINE, tickfont=dict(size=10, color=MUTED)),
                                 angularaxis=dict(gridcolor=LINE, linecolor=LINE)),
                      showlegend=len(items) > 1, legend=dict(orientation="h", y=-0.08))
    return fig


# ------------------------------------------------------------------ state, sidebar
PROFILE_HABITAT = {"spacecraft": "ISS-like cabin", "moon": "Moon base / pressurized rover",
                   "modeling": "Moon base / pressurized rover", "controls": "ISS-like cabin"}


def _apply_profile():
    for k, v in core.PROFILES[st.session_state["profile"]]["w"].items():
        st.session_state["w_" + k] = v


if "profile" not in st.session_state:
    st.session_state["profile"] = "moon"
    _apply_profile()
if "chat" not in st.session_state:
    st.session_state["chat"] = []

sb = st.sidebar
sb.markdown("#### Criteria weights")
sb.caption("Ratings are multiplied by these weights. Choosing a mission profile resets them.")
weights = {}
for k in core.CRIT_KEYS:
    weights[k] = sb.slider(core.CRITERIA[k][0], 0, 5, key="w_" + k, help=core.CRITERIA[k][1])

sb.markdown("#### Filters")
base_df = core.default_df()
upload = sb.file_uploader("Use your own investigations (CSV)", type="csv",
                          help="Same columns as the template in the Method tab.")
if upload is not None:
    try:
        parsed, errs = core.validate_user_df(pd.read_csv(upload))
        if errs:
            sb.error(" ".join(errs))
        else:
            base_df = parsed
            sb.success("Loaded {} investigations.".format(len(parsed)))
    except Exception as exc:
        sb.error("Could not read the CSV: " + str(exc))

status_sel = sb.multiselect("Data status", list(core.STATUS), default=list(core.STATUS),
                            format_func=lambda k: core.STATUS[k]["label"])
fuel_opts = sorted(base_df["fuel"].unique())
fuel_sel = sb.multiselect("Fuel type", fuel_opts, default=fuel_opts)
pillar_opts = sorted(base_df["pillar"].unique())
pillar_sel = sb.multiselect("Safety pillar", pillar_opts, default=pillar_opts)
only_psi = sb.toggle("Only data already in PSI", value=False)

sb.markdown("#### SI Analyst")
sb.text_input("Anthropic API key (optional)", type="password", key="user_api_key",
              help="Without a key the analyst uses the built-in local engine. You can also set ANTHROPIC_API_KEY in Streamlit secrets.")


def get_api_key():
    key = st.session_state.get("user_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            key = None
    return key or None


# ------------------------------------------------------------------ header and profile
html('<div class="brand">' + LOGO + '<div><div class="wordmark">IGNIS NOVA</div>'
     '<div class="tagline">Fire safety evidence for human space exploration</div></div></div>')
st.radio("Mission profile", list(core.PROFILES), format_func=lambda k: core.PROFILES[k]["name"], key="profile",
         on_change=_apply_profile, horizontal=True, label_visibility="collapsed")
pkey = st.session_state["profile"]
profile_name = core.PROFILES[pkey]["name"]
html('<div class="hint">' + core.PROFILES[pkey]["desc"] +
     ' Fine-tune weights and filters in the sidebar (arrow at the top left).</div>')

# ------------------------------------------------------------------ data
scored_all = core.score_frame(base_df, weights)
mask = (scored_all["status"].isin(["archived"] if only_psi else status_sel)
        & scored_all["fuel"].isin(fuel_sel) & scored_all["pillar"].isin(pillar_sel))
view = scored_all[mask].reset_index(drop=True)
view["rank"] = np.arange(1, len(view) + 1)
arch = base_df[base_df["status"] == "archived"]
pen_g = st.session_state.get("pen_g", 0.5)
pen_o2 = st.session_state.get("pen_o2", 0.4)

tab_over, tab_inv, tab_ready, tab_si, tab_earth, tab_meth = st.tabs(
    ["Overview", "Investigations", "Readiness", "SI Analyst", "Earth impact", "Method"])

# ------------------------------------------------------------------ overview
with tab_over:
    hab_name = PROFILE_HABITAT[pkey]
    er = core.evidence_readiness(base_df, core.HABITATS[hab_name], pen_g, pen_o2)
    vals = {p: er[p]["value"] for p in core.PILLARS}
    best = max(vals, key=vals.get)
    low = min(vals.values())
    weakest = [p for p in core.PILLARS if vals[p] <= low + 1.0 and p != best] or [min(vals, key=vals.get)]
    if low >= 75:
        headline = "For the {}, today's PSI evidence is solid on all three pillars; the lowest is {} at {:.0f}/100.".format(
            hab_name, " and ".join(weakest), low)
    elif vals[best] < 60:
        headline = "For the {}, today's PSI evidence only partly applies: best on {} ({:.0f}/100), weakest on {} ({:.0f}/100).".format(
            hab_name, best, vals[best], " and ".join(weakest), low)
    else:
        headline = "For the {}, today's PSI evidence is strongest on {} ({:.0f}/100) and thinnest on {} ({:.0f}/100).".format(
            hab_name, best, vals[best], " and ".join(weakest), low)
    n_partial = int((arch["gravity_regime"] == "partial").sum())
    planned = base_df[(base_df["status"] == "planned") & (base_df["gravity_regime"] == "partial")]["name"].tolist()
    top_arch = scored_all[scored_all["status"] == "archived"].head(1)
    sub = ""
    if len(top_arch):
        sub += "The closest match already in PSI is {} ({:.0f}/100 for this profile). ".format(
            top_arch.iloc[0]["name"], top_arch.iloc[0]["score"])
    sub += "PSI holds {} partial-gravity combustion datasets".format(n_partial)
    sub += "; {} should add some.".format(" and ".join(planned)) if planned else "."
    html('<div class="lead"><h2>' + headline + '</h2><p>' + sub + '</p></div>')

    o2m = arch["o2_max"].max()
    mean_ready = float(np.mean(list(vals.values())))
    html('<div class="figs">'
         '<div class="fig"><b>{}</b><span>experiments in PSI, across {} investigations</span></div>'
         '<div class="fig"><b>{}</b><span>partial-gravity datasets in PSI</span></div>'
         '<div class="fig"><b>{}</b><span>highest oxygen concentration tested (CFI-G, gas fuel)</span></div>'
         '<div class="fig"><b>{:.0f}/100</b><span>average evidence readiness, {}</span></div></div>'.format(
             int(arch["n_exp"].sum()), len(arch), n_partial,
             "{:.0f}%".format(o2m) if pd.notna(o2m) else "n/a", mean_ready, hab_name))

    if len(view) == 0:
        st.info("No investigation matches the current filters. Reset them in the sidebar.")
    else:
        left, right = st.columns([5, 3], gap="large")
        with left:
            html('<div class="sec">Most relevant investigations</div>')
            top = view.head(8)
            op = [{"archived": 1.0, "current": 0.6, "planned": 0.35}.get(s, 1.0) for s in top["status"]]
            fig = go.Figure(go.Bar(
                x=top["score"], y=top["name"], orientation="h",
                marker=dict(color=[rgba(FLAME, o) for o in op]),
                text=["{:.0f}".format(s) for s in top["score"]], textposition="outside", cliponaxis=False,
                customdata=top[["full", "drivers"]].to_numpy(),
                hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Relevance %{x:.0f}/100<br>Main drivers: %{customdata[1]}<extra></extra>"))
            fig.update_layout(bargap=0.5, xaxis=dict(range=[0, 108], visible=False),
                              yaxis=dict(autorange="reversed", showgrid=False))
            show(fig, height=34 * len(top) + 30)
            html('<div class="note">Relevance for the {} profile, 0 to 100. Lighter bars are data not yet in PSI.</div>'.format(
                profile_name.lower()))
        with right:
            html('<div class="sec">Start with these</div>')
            rows = ""
            for _, r in view.head(3).iterrows():
                rows += ('<div class="item"><div><div class="t">{}</div><div class="s">{}</div>'
                         '<div class="s">{}</div></div><div class="n">{:.0f}</div></div>').format(
                             r["name"], dot(r["status"]), r["drivers"] or "Balanced ratings", r["score"])
            html(rows)
        e1, e2, _sp = st.columns([2, 2, 4])
        e1.download_button("Download brief (.md)", data=core.make_brief(view, profile_name, weights, base_df),
                           file_name="ignis_nova_brief.md", mime="text/markdown")
        e2.download_button("Download ranking (.csv)",
                           data=view[["rank", "id", "name", "full", "status", "pillar", "fuel", "score", "drivers"]].to_csv(index=False),
                           file_name="ignis_nova_ranking.csv", mime="text/csv")

# ------------------------------------------------------------------ investigations
with tab_inv:
    mode = st.radio("View", ["Browse", "Compare", "Map"], horizontal=True, label_visibility="collapsed", key="inv_mode")
    if len(view) == 0:
        st.info("No investigation matches the current filters. Reset them in the sidebar.")
    elif mode == "Browse":
        pick = st.selectbox("Investigation", list(view["name"]))
        r = view[view["name"] == pick].iloc[0]
        a, b = st.columns([5, 4], gap="large")
        with a:
            html('<div class="sec">{}</div>'.format(r["full"]))
            html('<div class="note">{} &nbsp; {} &nbsp; {} &nbsp; Relevance {:.0f}/100, rank {} of {}</div>'.format(
                dot(r["status"]), r["pillar"], r["fuel"], r["score"], r["rank"], len(view)))
            html('<p style="margin:14px 0 0 0;font-size:15px;line-height:1.6">{}</p>'.format(r["objective"]))
            earth_lbl = "Earth link (cited)" if bool(r["earth_cited"]) else "Earth link (inferred)"
            html('<dl class="facts">'
                 '<dt>Hardware</dt><dd>{}</dd><dt>Fuels and samples</dt><dd>{}</dd><dt>Test conditions</dt><dd>{}</dd>'
                 '<dt>Measurements</dt><dd>{}</dd><dt>What varied</dt><dd>{}</dd><dt>Gravity</dt><dd>{}</dd>'
                 '<dt>{}</dt><dd>{}</dd><dt>Source</dt><dd>{}</dd></dl>'.format(
                     r["hardware"], r["fuels"], r["conditions"], r["measurements"], r["variables"],
                     core.GRAVITY_LABEL.get(r["gravity_regime"], str(r["gravity_regime"])), earth_lbl, r["earth"],
                     r["source"]))
        with b:
            show(radar([(r["name"], [r[k] for k in core.CRIT_KEYS], FLAME)]), height=380)
        with st.expander("Why it ranks here"):
            ct = core.contributions(r, weights)
            fig = go.Figure(go.Bar(x=ct["points"], y=ct["criterion"], orientation="h", marker_color=FLAME,
                                   text=["{:.1f}".format(p) for p in ct["points"]], textposition="outside",
                                   cliponaxis=False))
            fig.update_layout(bargap=0.5, xaxis=dict(visible=False), yaxis=dict(autorange="reversed", showgrid=False))
            show(fig, height=290)
            st.caption("Points each criterion adds to the {:.0f}/100 relevance score.".format(r["score"]))
        with st.expander("All investigations"):
            tbl = view[["rank", "name", "full", "pillar", "fuel", "status", "score", "drivers"]].copy()
            tbl["status"] = tbl["status"].map(lambda s: core.STATUS[s]["label"])
            st.dataframe(tbl, hide_index=True, column_config={
                "score": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=100, format="%.0f"),
                "full": "Investigation", "drivers": "Main drivers"})
    elif mode == "Compare":
        names = list(scored_all["name"])
        chosen = st.multiselect("Choose two or three investigations", names, default=list(view["name"].head(3)),
                                max_selections=3)
        if len(chosen) < 2:
            st.info("Choose at least two investigations to compare them.")
        else:
            sel = scored_all[scored_all["name"].isin(chosen)].reset_index(drop=True)
            c1, c2 = st.columns([4, 5], gap="large")
            with c1:
                show(radar([(r["name"], [r[k] for k in core.CRIT_KEYS], COMPARE[i % 3]) for i, (_, r) in enumerate(sel.iterrows())]),
                     height=380)
            with c2:
                html('<div class="sec">What they share and where they differ</div>')
                for line in core.compare_text(sel):
                    st.markdown("- " + line)
                attrs = {"full": "Investigation", "pillar": "Pillar", "fuel": "Fuel", "status": "Status",
                         "facility_group": "Hardware platform", "fuels": "Fuels and samples",
                         "conditions": "Test conditions", "measurements": "Measurements", "variables": "What varied",
                         "score": "Relevance"}
                comp = sel.set_index("name")[list(attrs)].copy()
                comp["status"] = comp["status"].map(lambda s: core.STATUS[s]["label"])
                comp["score"] = comp["score"].map(lambda s: "{:.0f}/100".format(s))
                comp.columns = list(attrs.values())
                st.dataframe(comp.T)
    else:
        which = st.radio("Map", ["Relevance landscape", "Hardware lineage", "Similarity"], horizontal=True,
                         label_visibility="collapsed", key="map_mode")
        if which == "Relevance landscape":
            jit = np.array([(((i * 37) % 11) - 5) * 0.045 for i in range(len(view))])
            fig = go.Figure(go.Scatter(
                x=view["crewEnv"] + jit, y=view["modeling"] + jit[::-1], mode="markers+text", text=view["name"],
                textposition="top center", textfont=dict(size=11, color=MUTED),
                marker=dict(size=10 + view["score"] / 100 * 26, color=rgba(FLAME, 0.7), line=dict(width=1, color=FLAME_SOFT)),
                hovertemplate="<b>%{text}</b><extra></extra>"))
            fig.update_layout(xaxis=dict(title="Relevance to crew habitats (0 to 5)", range=[-0.5, 5.7]),
                              yaxis=dict(title="Value for predictive models (0 to 5)", range=[0, 5.9]))
            show(fig, height=500)
            st.caption("Bubble size is the relevance score for the current profile. Upper right is the best of both.")
        elif which == "Hardware lineage":
            groups = base_df.groupby("facility_group")["n_exp"].sum()
            labels, parents, values = ["All investigations"], [""], [float(groups.sum())]
            for gname, gval in groups.items():
                labels.append(gname)
                parents.append("All investigations")
                values.append(float(gval))
            for _, r in base_df.iterrows():
                labels.append(str(r["name"]) + " ")
                parents.append(r["facility_group"])
                values.append(float(r["n_exp"]))
            fig = go.Figure(go.Treemap(labels=labels, parents=parents, values=values, branchvalues="total",
                                       marker=dict(colorscale=[[0, "#16294a"], [1, FLAME]]), textinfo="label+value"))
            show(fig, height=480)
            st.caption("Area is the number of experiments. Investigations on the same hardware produced directly comparable runs.")
        else:
            sim = core.similarity_matrix(base_df)
            fig = go.Figure(go.Heatmap(z=sim.to_numpy(), x=sim.columns, y=sim.index, zmin=0, zmax=1,
                                       colorscale=[[0, "#16294a"], [1, FLAME_SOFT]]))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            show(fig, height=560)
            st.caption("Cosine similarity of the seven criteria ratings: brighter means the two investigations answer the same questions.")

# ------------------------------------------------------------------ readiness
with tab_ready:
    html('<div class="lead"><h2 style="font-size:24px;margin-top:18px">How well does today\'s evidence support each habitat?</h2>'
         '<p>Evidence readiness compares the best archived PSI investigations per safety pillar with the conditions crews will live in. '
         'Missing partial-gravity data and oxygen-enriched air lower the score.</p></div>')
    with st.expander("Adjust assumptions"):
        st.slider("Penalty when PSI has no direct partial-gravity data", 0.0, 1.0, 0.5, 0.05, key="pen_g")
        st.slider("Penalty for oxygen-enriched atmospheres", 0.0, 1.0, 0.4, 0.05, key="pen_o2")
        st.caption("These two numbers are judgement calls, not measurements. Change them and the whole dashboard follows.")
    pg, po = st.session_state.get("pen_g", 0.5), st.session_state.get("pen_o2", 0.4)

    colA, colB = st.columns([6, 4], gap="large")
    with colA:
        html('<div class="sec">Evidence readiness by habitat</div>')
        head = "<table class='eri'><tr><th></th>" + "".join("<th>{}</th>".format(p.capitalize()) for p in core.PILLARS) + "<th>Main limits</th></tr>"
        body = ""
        for hname, h in core.HABITATS.items():
            e = core.evidence_readiness(base_df, h, pg, po)
            cells = ""
            for p in core.PILLARS:
                v = e[p]["value"]
                bg = _lerp("#1b2d4f", FLAME_SOFT, min(max(v / 100.0, 0), 1))
                fg = EMBER_TEXT if v < 40 else ("#0C1220" if _lum(bg) > 0.38 else TEXT)
                cells += '<td class="v" style="background:{};color:{}">{:.0f}</td>'.format(bg, fg, v)
            gs, os_ = e["prevention"]["gravity_severity"], e["prevention"]["oxygen_severity"]
            limits = []
            if gs > 0:
                limits.append("no partial-gravity data" if gs >= 0.99 else "partial gravity only as analog or planned")
            if os_ > 0:
                limits.append("oxygen-enriched air")
            body += '<tr><td class="n">{}<small>{:.0f}% O₂, {} psi, {} g</small></td>{}<td class="m">{}</td></tr>'.format(
                hname, h["o2"], h["psi"], h["g"], cells, ", ".join(limits) or "none identified")
        html(head + body + "</table>")
        html('<div class="note" style="margin-top:8px">0 means no usable evidence, 100 means strong and directly applicable evidence. '
             'It shows where evidence is thin, not whether a habitat is safe.</div>')
    with colB:
        html('<div class="sec">Evidence gaps found in the data</div>')
        gaps_html = ""
        for g in core.gap_report(base_df):
            colr = {"critical": EMBER_TEXT, "high": FLAME_SOFT}.get(g["severity"], MUTED)
            gaps_html += '<div class="gap"><b><span class="dot" style="background:{}"></span>{}</b><span>{}</span></div>'.format(
                colr, g["title"], g["detail"])
        html(gaps_html)

    html('<div class="sec" style="margin-top:30px">Where crews will be and where the evidence is</div>')
    ht = core.habitat_table()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ht["Gravity (g)"], y=ht["O₂ (%)"], mode="markers+text", text=ht["Habitat"], textposition="middle right",
        textfont=dict(color=TEXT, size=12), name="Expected habitat conditions",
        marker=dict(symbol="circle-open", size=17, line=dict(width=2, color=TEXT))))
    docs = arch[arch["o2_max"].notna()]
    fig.add_trace(go.Scatter(
        x=[0.0] * len(docs), y=docs["o2_max"], mode="markers+text", text=docs["name"], textposition="middle right",
        textfont=dict(color=FLAME_SOFT, size=12), name="Documented high-O₂ test (microgravity)",
        marker=dict(size=14, color=FLAME)))
    fig.add_vline(x=0.16, line_dash="dot", line_color=MUTED,
                  annotation_text="Planned lunar data (FM2, LUCI)", annotation_position="top right",
                  annotation_font=dict(color=MUTED, size=11))
    fig.update_layout(xaxis=dict(title="Gravity (g)", range=[-0.04, 0.62]),
                      yaxis=dict(title="O₂ concentration (%)", range=[0, 112]),
                      legend=dict(orientation="h", y=-0.2))
    show(fig, height=430)
    st.caption("Run-level O₂ and airflow for the other tests live in the PSI datasets; extracting them is the natural next step.")
    html('<div class="note" style="margin-top:8px"><b>Why concentration matters:</b> the Moon base O₂ partial pressure ({:.2f} psi) is close to Earth air '
         '({:.2f} psi), yet the O₂ concentration is {:+.0f}% higher. The brief names concentration as the driver of flammability, '
         'so partial pressure alone would hide the risk.</div>'.format(0.30 * 10.2, 0.21 * 14.7, (30 / 21 - 1) * 100))
    with st.expander("Habitat conditions"):
        st.dataframe(ht, hide_index=True)

# ------------------------------------------------------------------ SI analyst
with tab_si:
    has_key = bool(get_api_key())
    html('<div class="lead"><h2 style="font-size:24px;margin-top:18px">Ask the SI Analyst</h2><p>{}</p></div>'.format(
        "Answers are written by Claude and grounded in this dataset; they cite investigation ids." if has_key else
        "The built-in local engine answers from the dataset. Add an Anthropic API key in the sidebar for open-ended, "
        "LLM-written answers. Ratings are analyst judgements, not measurements."))
    prompts = ["Which tests matter for 30% oxygen on the Moon?", "What are the biggest evidence gaps?",
               "Best data for smoke detection?", "Which investigations share hardware?"]
    pcols = st.columns(len(prompts))
    for i, (col, ptxt) in enumerate(zip(pcols, prompts)):
        if col.button(ptxt, key="prompt_" + str(i)):
            st.session_state["pending"] = ptxt
    question = st.chat_input("Ask about oxygen, detection, suppression, partial gravity, modeling, hardware or gaps")
    if not question:
        question = st.session_state.pop("pending", None)
    if question:
        st.session_state["chat"].append({"role": "user", "content": question})
        engine = "Local engine"
        key = get_api_key()
        if key:
            try:
                reply = core.ask_llm(question, base_df, weights, profile_name, key)
                engine = "Claude, grounded in the dataset"
            except Exception as exc:
                reply = (core.answer_local(question, base_df, weights) +
                         "\n\nThe LLM was unavailable (" + type(exc).__name__ + "), so the local engine answered.")
        else:
            reply = core.answer_local(question, base_df, weights)
        st.session_state["chat"].append({"role": "assistant", "content": reply, "engine": engine})
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                st.caption(m.get("engine", "Local engine"))
    if st.session_state["chat"] and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()

# ------------------------------------------------------------------ earth impact
with tab_earth:
    html('<div class="lead"><h2 style="font-size:24px;margin-top:18px">Hard lessons in orbit, safer lives on Earth</h2>'
         '<p>Every fire in a spacecraft is a fire with nowhere to go. NASA has turned that constraint into technology that now '
         'protects firefighters, workplaces and homes.</p></div>')
    ev = pd.DataFrame(core.EVENTS)
    fig = go.Figure()
    for kind, colr, ypos in [("Incident", EMBER, 1), ("Spin-off", TEAL, 0)]:
        sub_ev = ev[ev["kind"] == kind]
        fig.add_trace(go.Scatter(
            x=sub_ev["year"], y=[ypos] * len(sub_ev), mode="markers+text", name={"Incident": "Incidents", "Spin-off": "Spin-offs"}[kind],
            text=[l + (" (approx.)" if a else "") for l, a in zip(sub_ev["label"], sub_ev["approx"])],
            textposition="top center", textfont=dict(size=11, color=TEXT), marker=dict(size=14, color=colr),
            customdata=sub_ev["detail"], hovertemplate="<b>%{text}</b> %{x}<br>%{customdata}<extra></extra>"))
    fig.update_layout(yaxis=dict(visible=False, range=[-0.8, 1.9], showgrid=False),
                      xaxis=dict(title="Year", showgrid=False), legend=dict(orientation="h", y=-0.25))
    show(fig, height=330)
    t1, t2 = st.columns(2, gap="large")
    with t1:
        html('<div class="sec">From space to the fire station</div>'
             '<div class="note">Fine-water-mist extinguishers developed for the ISS, smoke detectors that tell smoke from dust '
             '(fewer nuisance alarms, useful in mining), and firefighter clothing and air tanks adapted from spacesuit technology.</div>')
        for _, r in base_df[base_df["earth_cited"].astype(bool)].iterrows():
            st.markdown("- **{}**: {}".format(r["name"], r["earth"]))
    with t2:
        html('<div class="sec">What the PSI data can add</div>'
             '<div class="note">Flammability limits of fabrics and plastics support material standards, extinction limits guide suppressant design, '
             'smoke characterization improves detector discrimination, and ACME aims at efficient, low-emission combustion. '
             'Rocket and propellant fires are outside PSI: engine combustion is effectively independent of gravity.</div>')
    html('<div class="sec" style="margin-top:26px">Sources</div>')
    for title, url, note_ in core.RESOURCES:
        st.markdown("- [{}]({}): {}".format(title, url, note_))

# ------------------------------------------------------------------ method
with tab_meth:
    html('<div class="lead"><h2 style="font-size:24px;margin-top:18px">How the numbers are built</h2></div>')
    m1, m2 = st.columns(2, gap="large")
    with m1:
        html('<div class="sec">Ranking</div><div class="note">Each investigation is rated 0 to 5 on seven criteria '
             '(5 tests the topic directly, 3 partially or indirectly, 1 weakly, 0 not at all). '
             'Relevance = Σ(weight × rating) ÷ (5 × Σ weights) × 100. Profiles only change the weights.</div>')
        st.dataframe(pd.DataFrame([{"Criterion": v[0], "Meaning": v[1]} for v in core.CRITERIA.values()]), hide_index=True)
    with m2:
        html('<div class="sec">Limits to keep in mind</div><div class="note">'
             'Ratings are analyst judgements from the challenge brief and the NASA Combustion Science page, not computed from raw measurements.<br><br>'
             'Fields marked "confirm in PSI" are not stated in the brief; the PSI repository is the authority.<br><br>'
             'FM2 and LUCI are planned (no data yet). SoFIE and FLARE are listed as current NASA experiments; their PSI availability needs confirming.<br><br>'
             'Evidence readiness is a heuristic to show where evidence is thin, not a safety certification.</div>')
    html('<div class="sec" style="margin-top:26px">Use real PSI data</div><div class="note">'
         'Download the template, add one row per investigation from the '
         '<a href="https://psi.nasa.gov/physci/repo/search?q=combustion" style="color:#6da7ec">PSI repository</a> '
         '(ratings 0 to 5), then upload it in the sidebar. Every chart, ranking and the SI Analyst update.</div>')
    st.download_button("Download CSV template", data=base_df.to_csv(index=False), file_name="ignis_nova_template.csv",
                       mime="text/csv")
    with st.expander("Dataset"):
        st.dataframe(base_df.drop(columns=["hardware"]), hide_index=True)
