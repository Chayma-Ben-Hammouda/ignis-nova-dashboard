"""IGNIS NOVA · Flame in Freefall — SI-powered fire-safety dashboard (NASA Space Apps Challenge 2026).

Run locally:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ignis_core as core

st.set_page_config(page_title="IGNIS NOVA · Flame in Freefall", page_icon="🔥", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- style
FIRE, ICE, MINT, MUTED = "#ff9100", "#4fc3f7", "#69f0ae", "#8fa3d9"

st.markdown(
    """
<style>
.stApp{background:radial-gradient(1100px 520px at 85% -8%,#1a2247,#070b1d);}
[data-testid="stSidebar"]{background:#0b1128;border-right:1px solid #26305f;}
.hero{display:flex;align-items:center;gap:18px;padding:6px 0 14px 0;border-bottom:1px solid #26305f;margin-bottom:14px;}
.hero h1{margin:0;font-size:30px;letter-spacing:5px;font-weight:800;}
.hero h1 span{color:#ff9100;}
.hero p{margin:2px 0 0 0;color:#4fc3f7;font-size:12px;letter-spacing:3px;}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;margin-right:6px;border:1px solid #26305f;}
.card{background:#0f1530;border:1px solid #26305f;border-radius:14px;padding:14px 16px;margin-bottom:10px;}
.card h4{margin:0 0 6px 0;color:#e8ecff;}
.small{font-size:12px;color:#8fa3d9;}
div[data-testid="stMetric"]{background:#0f1530;border:1px solid #26305f;border-radius:12px;padding:10px 14px;}
div[data-testid="stMetricValue"]{color:#ff9100;}
</style>
""",
    unsafe_allow_html=True,
)

LOGO = (
    '<svg viewBox="0 0 800 800" width="64" height="64" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M400 130C440 215 525 270 525 385C525 475 468 535 400 535C332 535 275 475 275 385C275 320 318 290 340 240C352 275 368 292 385 298C372 240 382 185 400 130Z" fill="#ff9100"/>'
    '<path d="M400 290C425 340 462 365 462 425C462 472 435 500 400 500C365 500 338 472 338 425C338 385 365 365 380 335C388 352 396 360 404 362C398 335 396 312 400 290Z" fill="#4fc3f7"/>'
    '<ellipse cx="400" cy="380" rx="290" ry="85" transform="rotate(-20 400 380)" fill="none" stroke="#4fc3f7" stroke-width="22"/></svg>'
)
st.markdown(
    '<div class="hero">' + LOGO +
    '<div><h1>IGNIS<span> NOVA</span></h1>'
    '<p>FLAME IN FREEFALL · SI-POWERED FIRE SAFETY INSIGHTS FROM MICROGRAVITY COMBUSTION DATA</p></div></div>',
    unsafe_allow_html=True,
)


def show(fig, height=420):
    """Render a Plotly figure with the dashboard's dark styling."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=40, b=10), font=dict(color="#e8ecff"))
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------- sidebar
def _apply_profile():
    for k, v in core.PROFILES[st.session_state["profile"]]["w"].items():
        st.session_state["w_" + k] = v


if "profile" not in st.session_state:
    st.session_state["profile"] = "moon"
    _apply_profile()
if "chat" not in st.session_state:
    st.session_state["chat"] = []

sb = st.sidebar
sb.markdown("### Mission profile")
sb.selectbox("Who is the dashboard for?", list(core.PROFILES), format_func=lambda k: core.PROFILES[k]["name"],
             key="profile", on_change=_apply_profile)
sb.caption(core.PROFILES[st.session_state["profile"]]["desc"])

sb.markdown("### Criteria weights")
weights = {}
for k in core.CRIT_KEYS:
    weights[k] = sb.slider(core.CRITERIA[k][0], 0, 5, key="w_" + k, help=core.CRITERIA[k][1])

sb.markdown("### Filters")
base_df = core.default_df()
upload = sb.file_uploader("Load your own investigations CSV", type="csv",
                          help="Same columns as the template in the Method & Data tab.")
if upload is not None:
    try:
        parsed, errs = core.validate_user_df(pd.read_csv(upload))
        if errs:
            sb.error(" ".join(errs))
        else:
            base_df = parsed
            sb.success("Loaded {} investigations from your CSV.".format(len(parsed)))
    except Exception as exc:  # unreadable file
        sb.error("Could not read the CSV: " + str(exc))

status_sel = sb.multiselect("Data status", list(core.STATUS), default=list(core.STATUS),
                            format_func=lambda k: core.STATUS[k]["label"])
fuel_opts = sorted(base_df["fuel"].unique())
fuel_sel = sb.multiselect("Fuel type", fuel_opts, default=fuel_opts)
pillar_opts = sorted(base_df["pillar"].unique())
pillar_sel = sb.multiselect("Safety pillar", pillar_opts, default=pillar_opts)
only_psi = sb.toggle("Only data already in PSI", value=False)

sb.markdown("### SI Analyst (optional)")
sb.text_input("Anthropic API key", type="password", key="user_api_key",
              help="Optional. Without a key the analyst uses the built-in local engine. You can also set ANTHROPIC_API_KEY in Streamlit secrets.")


def get_api_key():
    key = st.session_state.get("user_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            key = None
    return key or None


# ----------------------------------------------------------------------------- data
profile_name = core.PROFILES[st.session_state["profile"]]["name"]
scored_all = core.score_frame(base_df, weights)
mask = (scored_all["status"].isin(["archived"] if only_psi else status_sel)
        & scored_all["fuel"].isin(fuel_sel) & scored_all["pillar"].isin(pillar_sel))
view = scored_all[mask].reset_index(drop=True)
view["rank"] = np.arange(1, len(view) + 1)
arch = base_df[base_df["status"] == "archived"]

tabs = st.tabs(["🎯 Mission Control", "🔎 Explorer", "⚖️ Compare", "🧭 Readiness & Gaps",
                "🤖 SI Analyst", "🌍 Earth Impact", "📚 Method & Data"])
tab_mc, tab_ex, tab_cmp, tab_gap, tab_si, tab_earth, tab_meth = tabs


def radar(items):
    """items: list of (name, [7 scores])."""
    labels = [core.CRITERIA[k][0] for k in core.CRIT_KEYS]
    fig = go.Figure()
    for name, vals in items:
        fig.add_trace(go.Scatterpolar(r=list(vals) + [vals[0]], theta=labels + [labels[0]], fill="toself",
                                      name=name, opacity=0.65))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5], visible=True, gridcolor="#26305f"),
                                 angularaxis=dict(gridcolor="#26305f"), bgcolor="rgba(0,0,0,0)"),
                      legend=dict(orientation="h", y=-0.1))
    return fig


# ----------------------------------------------------------------------------- 1. mission control
with tab_mc:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Investigations in view", len(view))
    c2.metric("Experiments covered in PSI", int(arch["n_exp"].sum()))
    c3.metric("Partial-g datasets in PSI", int((arch["gravity_regime"] == "partial").sum()),
              help="The challenge brief states there is currently no partial-gravity combustion data in PSI.")
    o2m = arch["o2_max"].max()
    c4.metric("Max documented O₂ (µg)", "{:.0f}%".format(o2m) if pd.notna(o2m) else "n/a")

    if len(view) == 0:
        st.warning("No investigation matches the current filters.")
    else:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("#### Relevance ranking · " + profile_name)
            n_max = len(view)
            top_n = st.slider("Show top N", 3, n_max, min(10, n_max)) if n_max > 3 else n_max
            top = view.head(top_n)
            fig = go.Figure(go.Bar(
                x=top["score"], y=top["name"], orientation="h",
                marker_color=[core.PILLAR_COLOR.get(p, FIRE) for p in top["pillar"]],
                text=["{:.0f}".format(s) for s in top["score"]], textposition="outside",
                customdata=top[["full", "drivers"]].to_numpy(),
                hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Score %{x:.0f}/100<br>Drivers: %{customdata[1]}<extra></extra>"))
            fig.update_layout(xaxis=dict(range=[0, 105], title="Relevance score (0-100)"),
                              yaxis=dict(autorange="reversed"), title="Bar colour = safety pillar "
                              "(orange Prevention · blue Detection · green Suppression)")
            show(fig, height=max(320, 34 * top_n + 90))
        with right:
            st.markdown("#### SI briefing")
            for _, r in view.head(3).iterrows():
                st.markdown(
                    '<div class="card"><h4>#{} {} · {:.0f}/100</h4><span class="badge" style="color:{}">{}</span>'
                    '<div class="small" style="margin-top:6px">{}</div>'
                    '<div class="small"><b>Driven by:</b> {}</div></div>'.format(
                        r["rank"], r["name"], r["score"], core.STATUS[r["status"]]["color"],
                        core.STATUS[r["status"]]["label"], r["objective"], r["drivers"] or "n/a"),
                    unsafe_allow_html=True)
            for g in core.gap_report(base_df)[:2]:
                msg = "**" + g["title"] + "** — " + g["detail"]
                (st.error if g["severity"] == "critical" else st.warning)(msg)

        st.markdown("#### Landscape: crew-habitat relevance vs modeling value")
        jit = np.array([(((i * 37) % 11) - 5) * 0.045 for i in range(len(view))])
        fig = go.Figure(go.Scatter(
            x=view["crewEnv"] + jit, y=view["modeling"] + jit[::-1], mode="markers+text", text=view["name"],
            textposition="top center",
            marker=dict(size=10 + view["score"] / 100 * 30, opacity=0.8,
                        color=[core.PILLAR_COLOR.get(p, FIRE) for p in view["pillar"]],
                        line=dict(width=1, color="#e8ecff"))))
        fig.update_layout(xaxis=dict(title="Crew-habitat relevance (0-5)", range=[-0.5, 5.6]),
                          yaxis=dict(title="Modeling value (0-5)", range=[0, 5.8]),
                          title="Bubble size = profile score")
        show(fig, height=470)

        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download mission brief (.md)",
                           data=core.make_brief(view, profile_name, weights, base_df),
                           file_name="ignis_nova_brief.md", mime="text/markdown")
        d2.download_button("⬇ Download ranking (.csv)",
                           data=view[["rank", "id", "name", "full", "status", "pillar", "fuel", "score", "drivers"]].to_csv(index=False),
                           file_name="ignis_nova_ranking.csv", mime="text/csv")

# ----------------------------------------------------------------------------- 2. explorer
with tab_ex:
    if len(view) == 0:
        st.info("No investigation matches the current filters.")
    else:
        tbl = view[["rank", "name", "full", "pillar", "fuel", "status", "score", "drivers"]].copy()
        tbl["status"] = tbl["status"].map(lambda s: core.STATUS[s]["label"])
        st.dataframe(tbl, hide_index=True, column_config={
            "score": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=100, format="%.0f"),
            "full": "Investigation", "drivers": "Top drivers"})
        pick = st.selectbox("Inspect an investigation", list(view["name"]))
        r = view[view["name"] == pick].iloc[0]
        a, b = st.columns([3, 2], gap="large")
        with a:
            st.markdown("### {} — {}".format(r["name"], r["full"]))
            st.markdown('<span class="badge" style="color:{}">{}</span><span class="badge">{}</span><span class="badge">{}</span>'.format(
                core.STATUS[r["status"]]["color"], core.STATUS[r["status"]]["label"], r["pillar"], r["fuel"]),
                unsafe_allow_html=True)
            st.markdown("**Objective.** " + r["objective"])
            st.markdown("- **Hardware:** " + r["hardware"] + "\n- **Fuels / samples:** " + r["fuels"] +
                        "\n- **Test conditions:** " + r["conditions"] + "\n- **Measurements:** " + r["measurements"] +
                        "\n- **What varied:** " + r["variables"] +
                        "\n- **Gravity:** " + core.GRAVITY_LABEL.get(r["gravity_regime"], str(r["gravity_regime"])) +
                        "\n- **Earth link" + (" (cited)" if bool(r["earth_cited"]) else " (analyst inference)") + ":** " + r["earth"] +
                        "\n- **Source:** " + str(r["source"]))
            ct = core.contributions(r, weights)
            fig = go.Figure(go.Bar(x=ct["points"], y=ct["criterion"], orientation="h", marker_color=FIRE,
                                   text=["{:.1f}".format(p) for p in ct["points"]], textposition="outside"))
            fig.update_layout(title="Why it ranks #{} ({:.0f}/100): points per criterion".format(r["rank"], r["score"]),
                              yaxis=dict(autorange="reversed"))
            show(fig, height=330)
        with b:
            show(radar([(r["name"], [r[k] for k in core.CRIT_KEYS])]), height=430)

# ----------------------------------------------------------------------------- 3. compare
with tab_cmp:
    names = list(scored_all["name"])
    default_pick = list(view["name"].head(3)) if len(view) else names[:3]
    chosen = st.multiselect("Pick 2-4 investigations", names, default=default_pick, max_selections=4)
    if len(chosen) < 2:
        st.info("Select at least two investigations to compare.")
    else:
        sel = scored_all[scored_all["name"].isin(chosen)].reset_index(drop=True)
        p1, p2 = st.columns([2, 3], gap="large")
        with p1:
            show(radar([(r["name"], [r[k] for k in core.CRIT_KEYS]) for _, r in sel.iterrows()]), height=420)
        with p2:
            st.markdown("#### Similarities and differences")
            for line in core.compare_text(sel):
                st.markdown("- " + line)
            attrs = {"full": "Investigation", "pillar": "Pillar", "fuel": "Fuel", "status": "Status",
                     "facility_group": "Hardware platform", "fuels": "Fuels / samples", "conditions": "Test conditions",
                     "measurements": "Measurements", "variables": "What varied", "score": "Profile score"}
            comp = sel.set_index("name")[list(attrs)].copy()
            comp["status"] = comp["status"].map(lambda s: core.STATUS[s]["label"])
            comp["score"] = comp["score"].map(lambda s: "{:.0f}/100".format(s))
            comp.columns = list(attrs.values())
            st.dataframe(comp.T)

    st.markdown("#### Hardware lineage: same platform, comparable data")
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
                               marker=dict(colorscale=[[0, "#151c3d"], [1, "#ff9100"]]), textinfo="label+value"))
    fig.update_layout(title="Area = number of experiments")
    show(fig, height=420)

    st.markdown("#### How similar are the investigations? (cosine similarity of criteria scores)")
    sim = core.similarity_matrix(base_df)
    fig = go.Figure(go.Heatmap(z=sim.to_numpy(), x=sim.columns, y=sim.index, zmin=0, zmax=1,
                               colorscale=[[0, "#0f1530"], [0.5, "#4fc3f7"], [1, "#ff9100"]]))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    show(fig, height=560)

# ----------------------------------------------------------------------------- 4. readiness & gaps
with tab_gap:
    st.markdown("#### Where crews will be vs where the evidence is")
    ht = core.habitat_table()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ht["Gravity (g)"], y=ht["O₂ (%)"], mode="markers+text", text=ht["Habitat"], textposition="top right",
        name="Target habitat conditions", marker=dict(symbol="star", size=18, color=FIRE)))
    docs = arch[arch["o2_max"].notna()]
    fig.add_trace(go.Scatter(
        x=[0.0] * len(docs), y=docs["o2_max"], mode="markers+text", text=docs["name"], textposition="top right",
        name="Documented O₂ evidence (µg)", marker=dict(symbol="circle", size=14, color=MINT)))
    fig.add_vline(x=0.16, line_dash="dot", line_color="#ef9a9a",
                  annotation_text="Planned lunar data: FM2, LUCI (O₂ to be defined)", annotation_position="top")
    fig.update_layout(xaxis=dict(title="Gravity (g)", range=[-0.05, 0.6]), yaxis=dict(title="O₂ concentration (%)", range=[0, 110]),
                      legend=dict(orientation="h", y=-0.2))
    show(fig, height=460)
    st.caption("Stars are expected habitat conditions from the brief; the green dot is the only documented high-O₂ test. "
               "Run-level O₂ and airflow for the other tests live in the PSI datasets: extracting them is the natural next step.")

    st.markdown("#### Habitat conditions")
    st.dataframe(ht, hide_index=True)
    st.info("Insight: the Moon base O₂ partial pressure ({:.2f} psi) is close to Earth air ({:.2f} psi), but the O₂ concentration is "
            "{:+.0f}% higher. The brief stresses concentration as the driver of flammability, so partial pressure alone would "
            "hide the risk.".format(0.30 * 10.2, 0.21 * 14.7, (30 / 21 - 1) * 100))

    st.markdown("#### Evidence Readiness Index")
    with st.expander("Tune the penalties (transparent heuristic)", expanded=False):
        pen_g = st.slider("Penalty when PSI has no direct partial-g data", 0.0, 1.0, 0.5, 0.05)
        pen_o2 = st.slider("Penalty for oxygen-enriched atmospheres", 0.0, 1.0, 0.4, 0.05)
    hab_names = list(core.HABITATS)
    z, txt = [], []
    for hname in hab_names:
        er = core.evidence_readiness(base_df, core.HABITATS[hname], pen_g, pen_o2)
        z.append([er[p]["value"] for p in core.PILLARS])
        txt.append(["{:.0f}".format(er[p]["value"]) for p in core.PILLARS])
    fig = go.Figure(go.Heatmap(z=z, x=[p.capitalize() for p in core.PILLARS], y=hab_names, text=txt, texttemplate="%{text}",
                               zmin=0, zmax=100, colorscale=[[0, "#3b0f0f"], [0.5, "#ff9100"], [1, "#69f0ae"]]))
    fig.update_layout(yaxis=dict(autorange="reversed"), title="0 = no usable evidence · 100 = strong, directly applicable evidence")
    show(fig, height=330)
    st.caption("ERI = mean of the 3 best archived investigations per pillar, reduced for missing partial-gravity data and for "
               "oxygen enrichment not covered by documented tests. It ranks where evidence is thin, not absolute safety.")

    st.markdown("#### Evidence gaps found in the data")
    for g in core.gap_report(base_df):
        msg = "**" + g["title"] + "** — " + g["detail"]
        if g["severity"] == "critical":
            st.error(msg)
        elif g["severity"] == "high":
            st.warning(msg)
        else:
            st.info(msg)

# ----------------------------------------------------------------------------- 5. SI analyst
with tab_si:
    st.markdown("#### Ask the SI Analyst")
    has_key = bool(get_api_key())
    st.caption(("Claude, grounded in the dataset: answers cite investigation ids." if has_key else
                "Local engine active. Add an Anthropic API key in the sidebar for open-ended, LLM-written answers.") +
               " Scores are analyst ratings, not measurements.")
    prompts = ["Which tests matter for 30% oxygen on the Moon?", "What are the biggest evidence gaps?",
               "Best data for smoke detection?", "Which investigations share the same hardware?"]
    cols = st.columns(len(prompts))
    for i, (col, ptxt) in enumerate(zip(cols, prompts)):
        if col.button(ptxt, key="prompt_" + str(i)):
            st.session_state["pending"] = ptxt
    question = st.chat_input("Ask about oxygen, detection, suppression, partial gravity, modeling, hardware, gaps…")
    if not question:
        question = st.session_state.pop("pending", None)
    if question:
        st.session_state["chat"].append({"role": "user", "content": question})
        engine = "Local engine"
        key = get_api_key()
        if key:
            try:
                reply = core.ask_llm(question, base_df, weights, profile_name, key)
                engine = "Claude · grounded in dataset"
            except Exception as exc:
                reply = core.answer_local(question, base_df, weights) + "\n\n_(LLM unavailable: " + type(exc).__name__ + "; used the local engine.)_"
        else:
            reply = core.answer_local(question, base_df, weights)
        st.session_state["chat"].append({"role": "assistant", "content": reply, "engine": engine})
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                st.caption("Engine: " + m.get("engine", "Local engine"))
    if st.session_state["chat"] and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()

# ----------------------------------------------------------------------------- 6. earth impact
with tab_earth:
    st.markdown("#### Fire safety: hard lessons and technology that came back to Earth")
    ev = pd.DataFrame(core.EVENTS)
    fig = go.Figure()
    for kind, colr, ypos in [("Incident", "#ef5350", 1), ("Spin-off", MINT, 0)]:
        sub = ev[ev["kind"] == kind]
        fig.add_trace(go.Scatter(
            x=sub["year"], y=[ypos] * len(sub), mode="markers+text", name=kind,
            text=[l + (" (~)" if a else "") for l, a in zip(sub["label"], sub["approx"])], textposition="top center",
            marker=dict(size=16, color=colr), customdata=sub["detail"],
            hovertemplate="<b>%{text}</b> %{x}<br>%{customdata}<extra></extra>"))
    fig.update_layout(yaxis=dict(visible=False, range=[-0.8, 1.8]), xaxis=dict(title="Year (~ = decade given in the source)"),
                      legend=dict(orientation="h", y=-0.25))
    show(fig, height=330)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown('<div class="card"><h4>🧯 From space to the fire station</h4>'
                    '<div class="small">Fine-water-mist extinguishers developed for the ISS, smoke detectors that tell smoke from dust '
                    '(fewer nuisance alarms, useful in mining), firefighter protective clothing and air tanks adapted from spacesuit technology.</div></div>',
                    unsafe_allow_html=True)
        cited = base_df[base_df["earth_cited"].astype(bool)]
        for _, r in cited.iterrows():
            st.markdown("- **{}**: {}".format(r["name"], r["earth"]))
    with e2:
        st.markdown('<div class="card"><h4>🔬 What PSI data can add</h4>'
                    '<div class="small">Flammability limits of fabrics and plastics help material standards; extinction limits guide suppressant design; '
                    'smoke characterization improves detector discrimination; ACME aims at efficient, low-emission combustion on Earth.</div></div>',
                    unsafe_allow_html=True)
        st.markdown("Rocket and propellant fires are outside PSI's scope: the combustion in engines is effectively independent of gravity.")
    st.markdown("#### Sources")
    for title, url, note in core.RESOURCES:
        st.markdown("- [{}]({}) — {}".format(title, url, note))

# ----------------------------------------------------------------------------- 7. method & data
with tab_meth:
    st.markdown("#### How the ranking works")
    st.markdown("`score = Σ(weight × rating) / (5 × Σ weights) × 100`. Each investigation is rated 0-5 on seven criteria. "
                "Anchors: **5** directly tests the topic · **3** partial or indirect · **1** weak link · **0** none. "
                "Profiles only change the weights; you can move any slider.")
    st.dataframe(pd.DataFrame([{"Criterion": v[0], "Meaning": v[1]} for v in core.CRITERIA.values()]), hide_index=True)
    st.markdown("#### Honest limits")
    st.markdown("- Ratings are **analyst-assigned** from the challenge brief and the NASA Combustion Science page, not computed from raw measurements.\n"
                "- Fields saying *confirm in PSI* are not stated in the brief; the PSI repository is the authority.\n"
                "- FM2 and LUCI are planned (no data yet); SoFIE and FLARE are listed as current NASA experiments, PSI availability to confirm.\n"
                "- The Evidence Readiness Index is a heuristic to show where evidence is thin, not a safety certification.")
    st.markdown("#### Plug in real PSI data")
    st.markdown("1. Export or transcribe investigations from the [PSI repository](https://psi.nasa.gov/physci/repo/search?q=combustion).\n"
                "2. Fill the CSV template below (one row per investigation; ratings 0-5).\n"
                "3. Upload it in the sidebar: every chart, ranking and the SI Analyst update.")
    st.download_button("⬇ Download CSV template (current data)", data=base_df.to_csv(index=False),
                       file_name="ignis_nova_template.csv", mime="text/csv")
    st.markdown("#### Dataset")
    st.dataframe(base_df.drop(columns=["hardware"]), hide_index=True)
