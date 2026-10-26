
import io
import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
# Backend address: the Flask container on the shared Docker network (Codespaces), or the backend
# Hugging Face Space URL set as the BACKEND_URL variable of the frontend Space
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")
os.environ.setdefault("BACKEND_URL", BACKEND_URL)        # the assistant's forecast tool calls the same API


# ----------------------------------------------------------------------------------------------
# Reference data and planning rules - shared with the planning assistant (assistant_core.py)
# ----------------------------------------------------------------------------------------------
# All reference numbers are produced by the notebook ("Reference data for the frontend") from SuperKart.csv and the
# test-set results, and shipped next to this file as reference_data.json - nothing is typed in by hand.
import assistant_core as core
from assistant_core import (MODEL_COLUMNS, OUTLET_DIST, PERISHABLES, PRODUCT_TYPES, REFERENCE, STORES, TRAINED_RANGES,
                            TYPE_RANGES, model_row, outlet_position, planning_facts, stock_range, sugar_options, tier_of)

TIER_COLOUR = {"Top 20%": "#0F7B5C", "Middle 60%": "#64748B", "Bottom 20%": "#B86E12"}
# Planning assistant switch (Space variables / secrets):
#   ASSISTANT_MODE = "off"     - no assistant at all (default)
#                    "preview" - only for a link with ?preview=<ASSISTANT_PREVIEW_CODE>, to test it on the live app
#                    "on"      - for everyone
#   ANTHROPIC_API_KEY (secret) - required for "preview" and "on"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ASSISTANT_MODE = os.getenv("ASSISTANT_MODE", "off").strip().lower()
ASSISTANT_PREVIEW_CODE = os.getenv("ASSISTANT_PREVIEW_CODE")


def assistant_switched_on():
    if not ANTHROPIC_API_KEY:
        return False
    if ASSISTANT_MODE == "on":
        return True
    if ASSISTANT_MODE == "preview" and ASSISTANT_PREVIEW_CODE:
        return st.query_params.get("preview") == ASSISTANT_PREVIEW_CODE
    return False


def match_outlet(row):
    """Name of the SuperKart outlet with this store profile, or None for a hypothetical profile."""
    for store_id, store in STORES.items():
        if (row["Store_Type"], row["Store_Size"], row["Store_Location_City_Type"]) == (
                store["Store_Type"], store["Store_Size"], store["Store_Location_City_Type"]):
            return store_id
    return None


@st.cache_resource
def claude_client():
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def md_safe(text):
    """Show $ as a dollar sign: Streamlit markdown would otherwise read $...$ as a maths formula."""
    return text.replace("\\$", "$").replace("$", "\\$")


def trace_badge(untraced, where=st):
    """Guardrail made visible: every number in an LLM text must match the model / data it was given."""
    if untraced:
        where.warning("Could not trace these figures to the model or SuperKart's data - treat them with caution: "
                      + ", ".join(untraced))
    else:
        where.caption("✓ Every figure in this text comes from the model or SuperKart's data.")


def read_aloud(text, key):
    """One button: Read aloud, which turns into Stop while speaking. Browser speech synthesis - free, runs in the
    user's browser, nothing is sent anywhere."""
    components.html(f"""
        <button id="speak{key}" style="background:#0F7B5C;color:white;border:none;border-radius:8px;padding:5px 14px;
                font-family:sans-serif;font-weight:600;cursor:pointer;min-width:120px;">🔊 Read aloud</button>
        <script>
          const text = {json.dumps(text)};
          const button = document.getElementById("speak{key}");
          const idle = () => {{ button.textContent = "🔊 Read aloud"; button.style.background = "#0F7B5C"; }};
          button.onclick = () => {{
              const synth = window.speechSynthesis;
              if (synth.speaking) {{ synth.cancel(); idle(); return; }}
              const u = new SpeechSynthesisUtterance(text.replace(/[*_#`]/g, ""));
              u.lang = "en-US";
              u.onend = idle; u.onerror = idle;
              synth.cancel(); synth.speak(u);
              button.textContent = "■ Stop"; button.style.background = "#B86E12";
          }};
        </script>""", height=44)


def assistant_error(error, where=st):
    """Short message for users; in preview mode also the API's own message. Full details go to the Space logs."""
    import sys, traceback
    traceback.print_exc(file=sys.stderr)
    detail = f" - {str(error)[:300]}" if ASSISTANT_MODE == "preview" else ""
    where.error(f"The assistant is not available right now ({type(error).__name__}){detail}. Please try again.")


LABELS = {"outlet": "Outlet", "product_type": "Category", "sugar_content": "Sugar content", "perishable": "Perishable",
          "product_family": "Product family", "store_type": "Store type", "city_tier": "City tier", "store_size": "Store size", "revenue_tier": "Revenue tier",
          "revenue": "revenue", "mrp": "MRP", "weight": "weight", "display_share_pct": "display share"}


def first_upper(text):
    return text[:1].upper() + text[1:]


def data_chart(call):
    """Chart drawn by code from a data-tool result (never by the LLM), so it always matches the numbers.
    One dimension: sorted bars. Two dimensions: a heatmap with the values written in the cells (readable even for
    16 categories x 4 outlets); grouped bars only when the second dimension has few values."""
    result, rows = call["result"], call["result"].get("rows") if isinstance(call["result"], dict) else None
    if not rows or (len(rows) < 2 and call["tool"] != "revenue_drivers"):
        return None
    frame = pd.DataFrame(rows)
    if call["tool"] == "correlation_matrix":
        frame["x"] = frame["x"].map(lambda v: first_upper(LABELS.get(v, v)))
        frame["y"] = frame["y"].map(lambda v: first_upper(LABELS.get(v, v)))
        base = alt.Chart(frame).encode(x=alt.X("x:N", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
                                       y=alt.Y("y:N", title=None))
        cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("correlation:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
                            legend=alt.Legend(title="Correlation (-1 to 1)", orient="bottom")), tooltip=list(frame.columns))
        text = base.mark_text(fontSize=13, color="#1F2A44").encode(text=alt.Text("correlation:Q", format=".2f"))
        return (cells + text).properties(height=260)
    if call["tool"] == "association_matrix":
        order = [first_upper(LABELS.get(c, c)) for c in result.get("columns", [])]
        frame["x"] = frame["x"].map(lambda v: first_upper(LABELS.get(v, v)))
        frame["y"] = frame["y"].map(lambda v: first_upper(LABELS.get(v, v)))
        base = alt.Chart(frame).encode(
            x=alt.X("x:N", sort=order, title=None, axis=alt.Axis(orient="top", labelAngle=-40, labelLimit=160)),
            y=alt.Y("y:N", sort=order, title=None, axis=alt.Axis(labelLimit=160)))
        cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("strength:Q", scale=alt.Scale(scheme="greens", domain=[0, 1]),
                            legend=alt.Legend(title="Strength of association (0-1)", orient="bottom")),
            tooltip=list(frame.columns))
        text = base.mark_text(fontSize=11).encode(
            text=alt.Text("strength:Q", format=".2f"),
            color=alt.condition(alt.datum.strength > 0.6, alt.value("white"), alt.value("#1F2A44")))
        return (cells + text).properties(height=max(260, 34 * len(order)))
    if call["tool"] == "revenue_drivers":
        frame = frame.dropna(subset=["explained_pct"]) if "explained_pct" in frame else frame.iloc[0:0]
        if frame.empty:
            return None
        group = result.get("group_by")
        if not group:
            used = frame["factors_used"].iloc[0]
            parts = pd.DataFrame([{"factor": first_upper(LABELS.get(f, f)), "share": frame[f"{kind}_pct_{f}"].iloc[0],
                                   "how": label} for f in used
                                  for kind, label in (("alone", "explains alone"), ("adds", "adds on top of the others"))])
            return alt.Chart(parts).mark_bar().encode(
                x=alt.X("share:Q", title=f"% of revenue differences explained - all together: "
                                         f"{frame['explained_pct'].iloc[0]:.0f}%", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("factor:N", sort=alt.EncodingSortField(field="share", op="max", order="descending"), title=None),
                yOffset="how:N",
                color=alt.Color("how:N", scale=alt.Scale(range=[BRAND, ACCENT]), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=list(parts.columns)).properties(height=max(140, 44 * len(used)))
        frame = frame.drop(columns=[c for c in frame.columns if c == "factors_used" or c.startswith(("alone_pct_", "adds_pct_"))])
        effects = frame.melt(id_vars=[c for c in frame.columns if not c.startswith("effect_")],
                             value_vars=[c for c in frame.columns if c.startswith("effect_")], var_name="factor",
                             value_name="effect")
        effects["factor"] = effects["factor"].str.replace("effect_", "", regex=False).map(lambda v: first_upper(LABELS.get(v, v)))
        base = alt.Chart(effects).encode(x=alt.X("factor:N", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
                                         y=alt.Y(f"{group}:N", title=None, axis=alt.Axis(labelLimit=260)))
        cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("effect:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
                            legend=alt.Legend(title="Relative effect (standardised)", orient="bottom")),
            tooltip=list(effects.columns))
        text = base.mark_text(fontSize=12, color="#1F2A44").encode(text=alt.Text("effect:Q", format=".2f"))
        explained = alt.Chart(frame).mark_bar(color=BRAND).encode(
            x=alt.X("explained_pct:Q", title="Explained by the factors together (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y(f"{group}:N", title=None, axis=alt.Axis(labelLimit=260)), tooltip=list(frame.columns))
        height = max(140, 28 * frame[group].nunique())
        return alt.hconcat((cells + text).properties(height=height, width=380), explained.properties(height=height, width=320))
    if call["tool"] == "data_relationship":
        group = result.get("group_by")
        if not group or "correlation" not in frame:
            return None
        frame = frame.dropna(subset=["correlation"])
        frame["direction"] = ["negative" if r < 0 else "positive" for r in frame["correlation"]]
        frame["sample"] = ["fewer than 30 products" if n < 30 else "30 products or more" for n in frame["products"]]
        low = min(-0.2, float(frame["correlation"].min()) - 0.1)
        x = alt.X("correlation:Q", scale=alt.Scale(domain=[low, 1]),
                  title=f"Correlation of {LABELS.get(result['x'], result['x'])} with {LABELS.get(result['y'], result['y'])}")
        y = alt.Y(f"{group}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260))
        bars = alt.Chart(frame).mark_bar().encode(
            x=x, y=y,
            color=alt.Color("direction:N", scale=alt.Scale(domain=["positive", "negative"], range=[BRAND, ACCENT]), legend=None),
            opacity=alt.Opacity("sample:N", scale=alt.Scale(domain=["30 products or more", "fewer than 30 products"],
                                                            range=[1.0, 0.4]),
                                legend=alt.Legend(title="Lighter bars: few products - treat with caution", orient="bottom")),
            tooltip=list(frame.columns))
        labels = alt.Chart(frame).mark_text(fontSize=11, color="#1F2A44", dx=4, align="left").encode(
            x=alt.X("plot_at:Q"), y=y, text=alt.Text("correlation:Q", format=".2f")).transform_calculate(
            plot_at="max(datum.correlation, 0)")
        return (bars + labels).properties(height=max(120, 28 * len(frame)))
    groups = result.get("group_by") or []
    if not groups:
        return None
    value = next((s for s in ("sum", "mean", "median", "max", "min") if s in frame), "products")
    measure = result.get("measure", "")
    money = value != "products" and measure in ("revenue", "mrp")
    name = {"sum": "Total", "mean": "Average", "median": "Median", "max": "Highest", "min": "Lowest"}.get(value, "")
    title = f"{name} {LABELS.get(measure, measure)}{' ($)' if money else ''}".strip() if value != "products" else "Products"
    number_format = ("$.3~s" if value == "sum" else "$,.0f") if money else (",.0f" if value in ("products", "sum") else ",.1f")

    if len(groups) == 2:
        first, second = groups
        many = max(frame[first].nunique(), frame[second].nunique()) > 5
        if many:
            rows_dim, cols_dim = (first, second) if frame[first].nunique() >= frame[second].nunique() else (second, first)
            middle = float(frame[value].max()) / 2
            base = alt.Chart(frame).encode(
                x=alt.X(f"{cols_dim}:N", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
                y=alt.Y(f"{rows_dim}:N", title=None,
                        sort=alt.EncodingSortField(field=value, op="sum", order="descending")))
            cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
                color=alt.Color(f"{value}:Q", scale=alt.Scale(scheme="greens"), legend=alt.Legend(title=title, orient="bottom")),
                tooltip=list(frame.columns))
            text = base.mark_text(fontSize=12).encode(
                text=alt.Text(f"{value}:Q", format=number_format),
                color=alt.condition(alt.datum[value] > middle, alt.value("white"), alt.value("#1F2A44")))
            return (cells + text).properties(height=max(160, 30 * frame[rows_dim].nunique()))
        bars = alt.Chart(frame).mark_bar().encode(
            x=alt.X(f"{value}:Q", title=title), y=alt.Y(f"{first}:N", title=None), yOffset=f"{second}:N",
            color=alt.Color(f"{second}:N", legend=alt.Legend(orient="bottom", title=LABELS.get(second, second))),
            tooltip=list(frame.columns))
        return bars.properties(height=min(24 * len(frame), 560))

    bars = alt.Chart(frame).mark_bar(color=BRAND).encode(
        x=alt.X(f"{value}:Q", title=title), y=alt.Y(f"{groups[0]}:N", sort="-x", title=LABELS.get(groups[0], groups[0])),
        tooltip=list(frame.columns))
    labels = bars.mark_text(align="left", dx=4, fontSize=11, color="#1F2A44").encode(
        text=alt.Text(f"{value}:Q", format=number_format))
    return (bars + labels).properties(height=max(120, 28 * len(frame)))


def relationship_heatmap(calls):
    """Several relationships for the same grouping (e.g. MRP, weight and display share vs revenue by category) in one
    heatmap: rows = groups, columns = measures, the correlation written in each cell."""
    group = calls[0]["result"]["group_by"]
    frame = pd.concat([pd.DataFrame(c["result"]["rows"]).assign(
        measure=f"{LABELS.get(c['result']['x'], c['result']['x'])} vs {LABELS.get(c['result']['y'], c['result']['y'])}")
        for c in calls], ignore_index=True).dropna(subset=["correlation"])
    base = alt.Chart(frame).encode(x=alt.X("measure:N", title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=220)),
                                   y=alt.Y(f"{group}:N", title=None, axis=alt.Axis(labelLimit=260)))
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("correlation:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
                        legend=alt.Legend(title="Correlation (-1 to 1)", orient="bottom")),
        tooltip=list(frame.columns))
    text = base.mark_text(fontSize=12, color="#1F2A44").encode(text=alt.Text("correlation:Q", format=".2f"))
    return (cells + text).properties(height=max(160, 28 * frame[group].nunique()))


def answer_charts(calls):
    """Charts for one answer: relationship results that share a grouping are merged into one heatmap; others are
    drawn one by one (at most four charts)."""
    chart_calls = [c for c in calls if c["tool"] in core.CHART_TOOLS and isinstance(c["result"], dict) and c["result"].get("rows")]
    charts, relationships = [], {}
    for c in chart_calls:
        group = c["result"].get("group_by") if c["tool"] == "data_relationship" else None
        if group:
            relationships.setdefault((group, json.dumps(c["result"].get("filters"), sort_keys=True)), []).append(c)
        else:
            charts.append(data_chart(c))
    for same in relationships.values():
        charts.append(relationship_heatmap(same) if len(same) > 1 else data_chart(same[0]))
    return [c for c in charts if c is not None][:4]


def predict_batch(frame):
    """Send rows to the backend batch endpoint; return predictions in row order."""
    csv_bytes = frame[MODEL_COLUMNS].to_csv(index=False).encode("utf-8")
    response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": ("batch.csv", csv_bytes)}, timeout=90)
    response.raise_for_status()
    result = response.json()
    return [result[str(i)] for i in range(len(frame))]


# ----------------------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------------------
st.set_page_config(page_title="SuperKart Planning Hub", page_icon="🛒", layout="wide")
ASSISTANT_ON = assistant_switched_on()

# Visual styling (colours match .streamlit/config.toml)
BRAND, BRAND_DARK, BRAND_LIGHT, ACCENT = "#0F7B5C", "#0B5A44", "#E8F3EF", "#F2A541"
st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; max-width: 1200px; }}
    html, body, p, li, label, input, textarea, button, h1, h2, h3, h4, h5, h6,
    [data-baseweb="select"] > div, [data-baseweb="tab"], [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"], [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {{
        font-family: "Source Sans Pro", "Source Sans 3", -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
    .sk-header {{ background: linear-gradient(90deg, {BRAND_DARK}, {BRAND}); color: white;
                  padding: 1.2rem 1.6rem; border-radius: 12px; margin-bottom: 1rem; }}
    .sk-header h1 {{ color: white; margin: 0; font-size: 2rem; }}
    .sk-header p {{ color: #E6F2EE; margin: 0.3rem 0 0 0; font-size: 1rem; }}
    .sk-badge {{ display: inline-block; background: {ACCENT}; color: #1F2A44; font-weight: 600;
                 padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.8rem; margin-right: 0.4rem; }}
    .sk-step {{ color: {BRAND_DARK}; font-weight: 700; font-size: 1.1rem; margin: 0.6rem 0 0.2rem 0; }}
    .sk-tier {{ display: inline-block; color: white; font-weight: 700; padding: 0.1rem 0.6rem; border-radius: 999px; }}
    div[data-testid="stMetric"] {{ background: {BRAND_LIGHT}; border-left: 6px solid {BRAND};
                                   padding: 0.8rem 1rem; border-radius: 10px; }}
    div[data-testid="stMetricValue"] {{ color: {BRAND_DARK}; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
<div class="sk-header">
    <h1>🛒 SuperKart Planning Hub</h1>
    <p>How much revenue will a product earn at a SuperKart outlet - and how much should we stock?
       For procurement, inventory and range planning.{" Ask about the sales data in plain English." if ASSISTANT_ON else ""}</p>
    <p style="margin-top:0.6rem;"><span class="sk-badge">XGBoost model</span>
       <span class="sk-badge">~4% average error on unseen products</span>
       <span class="sk-badge">4 outlets</span></p>
</div>
""", unsafe_allow_html=True)
with st.expander("How to use this tool"):
    st.markdown(
        "- **Forecast:** choose the outlet and describe the product as it is (its actual price, weight and "
        "display share). You get the forecast revenue, how it ranks among the outlet's products, and a suggested "
        "stock range.\n"
        "- **Stock planning (upload):** upload a CSV of the products you plan to stock to forecast them all at once, with each product's tier "
        "at its outlet, a stock range and the total forecast revenue per outlet.\n"
        + ("- **Ask SuperKart (first tab):** ask in plain English about the historical sales data or a product to forecast. The assistant calls the same model and "
           "explains the result; every figure it gives is checked against the model's output.\n" if ASSISTANT_ON else "")
        + "\n"
        "*Money* (MRP and revenue) is shown in US dollars (\\$).\n\n"
        "*What a forecast means:* the revenue a product is expected to earn at the outlet **over the same period as the "
        "historical sales data**, based on patterns in past sales. The data has no dates, so it is not a dated weekly or "
        "quarterly figure, and it is **not a pricing tool** - it describes products as they are, not what would happen "
        "if a price changed."
    )

# Tabs: the planning assistant gets its own full-width tab when it is switched on (room for answers and charts)
if ASSISTANT_ON:
    chat_tab, single_tab, batch_tab = st.tabs(["💬 Ask SuperKart", "Forecast", "Stock planning (upload)"])
else:
    chat_tab = None
    single_tab, batch_tab = st.tabs(["Forecast", "Stock planning (upload)"])

# ---------------------------------------- Planning assistant (chat tab) -------------------------
# The assistant (Claude via the Anthropic API) turns a plain-English question into the model's inputs, calls the same
# forecasting API and the backend's data endpoints as tools, and explains the result. Every number must come from a tool;
# the app checks this and flags anything it cannot trace. Charts are drawn by code from the tool results.
EXAMPLES = ["What are the sales of Frozen Foods in all outlets?",
            "Describe OUT002.",
            "Is MRP linked to revenue at each outlet?",
            "How many units of a snack food at MRP $150, weight 11 should OUT002 stock?"]
if ASSISTANT_ON:
    with chat_tab:
        st.caption("Ask about SuperKart's historical sales data, or forecast a product and how much to stock. "
                   "Out of scope: dated forecasts, price what-ifs, profit, cause and effect, other stores.")
        if ASSISTANT_MODE == "preview":
            st.warning("Preview - only visible through the preview link.")
        st.session_state.setdefault("chat_history", [])     # messages sent to the LLM
        st.session_state.setdefault("chat_display", [])     # (role, text, tool calls, untraced figures)
        for i, (role, text, calls, untraced) in enumerate(st.session_state["chat_display"]):
            with st.chat_message(role):
                st.markdown(md_safe(text))
                if role == "assistant":
                    for chart in answer_charts(calls):
                        st.altair_chart(chart, use_container_width=True)
                    trace_badge(untraced)
                    read_aloud(text, f"chat{i}")
                    if calls:
                        with st.expander(f"How this was answered ({len(calls)} step{'s' if len(calls) > 1 else ''})"):
                            for c in calls:
                                st.markdown(f"**{c['tool']}** `{json.dumps(c['input'])}`")
                                st.json(c["result"], expanded=False)
        clicked = None
        app_results = [q for q, key in (("Explain my forecast from the Forecast tab.", "context_forecast"),
                                        ("Explain my stock plan from the Stock planning tab.", "context_stock_plan"))
                       if st.session_state.get(key)]
        if app_results:
            st.markdown("**About your work in the other tabs:**")
            for column, example in zip(st.columns(len(app_results)), app_results):
                if column.button(example, use_container_width=True, key=f"ctx_{example}"):
                    clicked = example
        if not st.session_state["chat_display"]:
            st.markdown("**Try one of these:**")
            for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
                if column.button(example, use_container_width=True):
                    clicked = example
        with st.form("ask", clear_on_submit=True):
            question = st.text_area("Your question", height=110, label_visibility="collapsed",
                                    placeholder="Ask about sales, outlets, categories or prices - or a product to forecast")
            ask_col, clear_col = st.columns([4, 1])
            asked = ask_col.form_submit_button("Ask", type="primary", use_container_width=True)
            cleared = clear_col.form_submit_button("Clear chat", use_container_width=True)
        if cleared:
            st.session_state["chat_history"], st.session_state["chat_display"] = [], []
            st.rerun()
        question = clicked or (question.strip() if asked and question else "")
        if question:
            try:
                with st.spinner("Checking the data and the model..."):
                    history = list(core.trim_history(st.session_state["chat_history"]))
                    result = core.run_agent(claude_client(), history, question,
                                            context={"forecast": st.session_state.get("context_forecast"),
                                                     "stock_plan": st.session_state.get("context_stock_plan")})
            except Exception as error:          # API or network errors must not break the app
                st.session_state["chat_history"] = []          # start the conversation memory afresh after an error
                assistant_error(error)
            else:
                st.session_state["chat_history"] = history
                st.session_state["chat_display"] += [("user", question, [], []),
                                                     ("assistant", result["answer"], result["calls"], result["untraced"])]
                st.rerun()

# ---------------------------------------- Forecast (one product) -------------------------------
with single_tab:
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="sk-step">1 · Outlet</div>', unsafe_allow_html=True)
        store_id = st.selectbox(
            "SuperKart outlet", list(STORES),
            index=3,
            format_func=lambda s: f"{s} - {STORES[s]['Store_Type']}, {STORES[s]['Store_Size']} size, "
                                  f"{STORES[s]['Store_Location_City_Type']} city (opened {STORES[s]['Established']})",
        )
        store = STORES[store_id]

        st.markdown('<div class="sk-step">2 · Describe the product</div>', unsafe_allow_html=True)
        st.caption("One product (one SKU) - an existing product or one you are considering stocking.")
        product_name = st.text_input("Product name or ID (optional)", value="", help="Only used to label the result.")
        product_type = st.selectbox("Product category", PRODUCT_TYPES, index=PRODUCT_TYPES.index("Frozen Foods"),
                                    help="Sets the product family (food / drink / non-consumable) and the perishable "
                                         "flag the model uses, and the allowed weight and MRP range.")
        limits = TYPE_RANGES[product_type]
        clamp = lambda v, rng: min(max(v, rng[0]), rng[1])
        sugar = st.selectbox("Sugar content", sugar_options(product_type),
                             help="Household, health & hygiene and other non-food items are always 'No Sugar'.")
        mrp = st.number_input(f"Maximum retail price (MRP, \\$) - \\${limits['mrp'][0]:g} to \\${limits['mrp'][1]:g} for {product_type}",
                              min_value=float(limits["mrp"][0]), max_value=float(limits["mrp"][1]),
                              value=float(clamp(117.08, limits["mrp"])), step=1.0,
                              help=f"Enter the product's actual MRP. Limited to the range sold for {product_type}; "
                                   f"prices sold at {store_id} range from \\${store['mrp'][0]:.0f} to \\${store['mrp'][1]:.0f}.")
        weight = st.number_input(f"Product weight (unit as in the source data) - {limits['weight'][0]:g} to {limits['weight'][1]:g}",
                                 min_value=float(limits["weight"][0]), max_value=float(limits["weight"][1]),
                                 value=float(clamp(12.66, limits["weight"])), step=0.1,
                                 help=f"Limited to the range sold for {product_type}; weights sold at {store_id} range "
                                      f"from {store['weight'][0]:.1f} to {store['weight'][1]:.1f}.")
        area_low, area_high = (round(100 * v, 1) for v in TRAINED_RANGES["Product_Allocated_Area"])
        area_pct = st.number_input(f"Share of the outlet's display area (%) - {area_low:g} to {area_high:g}",
                                   min_value=area_low, max_value=area_high,
                                   value=2.7, step=0.1, help="The product's share of the store's total display area.")
        outside = []
        if not store["mrp"][0] <= mrp <= store["mrp"][1]:
            outside.append("MRP")
        if not store["weight"][0] <= weight <= store["weight"][1]:
            outside.append("weight")
        if outside:
            st.warning(f"The {' and '.join(outside)} is within the range for {product_type} but outside what "
                       f"{store_id} sells; the forecast is less reliable.")

    row = model_row(store_id, product_type, sugar, weight, mrp, area_pct / 100)

    with right:
        st.markdown('<div class="sk-step">3 · Forecast and stocking guidance</div>', unsafe_allow_html=True)
        if st.button("Forecast sales revenue", type="primary"):
            try:
                response = requests.post(f"{BACKEND_URL}/v1/predict", json=row, timeout=90)
                response.raise_for_status()
                st.session_state["forecast"] = (row, response.json()["Sales"])
            except requests.exceptions.RequestException as error:
                st.error(f"The forecasting service could not be reached: {error}. If it has been idle it may be waking up - please wait a minute and try again.")

        current = st.session_state.get("forecast")
        if current and current[0] == row:
            forecast = current[1]
            label = f" ({product_name.strip()})" if product_name.strip() else ""
            st.metric(f"Forecast revenue for this product{label} at {store_id}", f"${forecast:,.2f}")
            st.caption("Over the same period as the historical sales data.")

            # Where it ranks among the outlet's products
            tier = tier_of(store_id, forecast)
            colour = TIER_COLOUR[tier]
            position = outlet_position(store_id, forecast)
            vs_avg = forecast / store["avg"] - 1
            st.markdown(
                f'<span class="sk-tier" style="background:{colour}">{tier} at {store_id}</span>&nbsp; '
                f"Higher than about **{position:.0f}%** of {store_id}'s products; "
                f"**{vs_avg:+.0%}** vs its average product (\\${store['avg']:,}).",
                unsafe_allow_html=True)

            # Suggested stock range in units
            units, low, high = stock_range(store_id, forecast, mrp)
            perishable = product_type in PERISHABLES
            st.info(
                f"**Suggested stock: about {units:.0f} units** (range {low:.0f}-{high:.0f}).  \n"
                f"Forecast ± {store_id}'s 90th-percentile error ({store['err90']:.1%}), converted to units at MRP "
                f"\\${mrp:,.2f}. "
                + ("Perishable - stock near the middle of the range." if perishable
                   else "Long-life product - the upper end of the range is lower risk.")
            )
            if store["mape"] > 0.10:
                st.warning(f"Lower accuracy at {store_id}: average error about ±{store['mape']:.0%}. "
                           "Keep a safety margin and review the order by hand.")
            else:
                st.caption(f"Expected accuracy at {store_id}: average error ±{store['mape']:.1%}; "
                           f"9 in 10 products within ±{store['err90']:.1%}.")

            # The result as facts - for "Explain this result" and for the chat assistant ("explain my forecast")
            facts = {"tab": "Forecast", "outlet": store_id, "product_type": product_type, "sugar_content": sugar,
                     "mrp": mrp, "weight": weight, "display_share_pct": area_pct,
                     "outlet_profile": {k: store[k] for k in ("Store_Type", "Store_Size", "Store_Location_City_Type",
                                                              "Established")},
                     **planning_facts(store_id, product_type, mrp, forecast)}
            st.session_state["context_forecast"] = facts

            # Explain this result in plain English (Claude), and read it aloud
            if ASSISTANT_ON:
                explain_key = json.dumps(row, sort_keys=True)
                if st.button("✨ Explain this result"):
                    try:
                        with st.spinner("Writing the explanation..."):
                            st.session_state["explanation"] = (explain_key, core.explain(claude_client(), facts))
                    except Exception as error:
                        assistant_error(error)
                explanation = st.session_state.get("explanation")
                if explanation and explanation[0] == explain_key and explanation[1].get("empty"):
                    st.info("The explanation could not be written this time - please click the button again.")
                elif explanation and explanation[0] == explain_key:
                    st.markdown(md_safe(explanation[1]["answer"]))
                    trace_badge(explanation[1]["untraced"])
                    read_aloud(explanation[1]["answer"], "single")
        else:
            st.caption("Set the outlet and product, then click **Forecast sales revenue**.")

    # Where this product sits among the outlet's actual products
    current = st.session_state.get("forecast")
    if current and current[0] == row:
        st.divider()
        st.subheader(f"Where this product sits among {store_id}'s products")
        dist = OUTLET_DIST[store_id]
        bars = pd.DataFrame(dist["hist"], columns=["start", "end", "products"])
        middle = (bars["start"] + bars["end"]) / 2
        bars["zone"] = ["Bottom 20%" if m < dist["p20"] else "Top 20%" if m >= dist["p80"] else "Middle 60%" for m in middle]
        zone_scale = alt.Scale(domain=["Bottom 20%", "Middle 60%", "Top 20%"], range=[ACCENT, "#B8C4CF", BRAND])
        bars["zero"] = 0
        histogram = alt.Chart(bars).mark_rect().encode(
            x=alt.X("start:Q", title="Revenue per product at the outlet ($, actual, historical)"),
            x2="end:Q",
            y=alt.Y("products:Q", title="Number of products"),
            y2="zero:Q",
            color=alt.Color("zone:N", scale=zone_scale, legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("start:Q", title="From ($)", format=",.0f"), alt.Tooltip("end:Q", title="To ($)", format=",.0f"),
                     alt.Tooltip("products:Q", title="Products")])
        marker = alt.Chart(pd.DataFrame({"x": [current[1]], "label": ["This product"]})).mark_rule(
            color=BRAND_DARK, strokeDash=[6, 4], size=3).encode(x="x:Q", tooltip=[alt.Tooltip("x:Q", title="Forecast ($)", format=",.0f")])
        st.altair_chart(histogram + marker, use_container_width=True)
        st.caption(f"Bars: actual revenue of {store_id}'s products in the historical data, coloured by its bottom fifth, "
                   "middle and top fifth. Dashed line: the forecast for this product. Top-fifth products are candidates "
                   "for deeper stock; bottom-fifth products for leaner stock or review.")

# ---------------------------------------- Stock planning (upload) ------------------------------
with batch_tab:
    st.subheader("Stock planning for a list of products")
    st.write("Upload the products you plan to stock - a CSV with one row per product and outlet - to get each "
             "product's forecast, its tier at the outlet and a stock range, plus the total forecast revenue per outlet "
             "(the basis for each outlet's procurement budget). Columns needed (an optional `Store_Id` or `Product_Id` "
             "column is kept in the results):")
    st.code(", ".join(MODEL_COLUMNS), language=None)
    template = pd.DataFrame([model_row(s, "Frozen Foods", "Low Sugar", 12.66, 117.08, 0.027) for s in STORES])
    template.insert(0, "Store_Id", list(STORES))
    st.download_button("Download a CSV template", template.to_csv(index=False), "superkart_batch_template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        batch = pd.read_csv(uploaded_file)
        missing = [c for c in MODEL_COLUMNS if c not in batch.columns]
        if missing:
            st.error(f"These required columns are missing: {', '.join(missing)}")
        elif st.button("Forecast batch", type="primary"):
            try:
                batch["Forecast_Revenue"] = [round(p, 2) for p in predict_batch(batch)]
                outlets = []
                for _, r in batch.iterrows():
                    given = r.get("Store_Id") if "Store_Id" in batch.columns else None
                    outlets.append(given if given in STORES else match_outlet(r))
                batch["Outlet"] = [o or "not a SuperKart outlet" for o in outlets]
                in_range = pd.Series(True, index=batch.index)
                for column, (low, high) in TRAINED_RANGES.items():
                    in_range &= batch[column].between(low, high)
                batch["Data_Check"] = ["OK" if ok else "Outside range" for ok in in_range]
                tiers, units, ranges = [], [], []
                for o, f, m, ok in zip(outlets, batch["Forecast_Revenue"], batch["Product_MRP"], in_range):
                    if not ok:
                        tiers.append("n/a"); units.append("–"); ranges.append("n/a")
                    elif o:
                        u, lo, hi = stock_range(o, f, m)
                        tiers.append(tier_of(o, f)); units.append(str(round(u))); ranges.append(f"{lo:.0f}-{hi:.0f}")
                    else:
                        tiers.append("n/a"); units.append(str(round(f / m))); ranges.append("n/a")
                batch["Tier_at_Outlet"] = tiers
                batch["Units_Estimate"] = units
                batch["Stock_Range_Units"] = ranges
                first = [c for c in ["Outlet", "Product_Id", "Forecast_Revenue", "Tier_at_Outlet", "Units_Estimate",
                                     "Stock_Range_Units", "Data_Check"] if c in batch.columns]
                batch = batch[first + [c for c in batch.columns if c not in first]]
                summary = batch.groupby("Outlet").agg(
                    Products=("Forecast_Revenue", "size"),
                    Total_Forecast_Revenue=("Forecast_Revenue", "sum"),
                    Top_20=("Tier_at_Outlet", lambda t: int((t == "Top 20%").sum())),
                    Bottom_20=("Tier_at_Outlet", lambda t: int((t == "Bottom 20%").sum())),
                ).round(2)
                st.session_state["batch_result"] = (uploaded_file.file_id, batch, summary)   # kept for the planning note
            except requests.exceptions.RequestException as error:
                st.error(f"The forecasting service could not be reached: {error}. If it has been idle it may be waking up - please wait a minute and try again.")

        saved = st.session_state.get("batch_result")
        if saved and saved[0] == uploaded_file.file_id:
            _, batch, summary = saved
            st.success(f"Forecast {len(batch)} products - total forecast revenue \\${batch['Forecast_Revenue'].sum():,.2f} "
                       "(over the same period as the historical data)")
            st.dataframe(batch, use_container_width=True,
                         column_config={"Forecast_Revenue": st.column_config.NumberColumn("Forecast revenue ($)", format="$%.2f"),
                                        "Tier_at_Outlet": "Tier at outlet", "Units_Estimate": "≈ Units",
                                        "Stock_Range_Units": "Stock range (units)", "Data_Check": "Check"})
            st.write("**Forecast revenue by outlet** - the basis for each outlet's procurement budget")
            summary_chart = alt.Chart(summary.reset_index()).mark_bar(color=BRAND).encode(
                x=alt.X("Total_Forecast_Revenue:Q", title="Total forecast revenue ($)"),
                y=alt.Y("Outlet:N", sort="-x", title=None),
                tooltip=["Outlet", "Products", alt.Tooltip("Total_Forecast_Revenue:Q", title="Total forecast revenue ($)", format="$,.2f")])
            st.altair_chart(summary_chart, use_container_width=True)
            st.dataframe(summary, use_container_width=True,
                         column_config={"Total_Forecast_Revenue": st.column_config.NumberColumn("Total forecast revenue ($)", format="$%.2f"),
                                        "Top_20": "Top-20% products", "Bottom_20": "Bottom-20% products"})
            if (batch["Data_Check"] != "OK").any():
                st.error("Some rows have a weight, MRP or display share outside the range the model was trained on "
                         f"(weight {TRAINED_RANGES['Product_Weight'][0]:g}-{TRAINED_RANGES['Product_Weight'][1]:g}, "
                         f"MRP \\${TRAINED_RANGES['Product_MRP'][0]:g}-\\${TRAINED_RANGES['Product_MRP'][1]:g}, display share "
                         f"{100 * TRAINED_RANGES['Product_Allocated_Area'][0]:g}-{100 * TRAINED_RANGES['Product_Allocated_Area'][1]:g}%); "
                         "their forecasts are not reliable.")
            if (batch["Outlet"] == "not a SuperKart outlet").any():
                st.warning("Some rows describe a store profile that is not one of SuperKart's four outlets; "
                           "their forecasts are extrapolations, so no tier or stock range is given.")
            st.download_button("Download results (CSV)", batch.to_csv(index=False), "superkart_forecasts.csv", "text/csv")

            # The plan as facts - for the planning note and for the chat assistant ("explain my stock plan")
            row_columns = [c for c in ["Outlet", "Product_Id", "Forecast_Revenue", "Tier_at_Outlet", "Units_Estimate",
                                       "Stock_Range_Units", "Data_Check", "Product_MRP", "Product_Weight",
                                       "Product_Type_Category", "Product_Id_char"] if c in batch.columns]
            facts = {"tab": "Stock planning (upload)", "file": uploaded_file.name,
                     "total_forecast_revenue": round(float(batch["Forecast_Revenue"].sum()), 2),
                     "products": len(batch),
                     "by_outlet": json.loads(summary.reset_index().to_json(orient="records")),   # plain JSON types
                     "rows_outside_trained_range": int((batch["Data_Check"] != "OK").sum()),
                     "rows_not_a_superkart_outlet": int((batch["Outlet"] == "not a SuperKart outlet").sum()),
                     "outlet_accuracy": {s: {"average_error_pct": round(100 * STORES[s]["mape"], 1),
                                             "error_9_in_10_within_pct": round(100 * STORES[s]["err90"], 1)}
                                         for s in summary.index if s in STORES},
                     "stock_range_rule": "forecast +/- the outlet's 9-in-10 error, divided by MRP (units)",
                     "product_rows": json.loads(batch[row_columns].head(60).to_json(orient="records")),
                     "rows_shown": min(60, len(batch))}
            st.session_state["context_stock_plan"] = facts

            # Planning note (Claude) from the per-outlet summary - numbers only from these results
            if ASSISTANT_ON:
                if st.button("✨ Write a planning note"):
                    try:
                        with st.spinner("Writing the planning note..."):
                            st.session_state["plan_note"] = (uploaded_file.file_id,
                                                             core.explain(claude_client(), facts, kind="batch"))
                    except Exception as error:
                        assistant_error(error)
                note = st.session_state.get("plan_note")
                if note and note[0] == uploaded_file.file_id and note[1].get("empty"):
                    st.info("The planning note could not be written this time - please click the button again.")
                elif note and note[0] == uploaded_file.file_id:
                    st.markdown(md_safe(note[1]["answer"]))
                    trace_badge(note[1]["untraced"])
                    read_aloud(note[1]["answer"], "batch")

# ---------------------------------------- Reliability ------------------------------------------
st.divider()
with st.expander("How reliable are these forecasts?"):
    overall = REFERENCE["overall"]
    st.markdown(f"Measured on **{overall['test_rows']:,} products the model never saw** during training. Planning every "
                f"product from its outlet's average would be off by **{overall['store_average_mape']:.1%}** on average; "
                f"the model is off by **{overall['model_mape']:.1%}**.")
    st.dataframe(pd.DataFrame({
        "Outlet": list(STORES),
        "Test products": [STORES[s]["test_rows"] for s in STORES],
        "Average error": [f"±{STORES[s]['mape']:.1%}" for s in STORES],
        "9 in 10 products within": [f"±{STORES[s]['err90']:.1%}" for s in STORES],
    }).set_index("Outlet"), use_container_width=True)
    st.caption("Forecasts describe products as they are, using historical patterns - not cause and effect, not a pricing "
               "tool, and revenue rather than profit. Store profiles that are not one of the four outlets are extrapolations.")
