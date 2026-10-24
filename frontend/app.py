
import io
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
# Backend address: the Flask container on the shared Docker network (Codespaces), or the backend
# Hugging Face Space URL set as the BACKEND_URL variable of the frontend Space
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

# The ten inputs the deployed model pipeline expects (same columns as the batch file)
MODEL_COLUMNS = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Store_Age_Years", "Product_Type_Category",
]

# ----------------------------------------------------------------------------------------------
# Reference data from the training set (SuperKart.csv) and the model evaluation
# ----------------------------------------------------------------------------------------------
# HELPERS-START
STORES = {
    "OUT001": {"Store_Type": "Supermarket Type1", "Store_Size": "High", "Store_Location_City_Type": "Tier 2",
               "Established": 1987, "avg": 3924, "p10": 2388, "p90": 4807, "mape": 0.037,
               "weight": (6.16, 17.97), "mrp": (71.35, 226.59), "area": (0.004, 0.295)},
    "OUT002": {"Store_Type": "Food Mart", "Store_Size": "Small", "Store_Location_City_Type": "Tier 3",
               "Established": 1998, "avg": 1763, "p10": 1076, "p90": 2240, "mape": 0.159,
               "weight": (4.00, 19.82), "mrp": (31.00, 224.93), "area": (0.006, 0.292)},
    "OUT003": {"Store_Type": "Departmental Store", "Store_Size": "Medium", "Store_Location_City_Type": "Tier 1",
               "Established": 1999, "avg": 4947, "p10": 4257, "p90": 5887, "mape": 0.025,
               "weight": (7.35, 22.00), "mrp": (85.88, 266.00), "area": (0.004, 0.298)},
    "OUT004": {"Store_Type": "Supermarket Type2", "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
               "Established": 2009, "avg": 3299, "p10": 2678, "p90": 3876, "mape": 0.024,
               "weight": (7.34, 17.79), "mrp": (83.04, 197.66), "area": (0.004, 0.297)},
}
REFERENCE_YEAR = 2025          # store age is computed as 2025 - establishment year, as in the model

PRODUCT_TYPES = [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables",
    "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods",
    "Soft Drinks", "Starchy Foods",
]
PERISHABLES = {"Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"}
DRINKS = {"Hard Drinks", "Soft Drinks"}
NON_CONSUMABLES = {"Health and Hygiene", "Household", "Others"}

WHAT_IF_INPUTS = {
    "Display area share (%)": {"column": "Product_Allocated_Area", "range": (0.4, 30.0), "scale": 100, "store_key": "area"},
    "Product MRP": {"column": "Product_MRP", "range": (31.0, 266.0), "scale": 1, "store_key": "mrp"},
    "Product weight": {"column": "Product_Weight", "range": (4.0, 22.0), "scale": 1, "store_key": "weight"},
}

# Actual average sales of each outlet's products by band of an input: [band midpoint, average sales, products]
# (from SuperKart.csv; bands with at least 10 products; display area midpoints in %)
OBSERVED_AVERAGES = {"OUT001": {"weight": [[9.0, 2606, 104], [11.0, 2753, 277], [13.0, 4072, 421], [15.0, 4440, 676], [17.0, 4502, 100]], "mrp": [[80.0, 2947, 21], [100.0, 2602, 104], [120.0, 2478, 212], [140.0, 3415, 130], [160.0, 4182, 380], [180.0, 4456, 502], [200.0, 4609, 219], [220.0, 4761, 18]], "area": [[0.95, 4020, 165], [3.0, 3910, 372], [5.0, 3886, 300], [7.0, 3922, 254], [9.0, 3965, 140], [11.5, 3954, 180], [15.0, 3856, 115], [19.5, 3877, 49], [26.0, 3903, 11]]}, "OUT002": {"weight": [[7.0, 1378, 134], [9.0, 1787, 490], [11.0, 1944, 413], [13.0, 1583, 70], [15.0, 1395, 31]], "mrp": [[60.0, 982, 49], [80.0, 1497, 185], [100.0, 1877, 474], [120.0, 1971, 312], [140.0, 1772, 62], [160.0, 1151, 33], [180.0, 1465, 18], [200.0, 1570, 13]], "area": [[0.95, 1797, 149], [3.0, 1749, 280], [5.0, 1770, 188], [7.0, 1736, 153], [9.0, 1744, 122], [11.5, 1830, 130], [15.0, 1677, 85], [19.5, 1855, 38]]}, "OUT003": {"weight": [[9.0, 6079, 22], [11.0, 5327, 46], [13.0, 4696, 265], [15.0, 4733, 592], [17.0, 5218, 358], [19.0, 5670, 56]], "mrp": [[100.0, 5862, 11], [120.0, 6025, 30], [140.0, 5138, 48], [160.0, 4566, 315], [180.0, 4707, 477], [200.0, 5152, 299], [220.0, 5551, 136], [240.0, 6130, 26]], "area": [[0.95, 4867, 185], [3.0, 4970, 290], [5.0, 4973, 239], [7.0, 4905, 177], [9.0, 5029, 155], [11.5, 4941, 155], [15.0, 4995, 92], [19.5, 4887, 43], [26.0, 4590, 13]]}, "OUT004": {"weight": [[9.0, 2876, 229], [11.0, 3099, 1657], [13.0, 3398, 2199], [15.0, 3662, 559], [17.0, 3689, 27]], "mrp": [[100.0, 2797, 149], [120.0, 2972, 970], [140.0, 3262, 1974], [160.0, 3566, 1315], [180.0, 3732, 252], [200.0, 3880, 13]], "area": [[0.95, 3299, 541], [3.0, 3292, 1175], [5.0, 3299, 744], [7.0, 3314, 650], [9.0, 3340, 478], [11.5, 3288, 513], [15.0, 3285, 379], [19.5, 3265, 162], [26.0, 3200, 34]]}}

def product_family(product_type):
    """Product family code used by the model (first two letters of Product_Id)."""
    if product_type in DRINKS:
        return "DR"
    if product_type in NON_CONSUMABLES:
        return "NC"
    return "FD"


def sugar_options(product_type):
    """Sugar labels that occur for this product type in the data (non-consumables are always 'No Sugar')."""
    return ["No Sugar"] if product_type in NON_CONSUMABLES else ["Low Sugar", "Regular"]


def model_row(store_id, product_type, sugar, weight, mrp, area_share):
    """Translate the user's choices into the ten model inputs."""
    store = STORES[store_id]
    return {
        "Product_Weight": float(weight),
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": float(area_share),
        "Product_MRP": float(mrp),
        "Store_Size": store["Store_Size"],
        "Store_Location_City_Type": store["Store_Location_City_Type"],
        "Store_Type": store["Store_Type"],
        "Product_Id_char": product_family(product_type),
        "Store_Age_Years": REFERENCE_YEAR - store["Established"],
        "Product_Type_Category": "Perishables" if product_type in PERISHABLES else "Non Perishables",
    }


def match_outlet(row):
    """Name of the SuperKart outlet with this store profile, or None for a hypothetical profile."""
    for store_id, store in STORES.items():
        if (row["Store_Type"], row["Store_Size"], row["Store_Location_City_Type"]) == (
                store["Store_Type"], store["Store_Size"], store["Store_Location_City_Type"]):
            return store_id
    return None
# HELPERS-END


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
st.set_page_config(page_title="SuperKart Sales Forecasting", page_icon="🛒", layout="wide")

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
    div[data-testid="stMetric"] {{ background: {BRAND_LIGHT}; border-left: 6px solid {BRAND};
                                   padding: 0.8rem 1rem; border-radius: 10px; }}
    div[data-testid="stMetricValue"] {{ color: {BRAND_DARK}; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
<div class="sk-header">
    <h1>🛒 SuperKart Sales Forecasting</h1>
    <p>Forecast the sales revenue a product generates at a SuperKart outlet - for inventory,
       procurement and shelf-space planning.</p>
    <p style="margin-top:0.6rem;"><span class="sk-badge">XGBoost model</span>
       <span class="sk-badge">~4% average error on unseen products</span>
       <span class="sk-badge">4 outlets</span></p>
</div>
""", unsafe_allow_html=True)
with st.expander("How to use this tool"):
    st.markdown(
        "- **Single product:** choose the outlet and describe the product. The forecast appears with the "
        "outlet's typical range and the expected accuracy for that outlet.\n"
        "- **What-if:** see how the forecast responds when one input changes (display area, price or weight).\n"
        "- **Batch:** upload a CSV of products to forecast many at once and get totals per outlet.\n\n"
        "*Note:* the historical data has no dates, so a forecast is a product's total revenue at the outlet on the "
        "same basis as past sales - not a dated weekly or quarterly figure."
    )

single_tab, batch_tab = st.tabs(["Single product forecast", "Batch forecast (CSV)"])

# ---------------------------------------- Single product ---------------------------------------
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

        st.markdown('<div class="sk-step">2 · Product</div>', unsafe_allow_html=True)
        product_type = st.selectbox("Product type", PRODUCT_TYPES, index=PRODUCT_TYPES.index("Frozen Foods"))
        sugar = st.selectbox("Sugar content", sugar_options(product_type),
                             help="Household, health & hygiene and other non-food items are always 'No Sugar'.")
        mrp = st.number_input("Maximum retail price (MRP)", min_value=1.0, max_value=500.0, value=117.08, step=1.0,
                              help=f"Prices sold at {store_id} range from {store['mrp'][0]:.0f} to {store['mrp'][1]:.0f}.")
        weight = st.number_input("Product weight", min_value=0.5, max_value=40.0, value=12.66, step=0.1,
                                 help=f"Weights sold at {store_id} range from {store['weight'][0]:.1f} to {store['weight'][1]:.1f}.")
        area_pct = st.number_input("Share of the outlet's display area (%)", min_value=0.1, max_value=50.0, value=2.7,
                                   step=0.1, help="The product's share of the store's total display area.")

    row = model_row(store_id, product_type, sugar, weight, mrp, area_pct / 100)

    with right:
        st.markdown('<div class="sk-step">3 · Forecast</div>', unsafe_allow_html=True)
        if st.button("Forecast sales revenue", type="primary"):
            try:
                response = requests.post(f"{BACKEND_URL}/v1/predict", json=row, timeout=90)
                response.raise_for_status()
                st.session_state["forecast"] = (row, response.json()["Sales"])
            except requests.exceptions.RequestException as error:
                st.error(f"The forecasting service could not be reached: {error}. If it has been idle it may be waking up - please wait a minute and try again.")

        if "forecast" in st.session_state and st.session_state["forecast"][0] == row:
            forecast = st.session_state["forecast"][1]
            st.metric("Forecast sales revenue for this product", f"{forecast:,.2f}")
            st.write(
                f"**Context for {store_id}:** an average product sells **{store['avg']:,}**; most products "
                f"(10th-90th percentile) sell between **{store['p10']:,}** and **{store['p90']:,}**."
            )
            if store["mape"] > 0.10:
                st.warning(f"Lower accuracy at {store_id}: average error about ±{store['mape']:.0%}. "
                           "Use this forecast with a safety margin.")
            else:
                st.info(f"Expected accuracy at {store_id}: average error about ±{store['mape']:.1%}.")
        else:
            st.caption("Set the outlet and product, then click **Forecast sales revenue**.")

    # ---------------------------------------- What-if ----------------------------------------
    st.divider()
    st.subheader("What-if: how does the forecast respond to one input?")
    choice = st.radio("Input to vary", list(WHAT_IF_INPUTS), horizontal=True)
    spec = WHAT_IF_INPUTS[choice]
    low, high = spec["range"]
    grid = [low + (high - low) * i / 39 for i in range(40)]
    if st.button("Draw what-if chart"):
        rows = []
        for value in grid:
            what_if_row = dict(row)
            what_if_row[spec["column"]] = value / spec["scale"]
            rows.append(what_if_row)
        try:
            predictions = predict_batch(pd.DataFrame(rows))
            observed_low, observed_high = (v * spec["scale"] for v in store[spec["store_key"]])
            chart_data = pd.DataFrame({"input_value": grid, "forecast": predictions})
            chart_data["inside"] = chart_data["input_value"].between(observed_low, observed_high)
            actual = pd.DataFrame(OBSERVED_AVERAGES[store_id][spec["store_key"]],
                                  columns=["input_value", "actual_average", "products"])

            # Both series as % change from their own average, so their trends can be compared on the same footing:
            # the line is ONE product (other inputs fixed); the squares average ALL the outlet's products in each band.
            forecast_base = chart_data.loc[chart_data["inside"], "forecast"].mean()
            chart_data["pct"] = 100 * (chart_data["forecast"] / forecast_base - 1)
            actual_base = (actual["actual_average"] * actual["products"]).sum() / actual["products"].sum()
            actual["pct"] = 100 * (actual["actual_average"] / actual_base - 1)

            # y-axis at least +/-10% so small effects look small
            values = pd.concat([chart_data["pct"], actual["pct"]])
            y_scale = alt.Scale(domain=[min(-10, values.min() - 2), max(10, values.max() + 2)])
            x_axis = alt.X("input_value:Q", title=choice)
            y_title = "% above / below average"

            band = alt.Chart(pd.DataFrame({"start": [observed_low], "end": [observed_high]})).mark_rect(
                opacity=0.15, color=BRAND).encode(x="start:Q", x2="end:Q")
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9CA3AF").encode(y=alt.Y("y:Q", scale=y_scale))
            outside_line = alt.Chart(chart_data).mark_line(color=BRAND_DARK, strokeDash=[4, 4], opacity=0.35).encode(
                x=x_axis, y=alt.Y("pct:Q", title=y_title, scale=y_scale))
            inside_line = alt.Chart(chart_data[chart_data["inside"]]).mark_line(point=True, color=BRAND_DARK, strokeWidth=3).encode(
                x=x_axis, y=alt.Y("pct:Q", scale=y_scale),
                tooltip=[alt.Tooltip("input_value:Q", title=choice, format=",.1f"),
                         alt.Tooltip("forecast:Q", title="Model forecast", format=",.0f"),
                         alt.Tooltip("pct:Q", title="% vs average forecast", format="+.1f")])
            actual_points = alt.Chart(actual).mark_point(shape="square", size=90, filled=True, color="#6B7280").encode(
                x=x_axis, y=alt.Y("pct:Q", scale=y_scale),
                tooltip=[alt.Tooltip("input_value:Q", title=f"Band centre ({choice})", format=",.1f"),
                         alt.Tooltip("actual_average:Q", title="Actual average sales", format=",.0f"),
                         alt.Tooltip("pct:Q", title="% vs outlet average", format="+.1f"),
                         alt.Tooltip("products:Q", title="Products in band")])
            current_value = row[spec["column"]] * spec["scale"]
            marker = alt.Chart(pd.DataFrame({"x": [current_value]})).mark_rule(
                color=ACCENT, strokeDash=[6, 4], size=2).encode(x="x:Q")
            st.altair_chart(band + zero + outside_line + inside_line + actual_points + marker, use_container_width=True)

            inside = chart_data[chart_data["inside"]]
            model_low, model_high = inside["pct"].min(), inside["pct"].max()
            actual_low, actual_high = actual["pct"].min(), actual["pct"].max()
            def pct(value):
                value = round(value)
                return "0%" if value == 0 else f"{value:+d}%"
            st.markdown(
                f"**Within the range sold at {store_id}:** the forecast for this product moves between "
                f"**{pct(model_low)} and {pct(model_high)}** of its average; actual average sales of the outlet's "
                f"products move between **{pct(actual_low)} and {pct(actual_high)}** across the bands."
            )
            if spec["store_key"] == "area" and max(abs(model_low), abs(model_high)) < 5:
                st.success("Display share has little effect on forecast revenue at this outlet - reallocating shelf space "
                           "alone is unlikely to move sales; test changes before rolling them out.")
            st.caption(
                "**Solid line:** model forecast for this product (other inputs fixed), within the range actually sold at the "
                "outlet (shaded); **faded dashed line:** outside that range, where the model has no data. **Grey squares:** "
                "actual average sales of all the outlet's products in each band. Both are shown as % above/below their own "
                "average, so compare their **direction and size of change**, not exact values. **Orange line:** your current input."
            )
            if spec["store_key"] in ("mrp", "weight"):
                other = "weight" if spec["store_key"] == "mrp" else "MRP"
                other_value = row["Product_Weight"] if spec["store_key"] == "mrp" else row["Product_MRP"]
                st.warning(
                    f"Read with care: in real products price and weight move together (correlation 0.53) and the model uses "
                    f"their product (MRP x weight). Here {other} stays fixed at {other_value:g}, so the line describes this "
                    f"product only, while the squares mix products of all {other} levels - the two can differ in shape."
                )
            st.caption("Model sensitivity with other inputs held fixed - not cause and effect - and revenue, not profit.")
        except requests.exceptions.RequestException as error:
            st.error(f"The forecasting service could not be reached: {error}. If it has been idle it may be waking up - please wait a minute and try again.")

# ---------------------------------------- Batch -----------------------------------------------
with batch_tab:
    st.subheader("Forecast many products at once")
    st.write("Upload a CSV with one row per product and these columns (an optional `Store_Id` or "
             "`Product_Id` column is kept in the results):")
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
                if "Store_Id" not in batch.columns:
                    batch.insert(0, "Matched_Outlet", [match_outlet(r) or "not a SuperKart outlet" for _, r in batch.iterrows()])
                st.success(f"Forecast {len(batch)} products - total forecast revenue {batch['Forecast_Revenue'].sum():,.2f}")
                st.dataframe(batch, use_container_width=True)

                outlet_column = "Store_Id" if "Store_Id" in batch.columns else "Matched_Outlet"
                summary = batch.groupby(outlet_column)["Forecast_Revenue"].agg(Products="count", Total_Forecast_Revenue="sum")
                st.write("**Forecast revenue by outlet**")
                st.dataframe(summary.round(2), use_container_width=True)
                if "Matched_Outlet" in batch.columns and (batch["Matched_Outlet"] == "not a SuperKart outlet").any():
                    st.warning("Some rows describe a store profile that is not one of SuperKart's four outlets; "
                               "their forecasts are extrapolations and less reliable.")

                st.download_button("Download results (CSV)", batch.to_csv(index=False), "superkart_forecasts.csv", "text/csv")
            except requests.exceptions.RequestException as error:
                st.error(f"The forecasting service could not be reached: {error}. If it has been idle it may be waking up - please wait a minute and try again.")
