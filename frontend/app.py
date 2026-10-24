import io
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
# Backend address: the Flask container on the shared Docker network (override with BACKEND_URL)
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
    response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": ("batch.csv", csv_bytes)}, timeout=30)
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
                response = requests.post(f"{BACKEND_URL}/v1/predict", json=row, timeout=30)
                response.raise_for_status()
                st.session_state["forecast"] = (row, response.json()["Sales"])
            except requests.exceptions.RequestException as error:
                st.error(f"The forecasting service could not be reached: {error}")

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
            chart_data = pd.DataFrame({"input_value": grid, "forecast": predictions})
            observed_low, observed_high = (v * spec["scale"] for v in store[spec["store_key"]])
            band = alt.Chart(pd.DataFrame({"start": [observed_low], "end": [observed_high]})).mark_rect(
                opacity=0.18, color=BRAND).encode(x="start:Q", x2="end:Q")
            line = alt.Chart(chart_data).mark_line(point=True, color=BRAND_DARK).encode(
                x=alt.X("input_value:Q", title=choice),
                y=alt.Y("forecast:Q", title="Forecast sales revenue", scale=alt.Scale(zero=False)))
            current_value = row[spec["column"]] * spec["scale"]
            marker = alt.Chart(pd.DataFrame({"x": [current_value]})).mark_rule(color=ACCENT, strokeDash=[6, 4], size=2).encode(x="x:Q")
            st.altair_chart(band + line + marker, use_container_width=True)
            spread = max(predictions) - min(predictions)
            st.caption(
                f"Shaded band: range of {choice.lower()} actually sold at {store_id}; outside it the model has no data "
                f"and cannot extrapolate. Dashed orange line: the current input. Across the full range the forecast moves by "
                f"{spread:,.0f}. This shows the model's sensitivity with all other inputs held fixed - not cause and "
                "effect - and it is revenue, not profit."
            )
        except requests.exceptions.RequestException as error:
            st.error(f"The forecasting service could not be reached: {error}")

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
                st.error(f"The forecasting service could not be reached: {error}")
