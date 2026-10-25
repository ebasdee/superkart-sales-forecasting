
import io
import json
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
# (mape / err90 / test_rows: average and 90th-percentile % error of the final model on the test set, per outlet)
# ----------------------------------------------------------------------------------------------
# HELPERS-START
# All reference numbers are produced by the notebook ("Reference data for the frontend") from SuperKart.csv and the
# test-set results, and shipped next to this file - nothing is typed in by hand.
REFERENCE_FILE = os.getenv("REFERENCE_FILE", "reference_data.json")
with open(REFERENCE_FILE) as reference_file:
    REFERENCE = json.load(reference_file)
# Per outlet: profile, revenue per product (avg, p10, p90), ranges sold, and the final model's test-set error
# (mape = average % error, err90 = 90th-percentile % error, test_rows = test products)
STORES = REFERENCE["stores"]
REFERENCE_YEAR = 2025          # store age is computed as 2025 - establishment year, as in the model

PRODUCT_TYPES = [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables",
    "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods",
    "Soft Drinks", "Starchy Foods",
]
PERISHABLES = {"Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"}
DRINKS = {"Hard Drinks", "Soft Drinks"}
NON_CONSUMABLES = {"Health and Hygiene", "Household", "Others"}

# Range of weight and MRP actually sold for each product type - the form will not accept values outside these limits,
# because the model has no data there (a tree model would silently treat them as the edge value)
TYPE_RANGES = REFERENCE["type_ranges"]
TRAINED_RANGES = REFERENCE["trained_ranges"]


# Each outlet's actual revenue per product: percentiles 0-100 in steps of 5, top/bottom-fifth thresholds and a
# 25-bin histogram [from, to, products]
OUTLET_DIST = REFERENCE["outlet_dist"]


def outlet_position(store_id, value):
    """Share (%) of the outlet's products whose actual revenue is below this value (historical data)."""
    pct = OUTLET_DIST[store_id]["pct"]
    if value <= pct[0]:
        return 0.0
    if value >= pct[-1]:
        return 100.0
    for i in range(1, len(pct)):
        if value <= pct[i]:
            low, high = pct[i - 1], pct[i]
            fraction = 0 if high == low else (value - low) / (high - low)
            return 5 * (i - 1 + fraction)
    return 100.0


def tier_of(store_id, value):
    """Top 20% / Middle 60% / Bottom 20% of the outlet's products, with a badge colour."""
    d = OUTLET_DIST[store_id]
    if value >= d["p80"]:
        return "Top 20%", "#0F7B5C"
    if value < d["p20"]:
        return "Bottom 20%", "#B86E12"
    return "Middle 60%", "#64748B"


def stock_range(store_id, forecast, mrp):
    """Units at MRP, and the range forecast +/- the outlet's 90th-percentile error."""
    error = STORES[store_id]["err90"]
    units = forecast / mrp
    return units, units * (1 - error), units * (1 + error)

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
    .sk-tier {{ display: inline-block; color: white; font-weight: 700; padding: 0.1rem 0.6rem; border-radius: 999px; }}
    div[data-testid="stMetric"] {{ background: {BRAND_LIGHT}; border-left: 6px solid {BRAND};
                                   padding: 0.8rem 1rem; border-radius: 10px; }}
    div[data-testid="stMetricValue"] {{ color: {BRAND_DARK}; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
<div class="sk-header">
    <h1>🛒 SuperKart Sales Forecasting</h1>
    <p>How much revenue will a product earn at a SuperKart outlet - and how much should we stock?
       For procurement, inventory and range planning.</p>
    <p style="margin-top:0.6rem;"><span class="sk-badge">XGBoost model</span>
       <span class="sk-badge">~4% average error on unseen products</span>
       <span class="sk-badge">4 outlets</span></p>
</div>
""", unsafe_allow_html=True)
with st.expander("How to use this tool"):
    st.markdown(
        "- **Single product:** choose the outlet and describe the product as it is (its actual price, weight and "
        "display share). You get the forecast revenue, how it ranks among the outlet's products, and a suggested "
        "stock range.\n"
        "- **Plan a product list:** upload a CSV of products to forecast them all at once, with each product's tier "
        "at its outlet, a stock range and the total forecast revenue per outlet.\n\n"
        "*What a forecast means:* the revenue a product is expected to earn at the outlet **over the same period as the "
        "historical sales data**, based on patterns in past sales. The data has no dates, so it is not a dated weekly or "
        "quarterly figure, and it is **not a pricing tool** - it describes products as they are, not what would happen "
        "if a price changed."
    )

single_tab, batch_tab = st.tabs(["Single product", "Plan a product list (CSV)"])

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
        mrp = st.number_input(f"Maximum retail price (MRP) - {limits['mrp'][0]:g} to {limits['mrp'][1]:g} for {product_type}",
                              min_value=float(limits["mrp"][0]), max_value=float(limits["mrp"][1]),
                              value=float(clamp(117.08, limits["mrp"])), step=1.0,
                              help=f"Enter the product's actual MRP. Limited to the range sold for {product_type}; "
                                   f"prices sold at {store_id} range from {store['mrp'][0]:.0f} to {store['mrp'][1]:.0f}.")
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
            st.metric(f"Forecast revenue for this product{label} at {store_id}", f"{forecast:,.2f}")
            st.caption("Over the same period as the historical sales data.")

            # Where it ranks among the outlet's products
            tier, colour = tier_of(store_id, forecast)
            position = outlet_position(store_id, forecast)
            vs_avg = forecast / store["avg"] - 1
            st.markdown(
                f'<span class="sk-tier" style="background:{colour}">{tier} at {store_id}</span>&nbsp; '
                f"Higher than about **{position:.0f}%** of {store_id}'s products; "
                f"**{vs_avg:+.0%}** vs its average product ({store['avg']:,}).",
                unsafe_allow_html=True)

            # Suggested stock range in units
            units, low, high = stock_range(store_id, forecast, mrp)
            perishable = product_type in PERISHABLES
            st.info(
                f"**Suggested stock: about {units:.0f} units** (range {low:.0f}-{high:.0f}).  \n"
                f"Forecast ± {store_id}'s 90th-percentile error ({store['err90']:.1%}), converted to units at MRP "
                f"{mrp:,.2f}. "
                + ("Perishable - stock near the middle of the range." if perishable
                   else "Long-life product - the upper end of the range is lower risk.")
            )
            if store["mape"] > 0.10:
                st.warning(f"Lower accuracy at {store_id}: average error about ±{store['mape']:.0%}. "
                           "Keep a safety margin and review the order by hand.")
            else:
                st.caption(f"Expected accuracy at {store_id}: average error ±{store['mape']:.1%}; "
                           f"9 in 10 products within ±{store['err90']:.1%}.")
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
            x=alt.X("start:Q", title="Revenue per product at the outlet (actual, historical)"),
            x2="end:Q",
            y=alt.Y("products:Q", title="Number of products"),
            y2="zero:Q",
            color=alt.Color("zone:N", scale=zone_scale, legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("start:Q", title="From", format=",.0f"), alt.Tooltip("end:Q", title="To", format=",.0f"),
                     alt.Tooltip("products:Q", title="Products")])
        marker = alt.Chart(pd.DataFrame({"x": [current[1]], "label": ["This product"]})).mark_rule(
            color=BRAND_DARK, strokeDash=[6, 4], size=3).encode(x="x:Q", tooltip=[alt.Tooltip("x:Q", title="Forecast", format=",.0f")])
        st.altair_chart(histogram + marker, use_container_width=True)
        st.caption(f"Bars: actual revenue of {store_id}'s products in the historical data, coloured by its bottom fifth, "
                   "middle and top fifth. Dashed line: the forecast for this product. Top-fifth products are candidates "
                   "for deeper stock; bottom-fifth products for leaner stock or review.")

# ---------------------------------------- Plan a product list ----------------------------------
with batch_tab:
    st.subheader("Plan a product list")
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
                        tiers.append(tier_of(o, f)[0]); units.append(str(round(u))); ranges.append(f"{lo:.0f}-{hi:.0f}")
                    else:
                        tiers.append("n/a"); units.append(str(round(f / m))); ranges.append("n/a")
                batch["Tier_at_Outlet"] = tiers
                batch["Units_Estimate"] = units
                batch["Stock_Range_Units"] = ranges

                st.success(f"Forecast {len(batch)} products - total forecast revenue {batch['Forecast_Revenue'].sum():,.2f} "
                           "(over the same period as the historical data)")
                first = [c for c in ["Outlet", "Product_Id", "Forecast_Revenue", "Tier_at_Outlet", "Units_Estimate",
                                     "Stock_Range_Units", "Data_Check"] if c in batch.columns]
                batch = batch[first + [c for c in batch.columns if c not in first]]
                st.dataframe(batch, use_container_width=True,
                             column_config={"Forecast_Revenue": st.column_config.NumberColumn("Forecast revenue", format="%.2f"),
                                            "Tier_at_Outlet": "Tier at outlet", "Units_Estimate": "≈ Units",
                                            "Stock_Range_Units": "Stock range (units)", "Data_Check": "Check"})

                summary = batch.groupby("Outlet").agg(
                    Products=("Forecast_Revenue", "size"),
                    Total_Forecast_Revenue=("Forecast_Revenue", "sum"),
                    Top_20=("Tier_at_Outlet", lambda t: int((t == "Top 20%").sum())),
                    Bottom_20=("Tier_at_Outlet", lambda t: int((t == "Bottom 20%").sum())),
                ).round(2)
                st.write("**Forecast revenue by outlet** - the basis for each outlet's procurement budget")
                summary_chart = alt.Chart(summary.reset_index()).mark_bar(color=BRAND).encode(
                    x=alt.X("Total_Forecast_Revenue:Q", title="Total forecast revenue"),
                    y=alt.Y("Outlet:N", sort="-x", title=None),
                    tooltip=["Outlet", "Products", alt.Tooltip("Total_Forecast_Revenue:Q", format=",.2f")])
                st.altair_chart(summary_chart, use_container_width=True)
                st.dataframe(summary, use_container_width=True,
                             column_config={"Total_Forecast_Revenue": st.column_config.NumberColumn("Total forecast revenue", format="%.2f"),
                                            "Top_20": "Top-20% products", "Bottom_20": "Bottom-20% products"})
                if (batch["Data_Check"] != "OK").any():
                    st.error("Some rows have a weight, MRP or display share outside the range the model was trained on "
                             f"(weight {TRAINED_RANGES['Product_Weight'][0]:g}-{TRAINED_RANGES['Product_Weight'][1]:g}, "
                             f"MRP {TRAINED_RANGES['Product_MRP'][0]:g}-{TRAINED_RANGES['Product_MRP'][1]:g}, display share "
                             f"{100 * TRAINED_RANGES['Product_Allocated_Area'][0]:g}-{100 * TRAINED_RANGES['Product_Allocated_Area'][1]:g}%); "
                             "their forecasts are not reliable.")
                if (batch["Outlet"] == "not a SuperKart outlet").any():
                    st.warning("Some rows describe a store profile that is not one of SuperKart's four outlets; "
                               "their forecasts are extrapolations, so no tier or stock range is given.")

                st.download_button("Download results (CSV)", batch.to_csv(index=False), "superkart_forecasts.csv", "text/csv")
            except requests.exceptions.RequestException as error:
                st.error(f"The forecasting service could not be reached: {error}. If it has been idle it may be waking up - please wait a minute and try again.")

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
