
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json()

    # Extract relevant features from the input data
    sample = {
    'Product_Weight': data['Product_Weight'],
    'Product_Sugar_Content': data['Product_Sugar_Content'],
    'Product_Allocated_Area': data['Product_Allocated_Area'],
    'Product_MRP': data['Product_MRP'],
    'Store_Size': data['Store_Size'],
    'Store_Location_City_Type': data['Store_Location_City_Type'],
    'Store_Type': data['Store_Type'],
    'Product_Id_char': data['Product_Id_char'],
    'Store_Age_Years': data['Store_Age_Years'],
    'Product_Type_Category': data['Product_Type_Category']
}

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Sales': prediction})

# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for the batch data
    predictions = model.predict(input_data).tolist()

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return output_dict


# ----------------------------------------------------------------------------------------------
# Historical data endpoints (used by the planning assistant for EDA questions)
# Only aggregated numbers leave the API - never individual rows. Every input is checked against fixed lists.
# ----------------------------------------------------------------------------------------------
import json
import os

DATA_FILE = "superkart_data.csv"          # SuperKart.csv without product ids, written by the notebook
NOTES_FILE = "data_notes.json"            # data-quality facts from the notebook (e.g. the number of suspect records)
notes = json.load(open(NOTES_FILE)) if os.path.exists(NOTES_FILE) else {}
history = pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else None
if history is not None:
    # Each product's revenue tier within its own outlet - the same top / bottom fifth cut-offs as the app's Forecast tab
    sales = history["Product_Store_Sales_Total"]
    cut = history.groupby("Store_Id")["Product_Store_Sales_Total"].quantile([0.2, 0.8]).unstack()
    history["Revenue_Tier"] = np.where(sales >= history["Store_Id"].map(cut[0.8]), "Top 20%",
                                       np.where(sales < history["Store_Id"].map(cut[0.2]), "Bottom 20%", "Middle 60%"))

DIMENSIONS = {"outlet": "Store_Id", "product_type": "Product_Type", "sugar_content": "Product_Sugar_Content",
              "perishable": "Perishable", "product_family": "Product_Family", "store_type": "Store_Type",
              "city_tier": "Store_Location_City_Type", "store_size": "Store_Size", "revenue_tier": "Revenue_Tier"}
MEASURES = {"revenue": "Product_Store_Sales_Total", "mrp": "Product_MRP", "weight": "Product_Weight",
            "display_share_pct": "Display_Share_Pct"}
STATS = ["count", "sum", "mean", "median", "min", "max"]
NOTE = "Historical sales data (the data has no dates)."


def quality_note():
    """The notebook's data-quality finding, attached to every relationship result."""
    if not notes.get("suspect_records"):
        return None
    return (f"{notes['suspect_records']} product records ({notes['suspect_share_pct']}% of the data) have revenue that does not "
            "match their price and weight (data-quality check in the notebook). They weaken relationships within an outlet, "
            "so the links shown here are likely understated.")


def bad_request(message):
    return jsonify({"error": message}), 400


def filtered(filters):
    """Rows matching the filters, e.g. {"outlet": "OUT002", "product_type": ["Dairy", "Breads"]}."""
    frame = history
    for key, value in (filters or {}).items():
        if key not in DIMENSIONS:
            raise ValueError(f"Unknown filter '{key}'. Use one of: {', '.join(DIMENSIONS)}.")
        values = [str(v) for v in (value if isinstance(value, list) else [value])]
        known = sorted(history[DIMENSIONS[key]].astype(str).unique())
        unknown = [v for v in values if v not in known]
        if unknown:
            raise ValueError(f"Unknown value(s) {unknown} for '{key}'. Known values: {known}.")
        frame = frame[frame[DIMENSIONS[key]].astype(str).isin(values)]
    return frame


@superkart_api.post('/v1/data/summary')
def data_summary():
    """Group the historical data by up to two dimensions and summarise one measure."""
    if history is None:
        return jsonify({"error": "Historical data is not available on this server."}), 503
    request_data = request.get_json(silent=True) or {}
    group_by = request_data.get("group_by") or []
    group_by = [group_by] if isinstance(group_by, str) else list(group_by)
    measure = request_data.get("measure", "revenue")
    stats = request_data.get("stats") or ["sum"]
    stats = [stats] if isinstance(stats, str) else list(stats)
    if len(group_by) > 2 or any(g not in DIMENSIONS for g in group_by):
        return bad_request(f"group_by: up to two of {', '.join(DIMENSIONS)}.")
    if measure not in MEASURES:
        return bad_request(f"measure: one of {', '.join(MEASURES)}.")
    if not stats or any(s not in STATS for s in stats):
        return bad_request(f"stats: one or more of {', '.join(STATS)}.")
    try:
        frame = filtered(request_data.get("filters"))
    except ValueError as error:
        return bad_request(str(error))
    if frame.empty:
        return jsonify({"rows": [], "products": 0, "note": "No products match these filters.", "source": NOTE})

    column = MEASURES[measure]
    value_stats = [s for s in stats if s != "count"]          # 'count' = the products column, always included
    if group_by:
        grouped = frame.groupby([DIMENSIONS[g] for g in group_by])[column]
        table = grouped.agg(value_stats).reset_index() if value_stats else grouped.size().reset_index().iloc[:, :-1]
        table.insert(len(group_by), "products", grouped.size().values)
        table = table.rename(columns={DIMENSIONS[g]: g for g in group_by})
        table["share_of_products_pct"] = 100 * table["products"] / len(frame)
    else:
        table = pd.DataFrame([{"products": len(frame), **{s: frame[column].agg(s) for s in value_stats}}])
    if "sum" in value_stats and frame[column].sum():
        table["share_of_total_pct"] = 100 * table["sum"] / frame[column].sum()
    table = table.round(2)
    rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in table.to_dict(orient="records")]
    return jsonify({"measure": measure, "stats": stats, "group_by": group_by, "filters": request_data.get("filters") or {},
                    "rows": rows, "products": int(len(frame)), "source": NOTE})


@superkart_api.post('/v1/data/relationship')
def data_relationship():
    """Correlation between two numeric columns in the historical data, optionally per group."""
    if history is None:
        return jsonify({"error": "Historical data is not available on this server."}), 503
    request_data = request.get_json(silent=True) or {}
    x, y = request_data.get("x", "mrp"), request_data.get("y", "revenue")
    group_by = request_data.get("group_by")
    if x not in MEASURES or y not in MEASURES or x == y:
        return bad_request(f"x and y: two different values of {', '.join(MEASURES)}.")
    if group_by is not None and group_by not in DIMENSIONS:
        return bad_request(f"group_by: one of {', '.join(DIMENSIONS)}.")
    try:
        frame = filtered(request_data.get("filters"))
    except ValueError as error:
        return bad_request(str(error))

    def describe(part):
        r = part[MEASURES[x]].corr(part[MEASURES[y]])
        if pd.isna(r):
            return {"products": int(len(part)), "correlation": None, "strength": "not enough data"}
        size = abs(r)
        strength = ("no clear link" if size < 0.1 else "weak" if size < 0.3 else "moderate" if size < 0.5
                    else "strong" if size < 0.7 else "very strong")
        direction = "" if size < 0.1 else (" positive" if r > 0 else " negative")
        return {"products": int(len(part)), "correlation": round(float(r), 3), "strength": strength + direction}

    if group_by:
        rows = [{group_by: str(key), **describe(part)} for key, part in frame.groupby(DIMENSIONS[group_by])]
    else:
        rows = [describe(frame)]
    return jsonify({"x": x, "y": y, "group_by": group_by, "filters": request_data.get("filters") or {}, "rows": rows,
                    "source": NOTE, "caution": "A pattern in the historical data, not cause and effect.",
                    "data_quality_note": quality_note()})


@superkart_api.post('/v1/data/correlation_matrix')
def data_correlation_matrix():
    """Correlations between all numeric columns (revenue, MRP, weight, display share), optionally filtered."""
    if history is None:
        return jsonify({"error": "Historical data is not available on this server."}), 503
    request_data = request.get_json(silent=True) or {}
    try:
        frame = filtered(request_data.get("filters"))
    except ValueError as error:
        return bad_request(str(error))
    if len(frame) < 10:
        return bad_request("Fewer than 10 products match these filters - too few for correlations.")
    matrix = frame[list(MEASURES.values())].corr()
    names = {column: name for name, column in MEASURES.items()}
    rows = [{"x": names[a], "y": names[b], "correlation": round(float(matrix.loc[a, b]), 3)}
            for a in matrix.columns for b in matrix.columns]
    return jsonify({"filters": request_data.get("filters") or {}, "products": int(len(frame)), "rows": rows,
                    "source": NOTE, "caution": "Patterns in the historical data, not cause and effect.",
                    "data_quality_note": quality_note()})


CATEGORICAL_FACTORS = ["outlet", "product_type", "sugar_content", "perishable", "product_family"]
STORE_COLUMNS = ["store_type", "city_tier", "store_size"]
STORE_NOTE = ("Store type, city tier and store size each move one-for-one with the outlet (every outlet has its own "
              "combination), so their effects cannot be told apart from the outlet's.")


def design(part, factors):
    """Model matrix: standardised numeric factors and one-hot categorical factors (first level dropped)."""
    blocks = {}
    for f in factors:
        if f in MEASURES:
            values = part[MEASURES[f]].astype(float)
            blocks[f] = ((values - values.mean()) / values.std(ddof=0)).to_numpy()[:, None]
        else:
            dummies = pd.get_dummies(part[DIMENSIONS[f]].astype(str), drop_first=True, dtype=float)
            blocks[f] = dummies.to_numpy()
    return blocks


def explained_share(y, blocks, names):
    columns = [blocks[n] for n in names if blocks[n].shape[1]]
    matrix = np.column_stack([np.ones(len(y))] + columns) if columns else np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    return 1 - ((y - matrix @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum(), coef


@superkart_api.post('/v1/data/drivers')
def data_drivers():
    """Multivariate view: how much of the differences in revenue between products a set of factors explains together
    (linear fit), how much each factor explains alone and how much it adds on top of the others, and - for the numeric
    factors - its standardised effect. Default factors: MRP, weight and display share; outlet, category, sugar content,
    perishable and product family can be added."""
    if history is None:
        return jsonify({"error": "Historical data is not available on this server."}), 503
    request_data = request.get_json(silent=True) or {}
    group_by = request_data.get("group_by")
    factors = request_data.get("factors") or ["mrp", "weight", "display_share_pct"]
    if group_by is not None and group_by not in DIMENSIONS:
        return bad_request(f"group_by: one of {', '.join(DIMENSIONS)}.")
    if any(f in STORE_COLUMNS for f in factors):
        return bad_request("Use 'outlet' instead of store type, city tier or store size. " + STORE_NOTE)
    allowed = ["mrp", "weight", "display_share_pct"] + CATEGORICAL_FACTORS
    if not factors or any(f not in allowed for f in factors):
        return bad_request(f"factors: one or more of {', '.join(allowed)}.")
    try:
        frame = filtered(request_data.get("filters"))
    except ValueError as error:
        return bad_request(str(error))

    def fit(part):
        used = [f for f in factors if (part[MEASURES[f]].std() > 0 if f in MEASURES else part[DIMENSIONS[f]].nunique() > 1)]
        if len(part) < 10 or not used:
            return {"products": int(len(part)), "note": "too few products or no variation - not analysed"}
        y = part[MEASURES["revenue"]].to_numpy(float)
        if y.std() == 0:
            return {"products": int(len(part)), "note": "no variation in revenue - not analysed"}
        blocks = design(part, used)
        together, coef = explained_share(y, blocks, used)
        row = {"products": int(len(part)), "explained_pct": round(100 * float(together), 1),
               "factors_used": used}
        position = 1
        for f in used:
            alone, _ = explained_share(y, blocks, [f])
            without, _ = explained_share(y, blocks, [g for g in used if g != f])
            row[f"alone_pct_{f}"] = round(100 * float(alone), 1)
            row[f"adds_pct_{f}"] = round(100 * float(together - without), 1)
            if f in MEASURES:
                row[f"effect_{f}"] = round(float(coef[position]) / y.std(), 2)     # standardised effect
            position += blocks[f].shape[1]
        if "mrp" in used and "weight" in used:
            pxw = part[MEASURES["mrp"]] * part[MEASURES["weight"]]
            row["mrp_x_weight_alone_explained_pct"] = round(100 * float(np.corrcoef(pxw, y)[0, 1] ** 2), 1)
        return row

    if group_by:
        rows = [{group_by: str(key), **fit(part)} for key, part in frame.groupby(DIMENSIONS[group_by])]
    else:
        rows = [fit(frame)]
    return jsonify({"group_by": group_by, "factors": factors, "filters": request_data.get("filters") or {}, "rows": rows,
                    "source": NOTE,
                    "how_to_read": "explained_pct: share of the differences in revenue between products that the factors explain "
                                   "together (linear fit). alone_pct_<factor>: what the factor explains on its own. "
                                   "adds_pct_<factor>: what it adds on top of all the others. effect_<factor>: standardised "
                                   "effect of a numeric factor with the others held equal (compare sizes, 0 = no effect).",
                    "store_note": STORE_NOTE, "caution": "Patterns in the historical data, not cause and effect.",
                    "data_quality_note": quality_note()})


@superkart_api.post('/v1/data/association_matrix')
def data_association_matrix():
    """Strength of association (0-1) between every pair of columns, numeric and categorical: |correlation| for two
    numeric columns, the correlation ratio for a numeric and a categorical column, Cramer's V for two categorical ones."""
    if history is None:
        return jsonify({"error": "Historical data is not available on this server."}), 503
    request_data = request.get_json(silent=True) or {}
    try:
        frame = filtered(request_data.get("filters"))
    except ValueError as error:
        return bad_request(str(error))
    if len(frame) < 10:
        return bad_request("Fewer than 10 products match these filters - too few for associations.")
    numeric = [m for m in MEASURES if frame[MEASURES[m]].std() > 0]
    categorical = [d for d in DIMENSIONS if d != "revenue_tier" and frame[DIMENSIONS[d]].nunique() > 1]

    def strength(a, b):
        if a in MEASURES and b in MEASURES:
            r = float(frame[MEASURES[a]].corr(frame[MEASURES[b]]))
            return abs(r), "correlation", round(r, 3)
        if a in MEASURES or b in MEASURES:
            num, cat = (a, b) if a in MEASURES else (b, a)
            values, groups = frame[MEASURES[num]], frame[DIMENSIONS[cat]]
            between = (values.groupby(groups).count() * (values.groupby(groups).mean() - values.mean()) ** 2).sum()
            return float(np.sqrt(between / ((values - values.mean()) ** 2).sum())), "correlation ratio", None
        table = pd.crosstab(frame[DIMENSIONS[a]], frame[DIMENSIONS[b]]).to_numpy(float)
        expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
        chi2 = ((table - expected) ** 2 / expected).sum()
        return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))), "Cramer's V", None

    columns = numeric + categorical
    rows = []
    for a in columns:
        for b in columns:
            value, measure, signed = (1.0, "same column", None) if a == b else strength(a, b)
            rows.append({"x": a, "y": b, "strength": round(value, 3), "measure": measure,
                         **({"signed_correlation": signed} if signed is not None else {})})
    return jsonify({"filters": request_data.get("filters") or {}, "products": int(len(frame)), "columns": columns,
                    "rows": rows, "source": NOTE,
                    "how_to_read": "strength from 0 (no association) to 1 (complete): |correlation| for two numeric columns, "
                                   "the correlation ratio for a numeric and a categorical column, Cramer's V for two "
                                   "categorical columns. Columns with a single value under the filters are left out.",
                    "store_note": STORE_NOTE, "caution": "Patterns in the historical data, not cause and effect.",
                    "data_quality_note": quality_note()})


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
