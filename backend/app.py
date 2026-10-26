
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
import os

DATA_FILE = "superkart_data.csv"          # SuperKart.csv without product ids, written by the notebook
history = pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else None

DIMENSIONS = {"outlet": "Store_Id", "product_type": "Product_Type", "sugar_content": "Product_Sugar_Content",
              "perishable": "Perishable", "product_family": "Product_Family", "store_type": "Store_Type",
              "city_tier": "Store_Location_City_Type", "store_size": "Store_Size"}
MEASURES = {"revenue": "Product_Store_Sales_Total", "mrp": "Product_MRP", "weight": "Product_Weight",
            "display_share_pct": "Display_Share_Pct"}
STATS = ["count", "sum", "mean", "median", "min", "max"]
NOTE = "Historical sales data (the data has no dates)."


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
                    "source": NOTE, "caution": "A pattern in the historical data, not cause and effect."})


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
