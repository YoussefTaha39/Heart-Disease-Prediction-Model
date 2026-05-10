from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("outputs/heart_disease_model.pkl")

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # GET FORM DATA
        # ----------------------------------------------------

        data = request.form.to_dict()

        # ----------------------------------------------------
        # CONVERT VALUES TO FLOAT
        # ----------------------------------------------------

        for key in data:
            data[key] = float(data[key])

        # ----------------------------------------------------
        # CONVERT TO DATAFRAME
        # ----------------------------------------------------

        df = pd.DataFrame([data])

        # Ensure numeric types
        df = df.astype(float)

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(df)[0]

        probability = model.predict_proba(df)[0][1]

        # ----------------------------------------------------
        # RETURN JSON RESPONSE
        # ----------------------------------------------------

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400

# ============================================================
# RECOMMENDATIONS PAGE
# ============================================================

@app.route("/recommendations")
def recommendations():

    probability = request.args.get(
        "probability",
        0,
        type=float
    )

    prediction = request.args.get(
        "prediction",
        0,
        type=int
    )

    name = request.args.get(
        "name",
        "Patient",
        type=str
    )

    # --------------------------------------------------------
    # RISK LEVEL CLASSIFICATION
    # --------------------------------------------------------

    if probability < 0.30:
        level = "low"

    elif probability < 0.60:
        level = "mild"

    else:
        level = "high"

    # --------------------------------------------------------
    # RENDER TEMPLATE
    # --------------------------------------------------------

    return render_template(
        "recommendations.html",
        probability=probability,
        prediction=prediction,
        level=level,
        name=name
    )

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )