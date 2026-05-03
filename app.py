from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("heart_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    # Convert values
    for key in data:
        data[key] = float(data[key])

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

@app.route("/recommendations")
def recommendations():
    probability = request.args.get("probability", 0, type=float)
    prediction = request.args.get("prediction", 0, type=int)
    name = request.args.get("name", "Patient", type=str)

    # Classify risk level
    if probability < 0.3:
        level = "low"
    elif probability < 0.6:
        level = "mild"
    else:
        level = "high"

    return render_template("recommendations.html",
                           probability=probability,
                           prediction=prediction,
                           level=level,
                           name=name)

if __name__ == "__main__":
    app.run(debug=True)