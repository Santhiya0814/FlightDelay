import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request
from dotenv import load_dotenv
from database.database import db, init_db, PredictionHistory

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "static"),
)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
init_db(app)

# ── ML bundle ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
PKL_PATH      = os.path.join(BASE_DIR, "model", "all_models.pkl")
JSON_PATH     = os.path.join(BASE_DIR, "model", "accuracies.json")

_bundle = None

def load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(PKL_PATH):
            raise FileNotFoundError(
                "Model not found. Run:  python model/train_model.py"
            )
        _bundle = joblib.load(PKL_PATH)
    return _bundle

def get_accuracies():
    with open(JSON_PATH, "r") as f:
        return json.load(f)

def predict_all_models(airline, source, destination, distance,
                       departure_time, weather_condition):
    bundle     = load_bundle()
    models     = bundle["models"]
    accuracies = get_accuracies()

    input_df = pd.DataFrame([{
        "Airline":           airline,
        "Source":            source,
        "Destination":       destination,
        "Distance":          float(distance),
        "Departure_Time":    departure_time,
        "Weather_Condition": weather_condition,
    }])

    results = {}
    for name, pipeline in models.items():
        pred = pipeline.predict(input_df)[0]
        label = "Delayed" if pred == 1 else "On Time"
        confidence = None
        clf = pipeline.named_steps["classifier"]
        if hasattr(clf, "predict_proba"):
            confidence = round(float(max(pipeline.predict_proba(input_df)[0])) * 100, 2)
        results[name] = {
            "prediction": label,
            "confidence": confidence,
            "accuracy":   round(accuracies["accuracy"][name] * 100, 2)
                          if name in accuracies.get("accuracy", {})
                          else round(accuracies["accuracies"][name] * 100, 2),
        }

    # Majority vote
    votes       = [v["prediction"] for v in results.values()]
    delayed_cnt = votes.count("Delayed")
    ontime_cnt  = votes.count("On Time")
    majority    = "Delayed" if delayed_cnt >= ontime_cnt else "On Time"

    # Best model by accuracy
    best_model = max(results, key=lambda m: results[m]["accuracy"])
    best_pred  = results[best_model]["prediction"]

    # Final prediction = best model prediction (most reliable)
    final_prediction = best_pred

    return {
        "results":          results,
        "majority":         majority,
        "delayed_votes":    delayed_cnt,
        "ontime_votes":     ontime_cnt,
        "best_model":       best_model,
        "best_accuracy":    results[best_model]["accuracy"],
        "final_prediction": final_prediction,
        "method":           "Best Model",
    }

# ── Dropdown constants ────────────────────────────────────────────────────────
AIRLINES           = ["IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa Air", "Go First"]
CITIES             = ["Chennai", "Delhi", "Mumbai", "Bangalore",
                      "Hyderabad", "Kolkata", "Pune", "Ahmedabad"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Fog", "Storm", "Cloudy"]
DEPARTURE_TIMES    = ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"]
MODEL_NAMES        = ["KNN", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        airlines=AIRLINES, cities=CITIES,
        weather_conditions=WEATHER_CONDITIONS,
        departure_times=DEPARTURE_TIMES,
        model_names=MODEL_NAMES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        airline           = request.form["airline"]
        source            = request.form["source"]
        destination       = request.form["destination"]
        distance          = float(request.form["distance"])
        departure_time    = request.form["departure_time"]
        weather_condition = request.form["weather_condition"]

        if source == destination:
            return render_template(
                "index.html",
                error="Source and destination cannot be the same.",
                airlines=AIRLINES, cities=CITIES,
                weather_conditions=WEATHER_CONDITIONS,
                departure_times=DEPARTURE_TIMES,
            )

        data = predict_all_models(
            airline, source, destination, distance,
            departure_time, weather_condition
        )

        # Save final prediction (best model) to DB
        best = data["best_model"]
        db.session.add(PredictionHistory(
            airline=airline, source=source, destination=destination,
            distance=distance, departure_time=departure_time,
            weather_condition=weather_condition,
            model_used=best,
            prediction=data["final_prediction"],
            confidence=data["results"][best]["confidence"],
        ))
        db.session.commit()

        return render_template(
            "result.html",
            data=data,
            airline=airline, source=source, destination=destination,
            distance=distance, departure_time=departure_time,
            weather_condition=weather_condition,
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction failed: {str(e)}",
            airlines=AIRLINES, cities=CITIES,
            weather_conditions=WEATHER_CONDITIONS,
            departure_times=DEPARTURE_TIMES,
        )


@app.route("/dashboard")
def dashboard():
    accuracies = get_accuracies()
    recent     = PredictionHistory.query.order_by(
                     PredictionHistory.created_at.desc()
                 ).limit(10).all()
    total      = PredictionHistory.query.count()
    delayed    = PredictionHistory.query.filter_by(prediction="Delayed").count()
    on_time    = PredictionHistory.query.filter_by(prediction="On Time").count()

    return render_template(
        "dashboard.html",
        accuracies=accuracies,
        recent_predictions=[r.to_dict() for r in recent],
        total=total, delayed=delayed, on_time=on_time,
    )

if __name__ == "__main__":
    # Render kudukura port-ai dynamic-ah edukanum
    port = int(os.environ.get("PORT", 8080))
    # Production-la debug=False nu irukanum
    app.run(host='0.0.0.0', port=port, debug=False)