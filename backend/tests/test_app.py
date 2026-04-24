"""
Flight Delay Prediction — Test Suite
Run: cd backend && python -m pytest tests/ -v
"""

import os
import sys
import json
import pytest
import pandas as pd

# Ensure backend/ is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Use SQLite for all tests — never touch Supabase
os.environ.setdefault("SECRET_KEY",   "test-secret-key")
os.environ.setdefault("DATABASE_URL", "")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def flask_app():
    """Create Flask app configured for testing."""
    from app import app
    app.config["TESTING"]   = True
    app.config["WTF_CSRF_ENABLED"] = False
    return app


@pytest.fixture(scope="session")
def client(flask_app):
    """Flask test client."""
    return flask_app.test_client()


@pytest.fixture(scope="session")
def model_bundle():
    """Load the ML model bundle once for all tests."""
    import joblib
    bundle_path = os.path.join(os.path.dirname(__file__), "..", "model", "all_models.pkl")
    return joblib.load(bundle_path)


@pytest.fixture(scope="session")
def accuracies():
    """Load accuracies.json once for all tests."""
    json_path = os.path.join(os.path.dirname(__file__), "..", "model", "accuracies.json")
    with open(json_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def dataset():
    """Load the flight dataset once for all tests."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "flight_dataset.csv")
    return pd.read_csv(csv_path)


# ── 1. Environment Tests ──────────────────────────────────────────────────────

class TestEnvironment:

    def test_required_files_exist(self):
        base = os.path.join(os.path.dirname(__file__), "..")
        assert os.path.exists(os.path.join(base, "app.py")),                  "app.py missing"
        assert os.path.exists(os.path.join(base, "model", "all_models.pkl")), "all_models.pkl missing"
        assert os.path.exists(os.path.join(base, "model", "accuracies.json")),"accuracies.json missing"
        assert os.path.exists(os.path.join(base, "dataset", "flight_dataset.csv")), "flight_dataset.csv missing"
        assert os.path.exists(os.path.join(base, "database", "database.py")), "database.py missing"

    def test_requirements_file_exists(self):
        req = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        assert os.path.exists(req), "requirements.txt missing"

    def test_required_packages_importable(self):
        import flask
        import flask_sqlalchemy
        import pandas
        import sklearn
        import joblib
        import numpy
        assert True


# ── 2. Dataset Tests ──────────────────────────────────────────────────────────

class TestDataset:

    REQUIRED_COLS = [
        "Airline", "Source", "Destination", "Distance",
        "Departure_Time", "Weather_Condition", "Delay_Status"
    ]

    def test_dataset_loads(self, dataset):
        assert dataset is not None
        assert len(dataset) > 0

    def test_dataset_has_required_columns(self, dataset):
        for col in self.REQUIRED_COLS:
            assert col in dataset.columns, f"Missing column: {col}"

    def test_dataset_minimum_size(self, dataset):
        assert len(dataset) >= 1000, f"Dataset too small: {len(dataset)} rows"

    def test_dataset_no_nulls(self, dataset):
        null_count = dataset[self.REQUIRED_COLS].isnull().sum().sum()
        assert null_count == 0, f"Dataset has {null_count} null values"

    def test_dataset_valid_labels(self, dataset):
        labels = set(dataset["Delay_Status"].unique())
        assert labels == {"Delayed", "On Time"}, f"Unexpected labels: {labels}"

    def test_dataset_valid_airlines(self, dataset):
        expected = {"IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa Air", "Go First"}
        actual   = set(dataset["Airline"].unique())
        assert actual == expected, f"Unexpected airlines: {actual}"

    def test_dataset_valid_weather(self, dataset):
        expected = {"Clear", "Rain", "Fog", "Storm", "Cloudy"}
        actual   = set(dataset["Weather_Condition"].unique())
        assert actual == expected, f"Unexpected weather: {actual}"

    def test_dataset_distance_range(self, dataset):
        assert dataset["Distance"].min() > 0,    "Distance has non-positive values"
        assert dataset["Distance"].max() < 5000, "Distance has unrealistic values"

    def test_dataset_class_balance(self, dataset):
        vc = dataset["Delay_Status"].value_counts(normalize=True)
        # Neither class should be less than 20% (not severely imbalanced)
        assert vc.min() >= 0.20, f"Severe class imbalance: {vc.to_dict()}"


# ── 3. Model Tests ────────────────────────────────────────────────────────────

class TestModels:

    EXPECTED_MODELS = ["KNN", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]

    def test_bundle_loads(self, model_bundle):
        assert model_bundle is not None

    def test_bundle_has_models_key(self, model_bundle):
        assert "models" in model_bundle

    def test_all_five_models_present(self, model_bundle):
        loaded = list(model_bundle["models"].keys())
        for m in self.EXPECTED_MODELS:
            assert m in loaded, f"Model missing: {m}"

    def test_no_extra_models(self, model_bundle):
        loaded = list(model_bundle["models"].keys())
        assert len(loaded) == 5, f"Expected 5 models, found {len(loaded)}: {loaded}"

    def test_each_model_can_predict(self, model_bundle):
        test_df = pd.DataFrame([{
            "Airline": "IndiGo", "Source": "Chennai", "Destination": "Delhi",
            "Distance": 1760.0, "Departure_Time": "Evening",
            "Weather_Condition": "Storm"
        }])
        for name, pipeline in model_bundle["models"].items():
            pred = pipeline.predict(test_df)[0]
            assert pred in [0, 1], f"{name} returned unexpected prediction: {pred}"

    def test_each_model_predict_proba(self, model_bundle):
        test_df = pd.DataFrame([{
            "Airline": "IndiGo", "Source": "Chennai", "Destination": "Delhi",
            "Distance": 1760.0, "Departure_Time": "Morning",
            "Weather_Condition": "Clear"
        }])
        for name, pipeline in model_bundle["models"].items():
            clf = pipeline.named_steps["classifier"]
            if hasattr(clf, "predict_proba"):
                proba = pipeline.predict_proba(test_df)[0]
                assert len(proba) == 2,          f"{name}: proba length != 2"
                assert abs(sum(proba) - 1.0) < 1e-6, f"{name}: probabilities don't sum to 1"

    def test_prediction_consistency(self, model_bundle):
        """Same input should always give same output."""
        test_df = pd.DataFrame([{
            "Airline": "Vistara", "Source": "Mumbai", "Destination": "Bangalore",
            "Distance": 984.0, "Departure_Time": "Morning",
            "Weather_Condition": "Clear"
        }])
        for name, pipeline in model_bundle["models"].items():
            pred1 = pipeline.predict(test_df)[0]
            pred2 = pipeline.predict(test_df)[0]
            assert pred1 == pred2, f"{name}: inconsistent predictions"


# ── 4. Accuracies JSON Tests ──────────────────────────────────────────────────

class TestAccuracies:

    REQUIRED_KEYS  = ["accuracies", "precision", "recall", "f1_score", "cv_scores", "best_model"]
    EXPECTED_MODELS = ["KNN", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]

    def test_json_has_required_keys(self, accuracies):
        for key in self.REQUIRED_KEYS:
            assert key in accuracies, f"Missing key: {key}"

    def test_all_models_in_accuracies(self, accuracies):
        for m in self.EXPECTED_MODELS:
            assert m in accuracies["accuracies"], f"Missing model in accuracies: {m}"

    def test_accuracy_values_in_range(self, accuracies):
        for m, val in accuracies["accuracies"].items():
            assert 0.0 <= val <= 1.0, f"{m} accuracy out of range: {val}"

    def test_best_model_is_valid(self, accuracies):
        assert accuracies["best_model"] in self.EXPECTED_MODELS

    def test_best_model_has_highest_accuracy(self, accuracies):
        best     = accuracies["best_model"]
        best_acc = accuracies["accuracies"][best]
        for m, acc in accuracies["accuracies"].items():
            assert best_acc >= acc, f"{m} ({acc}) > best {best} ({best_acc})"


# ── 5. Flask Route Tests ──────────────────────────────────────────────────────

class TestRoutes:

    def test_home_page_loads(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_home_page_has_form(self, client):
        r    = client.get("/")
        body = r.data.decode()
        assert "predictForm"      in body
        assert "airline"          in body
        assert "source"           in body
        assert "destination"      in body
        assert "distance"         in body
        assert "departure_time"   in body
        assert "weather_condition" in body

    def test_home_page_has_all_airlines(self, client):
        r    = client.get("/")
        body = r.data.decode()
        for airline in ["IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa Air", "Go First"]:
            assert airline in body, f"Airline missing from form: {airline}"

    def test_home_page_has_all_cities(self, client):
        r    = client.get("/")
        body = r.data.decode()
        for city in ["Chennai", "Delhi", "Mumbai", "Bangalore",
                     "Hyderabad", "Kolkata", "Pune", "Ahmedabad"]:
            assert city in body, f"City missing from form: {city}"

    def test_dashboard_loads(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_dashboard_has_model_names(self, client):
        r    = client.get("/dashboard")
        body = r.data.decode()
        for m in ["KNN", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]:
            assert m in body, f"Model missing from dashboard: {m}"

    def test_predict_delayed_flight(self, client):
        r = client.post("/predict", data={
            "airline":           "SpiceJet",
            "source":            "Mumbai",
            "destination":       "Delhi",
            "distance":          "1415",
            "departure_time":    "Night",
            "weather_condition": "Storm",
        })
        assert r.status_code == 200
        body = r.data.decode()
        assert "DELAYED" in body or "ON TIME" in body

    def test_predict_clear_weather_flight(self, client):
        r = client.post("/predict", data={
            "airline":           "Vistara",
            "source":            "Chennai",
            "destination":       "Bangalore",
            "distance":          "346",
            "departure_time":    "Morning",
            "weather_condition": "Clear",
        })
        assert r.status_code == 200

    def test_predict_result_shows_all_5_models(self, client):
        r    = client.post("/predict", data={
            "airline":           "IndiGo",
            "source":            "Delhi",
            "destination":       "Mumbai",
            "distance":          "1415",
            "departure_time":    "Afternoon",
            "weather_condition": "Cloudy",
        })
        body = r.data.decode()
        for m in ["KNN", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]:
            assert m in body, f"Model result missing from result page: {m}"

    def test_predict_result_shows_best_model(self, client):
        r    = client.post("/predict", data={
            "airline":           "Air India",
            "source":            "Kolkata",
            "destination":       "Hyderabad",
            "distance":          "1490",
            "departure_time":    "Evening",
            "weather_condition": "Rain",
        })
        body = r.data.decode()
        assert "Best Model" in body or "best_model" in body.lower()

    def test_predict_result_shows_vote_count(self, client):
        r    = client.post("/predict", data={
            "airline":           "Akasa Air",
            "source":            "Pune",
            "destination":       "Ahmedabad",
            "distance":          "665",
            "departure_time":    "Morning",
            "weather_condition": "Fog",
        })
        body = r.data.decode()
        assert "Vote" in body or "vote" in body

    def test_predict_same_source_destination_rejected(self, client):
        r    = client.post("/predict", data={
            "airline":           "IndiGo",
            "source":            "Delhi",
            "destination":       "Delhi",
            "distance":          "0",
            "departure_time":    "Morning",
            "weather_condition": "Clear",
        })
        body = r.data.decode()
        assert "same" in body.lower() or "error" in body.lower()

    def test_404_returns_something(self, client):
        r = client.get("/nonexistent-page-xyz")
        assert r.status_code in [404, 302, 200]


# ── 6. predict_all_models Function Tests ─────────────────────────────────────

class TestPredictAllModels:

    def test_returns_all_five_results(self, flask_app):
        from app import predict_all_models
        with flask_app.app_context():
            data = predict_all_models(
                "IndiGo", "Chennai", "Delhi", 1760,
                "Evening", "Storm"
            )
        assert len(data["results"]) == 5

    def test_each_result_has_required_keys(self, flask_app):
        from app import predict_all_models
        with flask_app.app_context():
            data = predict_all_models(
                "Vistara", "Mumbai", "Bangalore", 984,
                "Morning", "Clear"
            )
        for name, res in data["results"].items():
            assert "prediction"  in res, f"{name} missing 'prediction'"
            assert "accuracy"    in res, f"{name} missing 'accuracy'"
            assert res["prediction"] in ["Delayed", "On Time"]

    def test_vote_counts_sum_to_five(self, flask_app):
        from app import predict_all_models
        with flask_app.app_context():
            data = predict_all_models(
                "Go First", "Delhi", "Kolkata", 1530,
                "Night", "Fog"
            )
        assert data["delayed_votes"] + data["ontime_votes"] == 5

    def test_best_model_is_in_results(self, flask_app):
        from app import predict_all_models
        with flask_app.app_context():
            data = predict_all_models(
                "SpiceJet", "Hyderabad", "Pune", 560,
                "Afternoon", "Cloudy"
            )
        assert data["best_model"] in data["results"]

    def test_final_prediction_is_valid(self, flask_app):
        from app import predict_all_models
        with flask_app.app_context():
            data = predict_all_models(
                "Air India", "Ahmedabad", "Chennai", 1955,
                "Late Night", "Storm"
            )
        assert data["final_prediction"] in ["Delayed", "On Time"]

    def test_best_accuracy_matches_best_model(self, flask_app):
        from app import predict_all_models
        with flask_app.app_context():
            data = predict_all_models(
                "Akasa Air", "Bangalore", "Kolkata", 1870,
                "Early Morning", "Rain"
            )
        best     = data["best_model"]
        expected = data["results"][best]["accuracy"]
        assert data["best_accuracy"] == expected
