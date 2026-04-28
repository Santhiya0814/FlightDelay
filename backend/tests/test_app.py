import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("SECRET_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "")

@pytest.fixture(scope="session")
def flask_app():
    from app import app
    app.config["TESTING"] = True
    return app

@pytest.fixture(scope="session")
def client(flask_app):
    return flask_app.test_client()

@pytest.fixture(scope="session")
def model_bundle():
    import joblib
    return joblib.load(os.path.join(os.path.dirname(__file__), "..", "model", "all_models.pkl"))

@pytest.fixture(scope="session")
def accuracies():
    import json
    with open(os.path.join(os.path.dirname(__file__), "..", "model", "accuracies.json")) as f:
        return json.load(f)

@pytest.fixture(scope="session")
def dataset():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "dataset", "flight_dataset.csv"))


class TestEnvironment:
    def test_required_files_exist(self):
        base = os.path.join(os.path.dirname(__file__), "..")
        for f in ["app.py", "model/all_models.pkl", "model/accuracies.json", 
                  "dataset/flight_dataset.csv", "database/database.py", "requirements.txt"]:
            assert os.path.exists(os.path.join(base, f)), f"{f} missing"

    def test_required_packages_importable(self):
        import flask, flask_sqlalchemy, pandas, sklearn, joblib, numpy


class TestDataset:
    COLS = ["Airline", "Source", "Destination", "Distance", "Departure_Time", "Weather_Condition", "Delay_Status"]

    def test_dataset_structure(self, dataset):
        assert len(dataset) >= 1000
        for col in self.COLS:
            assert col in dataset.columns
        assert dataset[self.COLS].isnull().sum().sum() == 0

    def test_dataset_values(self, dataset):
        assert set(dataset["Delay_Status"].unique()) == {"Delayed", "On Time"}
        assert set(dataset["Airline"].unique()) == {"IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa Air", "Go First"}
        assert set(dataset["Weather_Condition"].unique()) == {"Clear", "Rain", "Fog", "Storm", "Cloudy"}
        assert 0 < dataset["Distance"].min() and dataset["Distance"].max() < 5000
        assert dataset["Delay_Status"].value_counts(normalize=True).min() >= 0.20


class TestModels:
    MODELS = ["KNN", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]

    def test_bundle_structure(self, model_bundle):
        assert "models" in model_bundle
        assert len(model_bundle["models"]) == 5
        for m in self.MODELS:
            assert m in model_bundle["models"]

    def test_model_predictions(self, model_bundle):
        test_df = pd.DataFrame([{"Airline": "IndiGo", "Source": "Chennai", "Destination": "Delhi",
                                 "Distance": 1760.0, "Departure_Time": "Evening", "Weather_Condition": "Storm"}])
        for name, pipeline in model_bundle["models"].items():
            pred = pipeline.predict(test_df)[0]
            assert pred in [0, 1]
            assert pipeline.predict(test_df)[0] == pred  # consistency
            clf = pipeline.named_steps["classifier"]
            if hasattr(clf, "predict_proba"):
                proba = pipeline.predict_proba(test_df)[0]
                assert len(proba) == 2 and abs(sum(proba) - 1.0) < 1e-6


class TestAccuracies:
    KEYS = ["accuracies", "precision", "recall", "f1_score", "cv_scores", "best_model"]
    MODELS = ["KNN", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]

    def test_accuracies_structure(self, accuracies):
        for key in self.KEYS:
            assert key in accuracies
        for m in self.MODELS:
            assert m in accuracies["accuracies"]
            assert 0.0 <= accuracies["accuracies"][m] <= 1.0

    def test_best_model(self, accuracies):
        best = accuracies["best_model"]
        assert best in self.MODELS
        assert accuracies["accuracies"][best] == max(accuracies["accuracies"].values())


class TestRoutes:
    def test_home_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.data.decode()
        for field in ["predictForm", "airline", "source", "destination", "distance", "departure_time", "weather_condition"]:
            assert field in body
        for item in ["IndiGo", "Air India", "SpiceJet", "Chennai", "Delhi", "Mumbai"]:
            assert item in body

    def test_dashboard(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200
        body = r.data.decode()
        for m in ["KNN", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]:
            assert m in body

    def test_predict_success(self, client):
        r = client.post("/predict", data={"airline": "IndiGo", "source": "Delhi", "destination": "Mumbai",
                                           "distance": "1415", "departure_time": "Afternoon", "weather_condition": "Cloudy"})
        assert r.status_code == 200
        body = r.data.decode()
        for m in ["KNN", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]:
            assert m in body
        assert "Best Model" in body or "best_model" in body.lower()
        assert "Vote" in body or "vote" in body

    def test_predict_validation(self, client):
        r = client.post("/predict", data={"airline": "IndiGo", "source": "Delhi", "destination": "Delhi",
                                           "distance": "0", "departure_time": "Morning", "weather_condition": "Clear"})
        assert "same" in r.data.decode().lower() or "error" in r.data.decode().lower()


class TestPredictAllModels:
    def test_predict_all_models(self, flask_app):
        from app import predict_all_models
        with flask_app.app_context():
            data = predict_all_models("IndiGo", "Chennai", "Delhi", 1760, "Evening", "Storm")
        assert len(data["results"]) == 5
        assert data["delayed_votes"] + data["ontime_votes"] == 5
        assert data["best_model"] in data["results"]
        assert data["final_prediction"] in ["Delayed", "On Time"]
        assert data["best_accuracy"] == data["results"][data["best_model"]]["accuracy"]
        for name, res in data["results"].items():
            assert "prediction" in res and "accuracy" in res
            assert res["prediction"] in ["Delayed", "On Time"]
