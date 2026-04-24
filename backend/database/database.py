import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

db = SQLAlchemy()


def init_db(app):
    db_url = os.getenv("DATABASE_URL", "")
    sqlite_path = os.path.join(os.path.dirname(__file__), "..", "flightdelay.db")
    sqlite_uri  = f"sqlite:///{os.path.abspath(sqlite_path)}"

    if db_url and "[YOUR-PASSWORD]" not in db_url:
        app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = sqlite_uri
        print("[INFO] Using SQLite fallback database.")

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        try:
            db.create_all()
            print("[INFO] Database tables ready.")
        except Exception as e:
            print(f"[WARN] Supabase connection failed: {e}")
            print("[INFO] Falling back to SQLite.")
            app.config["SQLALCHEMY_DATABASE_URI"] = sqlite_uri
            # Re-init with SQLite
            db.engine.dispose()
            from sqlalchemy import create_engine
            app.config["SQLALCHEMY_DATABASE_URI"] = sqlite_uri
            with app.app_context():
                db.create_all()


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id                = db.Column(db.Integer, primary_key=True)
    airline           = db.Column(db.String(100), nullable=False)
    source            = db.Column(db.String(100), nullable=False)
    destination       = db.Column(db.String(100), nullable=False)
    distance          = db.Column(db.Float,        nullable=False)
    departure_time    = db.Column(db.String(30),   nullable=False)
    weather_condition = db.Column(db.String(50),   nullable=False)
    model_used        = db.Column(db.String(50),   nullable=False)
    prediction        = db.Column(db.String(20),   nullable=False)
    confidence        = db.Column(db.Float,        nullable=True)
    created_at        = db.Column(db.DateTime,     default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":                self.id,
            "airline":           self.airline,
            "source":            self.source,
            "destination":       self.destination,
            "distance":          self.distance,
            "departure_time":    self.departure_time,
            "weather_condition": self.weather_condition,
            "model_used":        self.model_used,
            "prediction":        self.prediction,
            "confidence":        round(self.confidence, 2) if self.confidence else None,
            "created_at":        self.created_at.strftime("%Y-%m-%d %H:%M"),
        }
