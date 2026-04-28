import os
import re
from datetime import datetime
from urllib.parse import urlparse, urlunparse, quote
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

db = SQLAlchemy()

PROJECT_ID = "warkcxvejmsbepabxlnp"


def fix_db_url(url: str) -> str:
    """
    Sanitise a Supabase DATABASE_URL for SQLAlchemy 2.0 + psycopg2.

    Fixes applied:
      1. postgres:// -> postgresql://   (SQLAlchemy 2.0 compatibility)
      2. Encode password special characters (like @) using quote()
      3. Username suffix postgres.<PROJECT_ID> (Required for Supabase pooler)
      4. Strip all query parameters (pgbouncer=true causes DSN errors)
    """
    if not url:
        return url

    # 1. Protocol fix
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # 2. Parse URL
    # urlparse handles multiple @ by taking the last one as the host separator
    parsed = urlparse(url)

    # 3. Extract components
    username = parsed.username or "postgres"
    password = parsed.password or ""
    hostname = parsed.hostname or ""
    port     = parsed.port or 6543 # Default to Supabase pooler port
    path     = parsed.path

    # 4. Fix username suffix if missing
    if "." not in username:
        username = f"{username}.{PROJECT_ID}"

    # 5. URL-encode password — decode first to avoid double-encoding
    from urllib.parse import unquote
    quoted_password = quote(unquote(password), safe='')

    # 6. Rebuild netloc (username:password@host:port)
    netloc = f"{username}:{quoted_password}@{hostname}:{port}"

    # 7. Rebuild final URL without query strings
    clean_url = urlunparse((
        "postgresql",
        netloc,
        path,
        "", # params
        "", # query (strips ?pgbouncer=true)
        ""  # fragment
    ))

    return clean_url


def init_db(app):
    raw_url    = os.getenv("DATABASE_URL", "")
    sqlite_path = os.path.join(os.path.dirname(__file__), "..", "flightdelay.db")
    sqlite_uri  = f"sqlite:///{os.path.abspath(sqlite_path)}"

    db_uri = sqlite_uri
    use_supabase = False

    if raw_url and "[YOUR-PASSWORD]" not in raw_url:
        candidate_uri = fix_db_url(raw_url)
        print(f"[INFO] Testing Supabase connection...")
        
        # Test connection with a temporary engine before committing to it
        from sqlalchemy import create_engine, text
        try:
            import threading
            result = {"success": False, "error": None}

            def _test():
                try:
                    test_engine = create_engine(candidate_uri, connect_args={"connect_timeout": 5})
                    with test_engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    test_engine.dispose()
                    result["success"] = True
                except Exception as e:
                    result["error"] = e

            t = threading.Thread(target=_test, daemon=True)
            t.start()
            t.join(timeout=8)

            if result["success"]:
                db_uri = candidate_uri
                use_supabase = True
                print("[INFO] Supabase connection successful.")
            else:
                err = result["error"] or "Connection timed out"
                print(f"[WARN] Supabase connection failed: {err}")
                print("[INFO] Falling back to SQLite for this session.")
        except Exception as e:
            print(f"[WARN] Supabase connection failed: {e}")
            print("[INFO] Falling back to SQLite for this session.")

    app.config["SQLALCHEMY_DATABASE_URI"]        = db_uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"]      = {
        "pool_pre_ping": True,
        "pool_recycle":  300,
    }

    db.init_app(app)

    with app.app_context():
        db.create_all()
        if use_supabase:
            print("[INFO] PostgreSQL database tables ready.")
        else:
            print("[INFO] SQLite database tables ready.")


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id                = db.Column(db.Integer,     primary_key=True)
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
