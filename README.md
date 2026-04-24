# Flight Delay Prediction using Ensemble Machine Learning

A production-ready final year project that predicts flight delays using 5 machine learning models with a Flask web application, Supabase PostgreSQL database, and a complete CI/CD pipeline using GitHub Actions.

---

## Tech Stack

| Layer       | Technology                              |
|-------------|------------------------------------------|
| Backend     | Python 3.11, Flask 3.0                  |
| ML Models   | Scikit-learn (KNN, Naive Bayes, SVM, LR, RF) |
| Database    | Supabase (PostgreSQL) + SQLite fallback |
| Frontend    | HTML5, CSS3, JavaScript (Lavender Theme)|
| CI/CD       | GitHub Actions                          |
| Deployment  | Render                                  |

---

## Project Structure

```
FlightDelay/
├── .github/
│   └── workflows/
│       ├── ci.yml          ← Continuous Integration pipeline
│       └── deploy.yml      ← Continuous Deployment to Render
│
├── backend/
│   ├── app.py              ← Flask app + all routes + ML logic
│   ├── .env                ← Local secrets (never committed)
│   ├── requirements.txt    ← Python dependencies
│   ├── database/
│   │   └── database.py     ← SQLAlchemy + PredictionHistory model
│   ├── dataset/
│   │   └── flight_dataset.csv  ← 6000-record training dataset
│   ├── model/
│   │   ├── train_model.py  ← Train all 5 models
│   │   ├── all_models.pkl  ← Saved model bundle (single file)
│   │   └── accuracies.json ← Model performance metrics
│   └── tests/
│       └── test_app.py     ← Pytest test suite (40+ tests)
│
├── frontend/
│   ├── static/
│   │   ├── style.css       ← Lavender theme CSS
│   │   └── script.js       ← Animations & interactions
│   └── templates/
│       ├── base.html
│       ├── index.html      ← Prediction form
│       ├── result.html     ← All 5 model results
│       └── dashboard.html  ← Analytics dashboard
│
├── .gitignore
├── .env.example            ← Environment variable template
├── Procfile                ← Render/Railway start command
├── render.yaml             ← Render deployment config
└── README.md
```

---

## Machine Learning Models

All 5 models are trained on the same dataset and saved in a **single** `all_models.pkl` file:

| Model               | Description                                      |
|---------------------|--------------------------------------------------|
| KNN                 | K-Nearest Neighbors — distance-based classifier |
| Naive Bayes         | Probabilistic classifier using Bayes theorem    |
| SVM                 | Support Vector Machine with RBF kernel          |
| Logistic Regression | Linear model for binary classification          |
| Random Forest       | Ensemble of 200 decision trees                  |

### Prediction Flow
1. User submits flight details (airline, route, weather, time)
2. All 5 models run simultaneously on the input
3. Results compared — majority vote + best model identified
4. Final prediction shown with confidence scores

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/FlightDelay.git
cd FlightDelay
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Configure environment
```bash
cp .env.example backend/.env
# Edit backend/.env and add your Supabase DATABASE_URL and SECRET_KEY
```

### 4. Train the models
```bash
cd backend
python model/train_model.py
```

### 5. Run the application
```bash
cd backend
python app.py
```

Open: http://127.0.0.1:8080

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

Run with coverage report:
```bash
python -m pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## CI/CD Pipeline

### Overview

```
Push to GitHub
      │
      ▼
┌─────────────────────────────────────────┐
│         CI Pipeline (ci.yml)            │
│                                         │
│  Job 1: Lint & Syntax Check             │
│    └─ flake8 + py_compile               │
│                                         │
│  Job 2: Model & Data Validation         │
│    ├─ Validate accuracies.json          │
│    ├─ Load all_models.pkl               │
│    ├─ Validate flight_dataset.csv       │
│    └─ End-to-end prediction test        │
│                                         │
│  Job 3: Flask App Tests (pytest)        │
│    ├─ Route tests (/, /predict, /dash)  │
│    ├─ All 5 model prediction tests      │
│    ├─ Dataset integrity tests           │
│    └─ predict_all_models() unit tests   │
│                                         │
│  Job 4: CI Summary                      │
└─────────────────────────────────────────┘
      │
      │ (only if all CI jobs pass on main)
      ▼
┌─────────────────────────────────────────┐
│       CD Pipeline (deploy.yml)          │
│                                         │
│  1. Pre-deploy validation               │
│  2. Trigger Render deploy hook          │
│  3. Wait for build                      │
│  4. Live smoke test                     │
│  5. Deployment summary                  │
└─────────────────────────────────────────┘
```

### CI Pipeline — `ci.yml`

Triggers on every `push` to `main`/`develop` and every Pull Request.

**Job 1 — Lint & Syntax Check**
- Runs `flake8` to catch syntax errors and undefined names
- Runs `py_compile` on all Python files

**Job 2 — Model & Data Validation**
- Verifies `accuracies.json` has all required keys and 5 models
- Loads `all_models.pkl` and confirms all 5 models are present
- Validates `flight_dataset.csv` columns, size, labels, and null values
- Runs a live end-to-end prediction through all 5 models

**Job 3 — Flask App Tests**
- Runs 40+ pytest tests covering:
  - All Flask routes return correct status codes
  - Prediction form validation
  - All 5 model results appear on result page
  - Dataset integrity
  - Model accuracy values
  - `predict_all_models()` function correctness

### CD Pipeline — `deploy.yml`

Triggers automatically after CI passes on `main`, or manually via GitHub Actions tab.

1. Validates app one final time before deploying
2. Calls Render deploy hook URL (stored as GitHub Secret)
3. Waits for Render to build and start
4. Runs a live HTTP smoke test on the deployed URL

---

## Deployment to Render

### Step 1 — Create Render account
Go to [render.com](https://render.com) and sign up.

### Step 2 — Connect GitHub repository
- New → Web Service → Connect your GitHub repo
- Render auto-detects `render.yaml`

### Step 3 — Set environment variables in Render
In Render dashboard → Environment:
```
DATABASE_URL = postgresql://postgres:<password>@db.<id>.supabase.co:5432/postgres
SECRET_KEY   = <generate a strong random key>
FLASK_ENV    = production
```

### Step 4 — Add GitHub Secrets for CD pipeline
In GitHub → Settings → Secrets → Actions:

| Secret Name            | Value                                      |
|------------------------|--------------------------------------------|
| `RENDER_DEPLOY_HOOK_URL` | From Render → Settings → Deploy Hook URL |
| `RENDER_APP_URL`         | Your live app URL, e.g. `https://flight-delay-prediction.onrender.com` |

### Step 5 — Push to main
```bash
git add .
git commit -m "Deploy: Flight Delay Prediction"
git push origin main
```

GitHub Actions will automatically run CI → then deploy to Render.

---

## GitHub Actions Secrets Required

| Secret                   | Purpose                          | Where to get it              |
|--------------------------|----------------------------------|------------------------------|
| `RENDER_DEPLOY_HOOK_URL` | Triggers Render deployment       | Render → Settings → Deploy Hook |
| `RENDER_APP_URL`         | Live smoke test URL              | Your Render service URL      |

> **Note:** `DATABASE_URL` and `SECRET_KEY` are set directly in Render's environment variables, not in GitHub Secrets, because they are only needed at runtime on the server — not during CI testing.

---

## Dataset

- **File:** `backend/dataset/flight_dataset.csv`
- **Records:** 6,000 flights
- **Features:** Airline, Source, Destination, Distance, Departure Time, Weather Condition
- **Target:** Delay Status (Delayed / On Time)
- **Balance:** ~45% Delayed, ~55% On Time

---

## API Endpoints

| Method | Route        | Description                        |
|--------|--------------|------------------------------------|
| GET    | `/`          | Prediction form                    |
| POST   | `/predict`   | Run all 5 models, return results   |
| GET    | `/dashboard` | Analytics dashboard                |

---

## Viva Explanation Points

### What is CI/CD?
- **CI (Continuous Integration):** Every time code is pushed to GitHub, automated tests run to verify the code works correctly before it is merged.
- **CD (Continuous Deployment):** After CI passes, the application is automatically deployed to the production server without manual steps.

### Why GitHub Actions?
- Free for public repositories
- Runs directly from the repository — no external tools needed
- YAML-based configuration — easy to read and explain
- Industry standard used by companies worldwide

### How does the pipeline protect the project?
1. Syntax errors are caught before they reach production
2. Model files are validated — ensures all 5 models load correctly
3. All routes are tested — ensures the web app works end-to-end
4. Secrets are stored in GitHub Secrets — never hardcoded in code
5. Deployment only happens if ALL tests pass

### What happens if a test fails?
- The pipeline stops immediately
- GitHub shows a red ✗ on the commit
- Deployment is blocked
- Developer gets notified by email

---

## License

This project is developed as a Final Year Project submission.
