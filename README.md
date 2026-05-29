# AblyWorks ATS — Recruiter Performance Gap Analysis

> An ML-powered Applicant Tracking System (ATS) analytics dashboard that identifies performance gaps, predicts rejection risk, and delivers ranked, actionable recommendations for every recruiter on the team.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [ML Models](#ml-models)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Database Configuration](#database-configuration)
- [CSV Fallback](#csv-fallback)
- [Data Schema](#data-schema)
- [Contributing](#contributing)

---

## Overview

AblyWorks ATS is a full-stack recruiter analytics platform built as part of a UPES internship project. It connects a Python/Flask ML backend with an interactive HTML dashboard to surface performance gaps across a recruiting team — highlighting bottlenecks, predicting rejection risk, and recommending concrete actions to improve conversion rates.

---

## Features

-  **Team Leaderboard** — Ranks all recruiters by a weighted 0–100 productivity score
-  **Rejection Risk Prediction** — Logistic Regression model flags HIGH / MEDIUM / LOW risk recruiters
-  **Bottleneck Detection** — Z-score anomaly detection identifies slow submission or scheduling stages
-  **Trend Forecasting** — Linear regression on weekly shortlist rates to detect improving/declining performance
-  **Peer Learning (Collaborative Filtering)** — Surfaces habits of top performers most similar to a target recruiter
-  **KPI Breakdown** — Per-recruiter scores across 6 KPIs with industry benchmark comparisons
-  **Funnel Drop-off Analysis** — Pinpoints the exact pipeline stage with the highest candidate loss
-  **Interactive Dashboard** — Browser-based HTML frontend served by Flask

---

## Architecture

```
┌─────────────────────────────┐
│   ablyworks_dashboard.html  │  ← Browser UI
└────────────┬────────────────┘
             │ REST API calls
┌────────────▼────────────────┐
│         app.py              │  ← Flask API Server (port 5000)
└────────────┬────────────────┘
             │
     ┌───────▼────────┐
     │ ml_recommender │  ← ML Engine (6 models)
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │ dbConnections  │  ← MySQL (falls back to CSV)
     └────────────────┘
```

---

## ML Models

| # | Model | Algorithm | Purpose |
|---|-------|-----------|---------|
| 1 | `RejectionRiskClassifier` | Logistic Regression | Predicts P(high rejection) for each recruiter |
| 2 | `RecruiterScorer` | Weighted KPI engine | Produces a 0–100 productivity score |
| 3 | `BottleneckDetector` | Z-score anomaly detection | Flags stages >1.5σ slower than team mean |
| 4 | `CollaborativeRecommender` | Item-based cosine similarity CF | Finds top-performer habits to emulate |
| 5 | `WeeklyTrendForecaster` | Linear Regression | Forecasts next-week shortlist conversion rate |
| 6 | `GapAnalysisEngine` | Orchestrator | Combines all models → ranked `Recommendation` list |

### KPI Weights (RecruiterScorer)

| KPI | Weight | Target |
|-----|--------|--------|
| Submission → Shortlist Rate | 25% | 70% |
| Shortlist → Interview Rate | 20% | 65% |
| Interview → Offer Rate | 25% | 40% |
| Time to Submit (days, lower=better) | 10% | 1.0d |
| Time to Schedule (days, lower=better) | 10% | 2.0d |
| JD Match Score | 10% | 80% |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the HTML dashboard |
| `GET` | `/api/data` | Returns all raw application records |
| `GET` | `/api/profiles` | Returns processed recruiter profiles |
| `GET` | `/api/leaderboard` | Returns the team leaderboard (scored & ranked) |
| `GET` | `/api/analyze/<name>` | Full ML gap analysis for a single recruiter |
| `GET` | `/api/insights` | ML gap analysis for all recruiters |

---

## Project Structure

```
.
├── app.py                          # Flask API server
├── ml_recommender.py               # ML models & gap analysis engine
├── dbConnections.py                # MySQL connection helper (gitignored)
├── ablyworks_dashboard.html        # Interactive browser dashboard
├── ats_recruiter_dataset_1000.csv  # Sample dataset (1000 applications)
├── requirements.txt                # Python dependencies
├── Icons/                          # Dashboard icon assets
└── README.md
```

> **Note:** `dbConnections.py` is excluded from version control (see `.gitignore`) because it contains database credentials. Each developer must create their own local copy.

---

## Prerequisites

- Python 3.8+
- MySQL Server (optional — falls back to CSV automatically)
- pip

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies:
- `flask` — Web framework & API server
- `flask-cors` — Cross-origin request support
- `mysql-connector-python` — MySQL database driver
- `pandas` — Data manipulation
- `numpy` — Numerical computing
- `scikit-learn` — ML models (Logistic Regression, Linear Regression, cosine similarity)

### 3. Configure the database connection

Create a `dbConnections.py` file in the project root (this file is gitignored):

```python
import mysql.connector
import pandas as pd

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="YOUR_PASSWORD",   # ← replace with your MySQL password
        database="ats_analytics"    # ← replace with your database name
    )

def fetch_data(query):
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df
```

> If no MySQL database is available, the app will automatically fall back to `ats_recruiter_dataset_1000.csv`.

---

## Running the App

```bash
python app.py
```

Then open your browser at:

```
http://localhost:5000
```

The terminal will print available API routes on startup.

---

## Database Configuration

The app expects a MySQL database named `ats_analytics` with a table called `applications`.

If you are using the CSV file instead, no additional setup is required — the fallback is automatic.

---

## CSV Fallback

If the database is unavailable or the `applications` table is empty, the system automatically reads from `ats_recruiter_dataset_1000.csv`. This file contains 1,000 synthetic application records and is suitable for development and demo purposes.

---

## Data Schema

The application expects the following columns (case-insensitive, snake_case or camelCase are both handled):

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | string | Unique candidate identifier |
| `candidate_name` | string | Candidate full name |
| `recruiter` | string | Recruiter name |
| `recruiter_experience_years` | int | Years of recruiter experience |
| `job_role` | string | Target job role |
| `client` | string | Client / hiring company |
| `submission_date` | date | Date profile was submitted |
| `submitted` | bool | Whether profile was submitted |
| `submission_delay_days` | float | Days taken to submit after sourcing |
| `skill_match_score` | float | JD–Resume skill match score (0–100) |
| `hm_review_status` | string | Hiring Manager review result (e.g. `Approved`) |
| `shortlisted` | bool | Whether candidate was shortlisted |
| `interview_scheduled` | bool | Whether an interview was scheduled |
| `interview_delay_days` | float | Days taken to schedule interview |
| `followup_delay_days` | float | Days taken for follow-up |
| `interview_result` | string | Interview outcome (e.g. `Cleared`) |
| `offer_status` | string | Offer extended (`Yes` / `No`) |
| `joining_status` | string | Candidate joining outcome |
| `rejection_reason` | string | Reason for rejection (if any) |
| `candidate_drop_reason` | string | Reason candidate dropped out (if any) |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

