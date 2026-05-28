"""
AblyWorks ATS — Flask API Server
Connects the HTML dashboard with the Python ML backend.

Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import sys
import json
import pandas as pd
from dataclasses import asdict

# ─── App Setup ────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ─── Import ML Backend ───────────────────────────────────
from ml_recommender import (
    load_team_from_db,
    GapAnalysisEngine,
    RecruiterScorer,
    RejectionRiskClassifier,
    WeeklyTrendForecaster,
    RecruiterProfile,
)
from dbConnections import fetch_data


# ─── Helper: Convert ML results to JSON-safe dicts ──────
def make_json_safe(obj):
    """Recursively convert numpy types and dataclasses to JSON-safe Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    else:
        return obj


# ─── Static File Routes ─────────────────────────────────
@app.route("/")
def serve_dashboard():
    return send_from_directory(".", "ablyworks_dashboard.html")


@app.route("/Icons/<path:filename>")
def serve_icons(filename):
    return send_from_directory("Icons", filename)


# ─── API: Raw Data ───────────────────────────────────────
@app.route("/api/data")
def api_data():
    """Return all application records from the database as JSON. Falls back to CSV if empty."""
    try:
        query = "SELECT * FROM applications"
        try:
            df = fetch_data(query)
        except Exception as db_err:
            print(f"Database fetch failed: {db_err}. Falling back to CSV.")
            df = pd.DataFrame()

        # Fallback to CSV if table is empty or DB connection failed
        if df.empty:
            import os
            csv_path = "ats_recruiter_dataset_1000.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                return jsonify({"error": "No data in DB and CSV not found"}), 404

        # Standardize column names (lowercase, strip, replace spaces/hyphens with underscores)
        standard_map = {
            "candidateid": "candidate_id",
            "candidatename": "candidate_name",
            "recruiter": "recruiter",
            "recruiterexperienceyears": "recruiter_experience_years",
            "jobrole": "job_role",
            "client": "client",
            "submissiondate": "submission_date",
            "submitted": "submitted",
            "submissiondelaydays": "submission_delay_days",
            "skillmatchscore": "skill_match_score",
            "hmreviewstatus": "hm_review_status",
            "shortlisted": "shortlisted",
            "interviewscheduled": "interview_scheduled",
            "interviewdelaydays": "interview_delay_days",
            "followupdelaydays": "followup_delay_days",
            "interviewresult": "interview_result",
            "offerstatus": "offer_status",
            "joiningstatus": "joining_status",
            "rejectionreason": "rejection_reason",
            "candidatedropreason": "candidate_drop_reason"
        }
        cleaned_cols = []
        for col in df.columns:
            c = str(col).lower().strip().replace(" ", "").replace("_", "").replace("-", "")
            if c in standard_map:
                cleaned_cols.append(standard_map[c])
            else:
                cleaned_cols.append(str(col).lower().strip().replace(" ", "_").replace("-", "_"))
        df.columns = cleaned_cols
        
        # Convert any datetime columns to string to prevent JSON serialization errors
        for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d')
            
        # Replace NaN/NaT/None with Python None for valid JSON nulls
        df = df.where(pd.notnull(df), None)
        
        records = df.to_dict(orient="records")
        # Convert any remaining nan values to None in the records
        safe_records = []
        for rec in records:
            safe_rec = {}
            for k, v in rec.items():
                if pd.isna(v):
                    safe_rec[k] = None
                else:
                    safe_rec[k] = v
            safe_records.append(safe_rec)
        return jsonify(make_json_safe(safe_records))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── API: Recruiter Profiles ────────────────────────────
@app.route("/api/profiles")
def api_profiles():
    """Return processed recruiter profiles."""
    try:
        team = load_team_from_db()
        profiles = []
        for p in team:
            profiles.append({
                "name": p.name,
                "submissions": int(p.submissions),
                "hm_reviewed": int(p.hm_reviewed),
                "shortlisted": int(p.shortlisted),
                "r1_scheduled": int(p.r1_scheduled),
                "r1_cleared": int(p.r1_cleared),
                "offered": int(p.offered),
                "avg_time_to_submit": round(float(p.avg_time_to_submit), 2),
                "avg_time_to_schedule": round(float(p.avg_time_to_schedule), 2),
                "jd_match_score": round(float(p.jd_match_score), 4),
                "weekly_submissions": p.weekly_submissions,
                "weekly_shortlists": p.weekly_shortlists,
            })
        return jsonify(profiles)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: Team Leaderboard ──────────────────────────────
@app.route("/api/leaderboard")
def api_leaderboard():
    """Return ranked team leaderboard with scores and risk levels."""
    try:
        team = load_team_from_db()
        engine = GapAnalysisEngine()
        df = engine.compare_team(team)
        records = df.reset_index().to_dict(orient="records")
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: Individual Recruiter Analysis ─────────────────
@app.route("/api/analyze/<recruiter_name>")
def api_analyze(recruiter_name):
    """Run full ML gap analysis for a specific recruiter."""
    try:
        team = load_team_from_db()
        engine = GapAnalysisEngine()

        # Find the recruiter
        target = None
        for r in team:
            if r.name.lower() == recruiter_name.lower():
                target = r
                break

        if target is None:
            return jsonify({"error": f"Recruiter '{recruiter_name}' not found"}), 404

        result = engine.analyze(target, team)

        # Convert Recommendation objects to dicts
        result["recommendations"] = [
            rec.to_dict() if hasattr(rec, "to_dict") else asdict(rec)
            for rec in result["recommendations"]
        ]

        return jsonify(make_json_safe(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: All Team Insights ─────────────────────────────
@app.route("/api/insights")
def api_insights():
    """Run ML gap analysis for every recruiter on the team."""
    try:
        team = load_team_from_db()
        engine = GapAnalysisEngine()

        all_insights = []
        for recruiter in team:
            result = engine.analyze(recruiter, team)
            result["recommendations"] = [
                rec.to_dict() if hasattr(rec, "to_dict") else asdict(rec)
                for rec in result["recommendations"]
            ]
            all_insights.append(make_json_safe(result))

        return jsonify(all_insights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Run Server ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  AblyWorks ATS — API Server")
    print("═" * 50)
    print("  Dashboard:  http://localhost:5000")
    print("  API Docs:")
    print("    GET /api/data          — Raw application data")
    print("    GET /api/profiles      — Recruiter profiles")
    print("    GET /api/leaderboard   — Team leaderboard")
    print("    GET /api/analyze/<name> — ML gap analysis")
    print("    GET /api/insights      — All team insights")
    print("═" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
