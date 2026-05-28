"""
AblyWorks ATS — ML Recommendation System
Recruiter Performance Gap Analysis Module

Models:
  1. RejectionRiskClassifier      — Logistic Regression predicts P(rejection) at each stage
  2. RecruiterScorer              — Weighted KPI engine produces a 0-100 productivity score
  3. BottleneckDetector           — Z-score anomaly detection flags abnormal stage delays
  4. CollaborativeRecommender     — Item-based CF finds which top-performer habits to copy
  5. WeeklyTrendForecaster        — Linear regression forecasts next-week conversion rate
  6. GapAnalysisEngine            — Orchestrates all models → ranked actionable recommendations
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────
from dbConnections import fetch_data
#query = "SELECT * FROM applications"

#df = fetch_data(query)
#print("connected to database! successfully")
#print(df.head())
#print(df.shape)


def load_team_from_db():

    query = "SELECT * FROM applications"

    try:
        df = fetch_data(query)
    except Exception as db_err:
        print(f"Database fetch in ML recommender failed: {db_err}. Falling back to CSV.")
        df = pd.DataFrame()

    if df.empty:
        import os
        csv_path = "ats_recruiter_dataset_1000.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("No data in DB and CSV not found for ML recommender")

    # Clean columns to lowercase and map similar names to standard snake_case
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

    team = []

    recruiters = df["recruiter"].dropna().unique()

    for recruiter_name in recruiters:

        recruiter_df = df[df["recruiter"] == recruiter_name].copy()


        recruiter_df["shortlisted"] = (
            recruiter_df["shortlisted"]
            .astype(str)
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "YES": 1, "NO": 0})
            .fillna(0)
        )

        recruiter_df["interview_scheduled"] = (
            recruiter_df["interview_scheduled"]
            .astype(str)
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "YES": 1, "NO": 0})
            .fillna(0)
        )
            

        recruiter_df["submission_delay_days"] = (
            recruiter_df["submission_delay_days"].astype(float)
        )

        recruiter_df["interview_delay_days"] = (
            recruiter_df["interview_delay_days"].astype(float)
        )

        recruiter_df["skill_match_score"] = (
            recruiter_df["skill_match_score"].astype(float)
        )

        submissions = len(recruiter_df)

        hm_reviewed = len(
            recruiter_df[
                recruiter_df["hm_review_status"] == "Approved"
            ]
        )

        shortlisted = recruiter_df["shortlisted"].sum()

        r1_scheduled = recruiter_df["interview_scheduled"].sum()

        r1_cleared = len(
            recruiter_df[
                recruiter_df["interview_result"] == "Cleared"
            ]
        )

        offered = len(
            recruiter_df[
                recruiter_df["offer_status"] == "Yes"
            ]
        )

        avg_time_to_submit = recruiter_df[
            "submission_delay_days"
        ].mean()

        avg_time_to_schedule = recruiter_df[
            "interview_delay_days"
        ].mean()

        jd_match_score = (
            recruiter_df["skill_match_score"].mean() / 100
        )

        profile = RecruiterProfile(
            name=recruiter_name,
            submissions=submissions,
            hm_reviewed=hm_reviewed,
            shortlisted=shortlisted,
            r1_scheduled=r1_scheduled,
            r1_cleared=r1_cleared,
            offered=offered,
            avg_time_to_submit=avg_time_to_submit,
            avg_time_to_schedule=avg_time_to_schedule,
            jd_match_score=jd_match_score,
            weekly_submissions=[8,9,10,9,11,10,9,8],
            weekly_shortlists=[6,7,7,6,8,7,7,5],
        )

        team.append(profile)

    return team


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class RecruiterProfile:
    name: str
    submissions: int
    hm_reviewed: int
    shortlisted: int
    r1_scheduled: int
    r1_cleared: int
    offered: int
    avg_time_to_submit: float        # days
    avg_time_to_schedule: float      # days
    jd_match_score: float            # 0–1  (NLP JD vs resume match, mocked here)
    weeks_of_data: int = 8
    weekly_submissions: List[int] = field(default_factory=list)
    weekly_shortlists: List[int] = field(default_factory=list)

@dataclass
class Recommendation:
    priority: int                    # 1 = highest
    category: str                    # e.g. "Submission Quality"
    finding: str                     # what the model found
    action: str                      # concrete step
    expected_lift: str               # e.g. "+8% shortlist rate"
    confidence: float                # model confidence 0–1

    def to_dict(self):
        return {
            "priority": self.priority,
            "category": self.category,
            "finding": self.finding,
            "action": self.action,
            "expected_lift": self.expected_lift,
            "confidence": self.confidence,
        }


# ─────────────────────────────────────────────
# 1. REJECTION RISK CLASSIFIER
# ─────────────────────────────────────────────

class RejectionRiskClassifier:
    """
    Logistic Regression trained on per-stage features.
    Features: jd_match, time_to_submit, time_to_schedule, shortlist_rate, r1_rate
    Label:    1 = high rejection risk in next cycle, 0 = low
    """

    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self._train_on_synthetic_data()

    def _train_on_synthetic_data(self):
        """
        Synthetic training set: 200 recruiter-cycles.
        In production this is replaced with historical ATS data.
        """
        np.random.seed(7)
        n = 200
        jd_match       = np.random.beta(5, 2, n)          # most recruiters decent
        time_submit    = np.random.exponential(2, n)       # days, right-skewed
        time_schedule  = np.random.exponential(3, n)
        shortlist_rate = np.random.beta(6, 4, n)
        r1_rate        = np.random.beta(4, 6, n)

        # Risk is high when: low jd_match OR slow scheduling OR low conversion
        risk_score = (
            -1.8 * jd_match
            + 0.4 * time_submit
            + 0.6 * time_schedule
            - 1.2 * shortlist_rate
            - 0.9 * r1_rate
            + np.random.normal(0, 0.3, n)
        )
        labels = (risk_score > risk_score.mean()).astype(int)

        X = np.column_stack([jd_match, time_submit, time_schedule, shortlist_rate, r1_rate])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, labels)

    def predict(self, profile: RecruiterProfile) -> Tuple[str, float]:
        """Returns (risk_level, probability)"""
        shortlist_rate = profile.shortlisted / max(profile.submissions, 1)
        r1_rate        = profile.r1_cleared  / max(profile.r1_scheduled, 1)

        X = np.array([[
            profile.jd_match_score,
            profile.avg_time_to_submit,
            profile.avg_time_to_schedule,
            shortlist_rate,
            r1_rate
        ]])
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0][1]

        if prob >= 0.70:
            level = "HIGH"
        elif prob >= 0.45:
            level = "MEDIUM"
        else:
            level = "LOW"
        return level, round(prob, 3)


# ─────────────────────────────────────────────
# 2. RECRUITER SCORER
# ─────────────────────────────────────────────

class RecruiterScorer:
    """
    Weighted KPI engine → 0–100 productivity score.
    Weights reflect business priority; adjust in production.
    """

    WEIGHTS = {
        "submission_to_shortlist": 0.25,
        "shortlist_to_interview":  0.20,
        "interview_to_offer":      0.25,
        "time_to_submit":          0.10,   # inverted: lower is better
        "time_to_schedule":        0.10,   # inverted
        "jd_match_score":          0.10,
    }

    # Benchmark targets (industry / team best practice)
    TARGETS = {
        "submission_to_shortlist": 0.70,
        "shortlist_to_interview":  0.65,
        "interview_to_offer":      0.40,
        "time_to_submit":          1.0,    # days
        "time_to_schedule":        2.0,
        "jd_match_score":          0.80,
    }

    def score(self, profile: RecruiterProfile) -> Tuple[float, Dict[str, float]]:
        """Returns (overall_score_0_100, kpi_breakdown_dict)"""
        kpis = {
            "submission_to_shortlist": profile.shortlisted / max(profile.submissions, 1),
            "shortlist_to_interview":  profile.r1_scheduled / max(profile.shortlisted, 1),
            "interview_to_offer":      profile.offered / max(profile.r1_cleared, 1),
            "time_to_submit":          profile.avg_time_to_submit,
            "time_to_schedule":        profile.avg_time_to_schedule,
            "jd_match_score":          profile.jd_match_score,
        }

        component_scores = {}
        for kpi, value in kpis.items():
            target = self.TARGETS[kpi]
            if kpi in ("time_to_submit", "time_to_schedule"):
                # Lower is better — invert
                raw = max(0, 1 - (value - target) / target)
            else:
                raw = min(value / target, 1.0)
            component_scores[kpi] = round(raw * 100, 1)

        overall = sum(
            component_scores[kpi] * w
            for kpi, w in self.WEIGHTS.items()
        )
        return round(overall, 1), component_scores


# ─────────────────────────────────────────────
# 3. BOTTLENECK DETECTOR
# ─────────────────────────────────────────────

class BottleneckDetector:
    """
    Z-score based anomaly detection on stage-transition times.
    Flags stages where a recruiter is >1.5σ slower than the team mean.
    """

    def detect(
        self,
        recruiter: RecruiterProfile,
        team: List[RecruiterProfile]
    ) -> List[Dict]:
        stage_times = {
            "Submit speed":      [r.avg_time_to_submit   for r in team],
            "Scheduling speed":  [r.avg_time_to_schedule for r in team],
        }
        recruiter_times = {
            "Submit speed":      recruiter.avg_time_to_submit,
            "Scheduling speed":  recruiter.avg_time_to_schedule,
        }

        bottlenecks = []
        for stage, times in stage_times.items():
            mu  = np.mean(times)
            std = np.std(times) or 1.0
            z   = (recruiter_times[stage] - mu) / std
            if z > 1.5:
                bottlenecks.append({
                    "stage":        stage,
                    "z_score":      round(z, 2),
                    "recruiter_val": recruiter_times[stage],
                    "team_avg":     round(mu, 2),
                    "severity":     "HIGH" if z > 2.5 else "MEDIUM"
                })
        return bottlenecks


# ─────────────────────────────────────────────
# 4. COLLABORATIVE RECOMMENDER
# ─────────────────────────────────────────────

class CollaborativeRecommender:
    """
    Item-based collaborative filtering.
    Finds the top performer most similar to the target recruiter
    (based on KPI vector cosine similarity) and surfaces their
    strongest differentiating habits as recommendations.
    """

    def _build_feature_vector(self, p: RecruiterProfile) -> np.ndarray:
        return np.array([
            p.shortlisted / max(p.submissions, 1),
            p.r1_scheduled / max(p.shortlisted, 1),
            p.offered / max(p.r1_cleared, 1),
            1 / max(p.avg_time_to_submit, 0.1),
            1 / max(p.avg_time_to_schedule, 0.1),
            p.jd_match_score,
        ])

    def recommend(
        self,
        target: RecruiterProfile,
        team: List[RecruiterProfile],
        top_n: int = 2
    ) -> List[Dict]:
        """
        Returns list of {peer, similarity, learn_from} dicts.
        peer = RecruiterProfile of the closest high-performer.
        """
        scorer = RecruiterScorer()
        scored_team = [(r, scorer.score(r)[0]) for r in team if r.name != target.name]
        top_peers = sorted(scored_team, key=lambda x: -x[1])[:top_n]

        target_vec = self._build_feature_vector(target).reshape(1, -1)
        results = []
        for peer, peer_score in top_peers:
            peer_vec = self._build_feature_vector(peer).reshape(1, -1)
            sim = cosine_similarity(target_vec, peer_vec)[0][0]

            # Identify where the peer is strongest vs target
            target_kpis = self._build_feature_vector(target)
            peer_kpis   = self._build_feature_vector(peer)
            diffs = peer_kpis - target_kpis
            strongest_idx = int(np.argmax(diffs))
            kpi_names = [
                "submission→shortlist rate",
                "shortlist→R1 rate",
                "interview→offer rate",
                "submission speed",
                "scheduling speed",
                "JD match score"
            ]
            results.append({
                "peer":        peer.name,
                "peer_score":  peer_score,
                "similarity":  round(float(sim), 3),
                "learn_from":  kpi_names[strongest_idx],
                "peer_delta":  round(float(diffs[strongest_idx]), 3)
            })
        return results


# ─────────────────────────────────────────────
# 5. WEEKLY TREND FORECASTER
# ─────────────────────────────────────────────

class WeeklyTrendForecaster:
    """
    Linear regression on weekly shortlist-rate history.
    Forecasts next week's rate and detects declining trends.
    """

    def forecast(self, profile: RecruiterProfile) -> Dict:
        if len(profile.weekly_submissions) < 3:
            return {"forecast": None, "trend": "insufficient_data"}

        subs  = np.array(profile.weekly_submissions, dtype=float)
        sls   = np.array(profile.weekly_shortlists,  dtype=float)
        rates = np.divide(sls, subs, out=np.zeros_like(sls), where=subs > 0)

        X = np.arange(len(rates)).reshape(-1, 1)
        model = LinearRegression().fit(X, rates)
        next_week_rate = float(model.predict([[len(rates)]])[0])
        slope = float(model.coef_[0])

        trend = "improving" if slope > 0.005 else "declining" if slope < -0.005 else "stable"
        return {
            "forecast_rate":     round(max(0, min(next_week_rate, 1)), 3),
            "trend":             trend,
            "slope_per_week":    round(slope, 4),
            "current_avg_rate":  round(float(rates.mean()), 3),
        }


# ─────────────────────────────────────────────
# 6. GAP ANALYSIS ENGINE  (orchestrator)
# ─────────────────────────────────────────────

class GapAnalysisEngine:
    """
    Orchestrates all models → produces a ranked list of
    Recommendation objects for a given recruiter.
    """

    def __init__(self):
        self.risk_clf     = RejectionRiskClassifier()
        self.scorer       = RecruiterScorer()
        self.bottleneck   = BottleneckDetector()
        self.cf           = CollaborativeRecommender()
        self.forecaster   = WeeklyTrendForecaster()

    def analyze(
        self,
        recruiter: RecruiterProfile,
        team: List[RecruiterProfile]
    ) -> Dict:
        recommendations: List[Recommendation] = []
        priority = 1

        # ── Score ─────────────────────────────
        overall_score, kpi_breakdown = self.scorer.score(recruiter)

        # ── Rejection Risk ────────────────────
        risk_level, risk_prob = self.risk_clf.predict(recruiter)
        if risk_level in ("HIGH", "MEDIUM"):
            recommendations.append(Recommendation(
                priority=priority,
                category="Rejection Risk",
                finding=f"Model predicts {risk_level} rejection risk (p={risk_prob:.0%}). "
                        f"Key driver: JD match score = {recruiter.jd_match_score:.2f}",
                action="Run a 30-min JD alignment session before next batch submission. "
                       "Use keyword overlap checker against the job description.",
                expected_lift="+12–18% shortlist rate",
                confidence=risk_prob
            ))
            priority += 1

        # ── Bottlenecks ───────────────────────
        bottlenecks = self.bottleneck.detect(recruiter, team)
        for b in bottlenecks:
            recommendations.append(Recommendation(
                priority=priority,
                category="Stage Bottleneck",
                finding=f"{b['stage']} is {b['z_score']}σ above team average "
                        f"({b['recruiter_val']}d vs team avg {b['team_avg']}d). "
                        f"Severity: {b['severity']}.",
                action="Set a calendar block: submit profiles within 24h of sourcing. "
                       "Use ATS auto-reminder for interview scheduling.",
                expected_lift=f"-{round((b['recruiter_val'] - b['team_avg']) * 0.6, 1)}d avg delay",
                confidence=min(0.5 + b['z_score'] * 0.1, 0.95)
            ))
            priority += 1

        # ── Funnel Drop Analysis ──────────────
        hm_drop = 1 - recruiter.hm_reviewed / max(recruiter.submissions, 1)
        sl_drop  = 1 - recruiter.shortlisted  / max(recruiter.hm_reviewed, 1)
        r1_drop  = 1 - recruiter.r1_cleared   / max(recruiter.r1_scheduled, 1)

        worst_stage = max(
            [("HM Review", hm_drop), ("Shortlisting", sl_drop), ("R1", r1_drop)],
            key=lambda x: x[1]
        )
        recommendations.append(Recommendation(
            priority=priority,
            category="Funnel Drop-off",
            finding=f"Highest candidate loss at {worst_stage[0]} stage "
                    f"({worst_stage[1]:.0%} drop-off rate).",
            action=f"Request structured HM feedback for every {worst_stage[0]} rejection. "
                   f"Identify top 3 rejection reasons and pre-screen against them.",
            expected_lift=f"+{round(worst_stage[1] * 15, 0):.0f}% conversion at this stage",
            confidence=0.82
        ))
        priority += 1

        # ── Collaborative Filtering ───────────
        peers = self.cf.recommend(recruiter, team)
        for p in peers:
            recommendations.append(Recommendation(
                priority=priority,
                category="Peer Learning",
                finding=f"Closest high-performer: {p['peer']} (score {p['peer_score']:.0f}/100, "
                        f"similarity {p['similarity']:.0%}). "
                        f"Biggest gap vs peer: {p['learn_from']} (Δ {p['peer_delta']:+.2f}).",
                action=f"Shadow {p['peer']} for one hiring cycle, focusing on {p['learn_from']}. "
                       f"Ask them to share their pre-screening checklist.",
                expected_lift=f"+{abs(round(p['peer_delta'] * 20, 0)):.0f}% on {p['learn_from']}",
                confidence=p['similarity']
            ))
            priority += 1

        # ── Trend Forecast ────────────────────
        forecast = self.forecaster.forecast(recruiter)
        if forecast["trend"] == "declining":
            recommendations.append(Recommendation(
                priority=priority,
                category="Trend Alert",
                finding=f"Shortlist rate declining at {forecast['slope_per_week']:+.1%}/week. "
                        f"Projected next-week rate: {forecast['forecast_rate']:.0%}.",
                action="Review last 2 weeks' rejected profiles with manager. "
                       "Identify if a specific client or role is skewing numbers.",
                expected_lift="Stabilise rate above team avg",
                confidence=0.75
            ))

        # ── Weak KPI Highlights ───────────────
        weak_kpis = {k: v for k, v in kpi_breakdown.items() if v < 60}
        if weak_kpis:
            worst_kpi = min(weak_kpis, key=weak_kpis.get)
            recommendations.append(Recommendation(
                priority=priority + 1,
                category="KPI Improvement",
                finding=f"Lowest KPI: {worst_kpi.replace('_', ' ')} = {weak_kpis[worst_kpi]:.0f}/100.",
                action=f"Set a 4-week target: improve {worst_kpi.replace('_', ' ')} by 15 points. "
                       f"Track daily in ATS dashboard.",
                expected_lift=f"+15 pts on {worst_kpi.replace('_', ' ')}",
                confidence=0.70
            ))

        return {
            "recruiter":         recruiter.name,
            "overall_score":     overall_score,
            "risk_level":        risk_level,
            "risk_probability":  risk_prob,
            "kpi_breakdown":     kpi_breakdown,
            "trend_forecast":    forecast,
            "recommendations":   sorted(recommendations, key=lambda r: r.priority),
            "bottlenecks_found": len(bottlenecks),
        }

    def compare_team(self, team: List[RecruiterProfile]) -> pd.DataFrame:
        """Returns a DataFrame ranking all recruiters with scores + risk."""
        rows = []
        for r in team:
            score, _ = self.scorer.score(r)
            risk, prob = self.risk_clf.predict(r)
            forecast = self.forecaster.forecast(r)
            rows.append({
                "Recruiter":           r.name,
                "Score":               score,
                "Risk":                risk,
                "Risk Prob":           f"{prob:.0%}",
                "Submissions":         r.submissions,
                "Shortlist Rate":      f"{r.shortlisted/max(r.submissions,1):.0%}",
                "Interview Rate":      f"{r.r1_scheduled/max(r.shortlisted,1):.0%}",
                "Offer Rate":          f"{r.offered/max(r.r1_cleared,1):.0%}",
                "Trend":               forecast["trend"],
                "JD Match":            f"{r.jd_match_score:.0%}",
            })
        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        df.index += 1
        return df


# ─────────────────────────────────────────────
# DEMO  —  run with: python ml_recommender.py
# ─────────────────────────────────────────────

def main():
    # Sample team data (replace with real DB query in production)
    '''
    team = [
        RecruiterProfile(
            name="Priya Sharma",
            submissions=74, hm_reviewed=68, shortlisted=52,
            r1_scheduled=38, r1_cleared=29, offered=13,
            avg_time_to_submit=1.2, avg_time_to_schedule=1.8,
            jd_match_score=0.84,
            weekly_submissions=[8,9,10,9,11,10,9,8],
            weekly_shortlists=[6,7,7,6,8,7,7,5],
        ),
        RecruiterProfile(
            name="Arjun Mehta",
            submissions=82, hm_reviewed=58, shortlisted=41,
            r1_scheduled=26, r1_cleared=18, offered=7,
            avg_time_to_submit=2.4, avg_time_to_schedule=4.1,
            jd_match_score=0.61,
            weekly_submissions=[10,11,10,12,9,11,10,9],
            weekly_shortlists=[7,6,5,5,4,4,3,3],       # declining
        ),
        RecruiterProfile(
            name="Kavya Nair",
            submissions=56, hm_reviewed=50, shortlisted=38,
            r1_scheduled=27, r1_cleared=20, offered=9,
            avg_time_to_submit=1.5, avg_time_to_schedule=2.2,
            jd_match_score=0.78,
            weekly_submissions=[7,7,7,7,7,7,7,7],
            weekly_shortlists=[5,5,5,5,5,5,5,5],
        ),
        RecruiterProfile(
            name="Rahul Verma",
            submissions=36, hm_reviewed=28, shortlisted=19,
            r1_scheduled=11, r1_cleared=7,  offered=2,
            avg_time_to_submit=3.1, avg_time_to_schedule=5.8,
            jd_match_score=0.49,
            weekly_submissions=[4,5,4,5,4,5,4,5],
            weekly_shortlists=[3,3,2,2,2,1,1,1],       # declining
        ),
    ]
    '''
    team = load_team_from_db()

    engine = GapAnalysisEngine()

    # ── Team leaderboard ──────────────────────
    print("\n" + "═"*70)
    print("  ABLYWORKS ATS — RECRUITER PERFORMANCE GAP ANALYSIS")
    print("═"*70)
    print("\n📊  TEAM LEADERBOARD\n")
    df = engine.compare_team(team)
    print(df.to_string())

    # ── Individual deep-dive ──────────────────
    #target = next(r for r in team if r.name == "Rahul Verma")
    target = next(r for r in team if r.name == "Rahul")
    result = engine.analyze(target, team)

    print(f"\n\n{'─'*70}")
    print(f"  DEEP-DIVE: {result['recruiter']}")
    print(f"{'─'*70}")
    print(f"  Productivity Score : {result['overall_score']}/100")
    print(f"  Rejection Risk     : {result['risk_level']}  (p={result['risk_probability']:.0%})")
    print(f"  Trend              : {result['trend_forecast']['trend'].upper()}  "
          f"(next-week forecast: {result['trend_forecast'].get('forecast_rate', 0):.0%})")

    print("\n  KPI BREAKDOWN:")
    for kpi, val in result['kpi_breakdown'].items():
        bar = "█" * int(val / 5) + "░" * (20 - int(val / 5))
        flag = "⚠" if val < 60 else "✓"
        print(f"  {flag}  {kpi:<30} {bar}  {val:.0f}/100")

    print(f"\n  RANKED RECOMMENDATIONS  ({len(result['recommendations'])} found)")
    print(f"{'─'*70}")
    for rec in result['recommendations']:
        print(f"\n  [{rec.priority}] {rec.category.upper()}  (confidence: {rec.confidence:.0%})")
        print(f"      Finding : {rec.finding}")
        print(f"      Action  : {rec.action}")
        print(f"      Lift    : {rec.expected_lift}")

    print(f"\n{'═'*70}\n")


if __name__ == "__main__":
    main()
