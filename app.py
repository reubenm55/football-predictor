


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Football Predictor | AI Match Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PROFESSIONAL DESIGN ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #101d31 100%);
        color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background: #0b1424;
        border-right: 1px solid #26364c;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9aacc4;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 25px;
    }

    .section-card {
        background: rgba(20, 36, 57, 0.9);
        border: 1px solid #2a405b;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .team-label {
        color: #a9bad0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .team-name {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }

    .metric-label {
        color: #a9bad0;
        font-size: 0.9rem;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 800;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #0878f9, #00b894);
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 48px;
        font-weight: 750;
        width: 100%;
    }

    div.stButton > button:hover {
        border: 1px solid #ffffff;
        color: white;
    }

    div[data-testid="stProgressBar"] > div > div {
        background-color: #00c897;
    }

    .footer {
        color: #8294ad;
        text-align: center;
        font-size: 0.8rem;
        padding-top: 30px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD YOUR EXISTING MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("football_predictor_model.joblib")


try:
    bundle = load_model()

    model = bundle["model"]
    team_history = bundle["team_history"]
    feature_columns = bundle["feature_columns"]
    teams = sorted(bundle["teams"])

except Exception as e:
    st.error("The model could not be loaded.")
    st.write("Check that football_predictor_model.joblib is in your GitHub repository.")
    st.code(str(e))
    st.stop()


# ---------------- RECENT FORM ----------------
def recent_form(team):
    recent = team_history.get(team, [])[-5:]

    if not recent:
        return [0.0, 0.0, 0.0, 0, 0]

    return [
        float(np.mean([x[0] for x in recent])),
        float(np.mean([x[1] for x in recent])),
        float(np.mean([x[2] for x in recent])),
        int(sum(x[0] == 3 for x in recent)),
        int(sum(x[0] == 0 for x in recent))
    ]


def form_summary(team):
    recent = team_history.get(team, [])[-5:]

    if not recent:
        return "No recent form available"

    # Assumes each history record stores points in position 0.
    # 3 points = win, 1 = draw, 0 = loss.
    results = []
    for match in recent:
        points = match[0]
        if points == 3:
            results.append("W")
        elif points == 1:
            results.append("D")
        else:
            results.append("L")

    return " · ".join(results)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚽ FOOTBALL AI")
    st.caption("MATCH ANALYSIS PLATFORM")
    st.divider()

    st.markdown("### Dashboard")
    st.write("🏟️ Match predictor")
    st.write("📊 Outcome probabilities")
    st.write("📈 Recent team form")

    st.divider()
    st.caption("Model: Random Forest")
    st.caption("Data: Historical match results")


# ---------------- MAIN HEADER ----------------
st.markdown(
    '<div class="main-title">⚽ FOOTBALL PREDICTOR</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered match analysis using historical team performance.</div>',
    unsafe_allow_html=True
)

st.markdown("### 🏆 Predict a Match")
st.write("Choose the home and away teams, then run your prediction.")

left, middle, right = st.columns([5, 1, 5])

with left:
    home_team = st.selectbox(
        "🏠 Home team",
        teams,
        key="home_team"
    )

with middle:
    st.markdown(
        "<div style='text-align:center; padding-top:35px;"
        "font-size:1.4rem; font-weight:800; color:#9aacc4;'>VS</div>",
        unsafe_allow_html=True
    )

with right:
    away_options = [team for team in teams if team != home_team]
    away_team = st.selectbox(
        "✈️ Away team",
        away_options,
        key="away_team"
    )

st.markdown("")

predict_clicked = st.button(
    "⚡ ANALYSE MATCH",
    type="primary",
    use_container_width=True
)


# ---------------- PREDICTION ----------------
if predict_clicked:
    home = recent_form(home_team)
    away = recent_form(away_team)

    row = pd.DataFrame([{
        "HomePoints": home[0],
        "HomeScored": home[1],
        "HomeConceded": home[2],
        "HomeWins": home[3],
        "HomeLosses": home[4],
        "AwayPoints": away[0],
        "AwayScored": away[1],
        "AwayConceded": away[2],
        "AwayWins": away[3],
        "AwayLosses": away[4]
    }])

    try:
        # Keep the exact feature order used during training.
        row = row[feature_columns]

        probabilities = model.predict_proba(row)[0]
        classes = model.classes_

        labels = {
            "H": "Home win",
            "D": "Draw",
            "A": "Away win"
        }

        names = {
            "H": home_team,
            "D": "Draw",
            "A": away_team
        }

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True
        )

        best_class = str(ranked[0][0])
        best_label = labels.get(best_class, best_class)

        st.markdown("---")
        st.markdown("### 📊 Match Analysis")

        home_box, draw_box, away_box = st.columns(3)

        probability_map = {
            str(outcome): float(probability)
            for outcome, probability in zip(classes, probabilities)
        }

        with home_box:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">🏠 HOME WIN</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{probability_map.get("H", 0):.1%}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**{home_team}**")
            st.progress(probability_map.get("H", 0.0))
            st.markdown('</div>', unsafe_allow_html=True)

        with draw_box:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">🤝 DRAW</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{probability_map.get("D", 0):.1%}</div>',
                unsafe_allow_html=True
            )
            st.markdown("**Draw**")
            st.progress(probability_map.get("D", 0.0))
            st.markdown('</div>', unsafe_allow_html=True)

        with away_box:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">✈️ AWAY WIN</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{probability_map.get("A", 0):.1%}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**{away_team}**")
            st.progress(probability_map.get("A", 0.0))
            st.markdown('</div>', unsafe_allow_html=True)

        st.success(f"Most likely outcome according to the model: {best_label}")

        st.markdown("### 📈 Recent Form (Last 5 Recorded Matches)")
        form_left, form_right = st.columns(2)

        with form_left:
            st.markdown(f"**{home_team}**")
            st.info(form_summary(home_team))

        with form_right:
            st.markdown(f"**{away_team}**")
            st.info(form_summary(away_team))

        st.caption(
            "Probabilities are model estimates, not guarantees. "
            "They may not be calibrated probabilities of the true outcome."
        )

    except Exception as e:
        st.error("Prediction failed. Your model is still loaded, but the prediction step needs checking.")
        st.code(str(e))


# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">FOOTBALL PREDICTOR · Built with Python, '
    'Streamlit & Machine Learning<br>Predictions are estimates, '
    'not guarantees.</div>',
    unsafe_allow_html=True
)

)
