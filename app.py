

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="centered"
)

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
    st.error(f"Could not load the model: {e}")
    st.stop()

st.title("⚽ Football Match Predictor")
st.write("Predict the likely outcome using historical match data.")

home_team = st.selectbox("Home team", teams)
away_options = [t for t in teams if t != home_team]
away_team = st.selectbox("Away team", away_options)

def recent_form(team):
    recent = team_history.get(team, [])[-5:]

    if not recent:
        return [0, 0, 0, 0, 0]

    return [
        float(np.mean([x[0] for x in recent])),
        float(np.mean([x[1] for x in recent])),
        float(np.mean([x[2] for x in recent])),
        int(sum(x[0] == 3 for x in recent)),
        int(sum(x[0] == 0 for x in recent))
    ]

if st.button("Predict Match", type="primary"):
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
        row = row[feature_columns]
        probabilities = model.predict_proba(row)[0]
        classes = model.classes_ if hasattr(
            model, "classes_"
        ) else model.named_steps[
            "randomforestclassifier"
        ].classes_

        labels = {
            "H": "Home win",
            "D": "Draw",
            "A": "Away win"
        }

        st.subheader(f"{home_team} vs {away_team}")

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True
        )

        st.success(
            "Most likely outcome: "
            + labels.get(str(ranked[0][0]), str(ranked[0][0]))
        )

        for outcome, probability in ranked:
            st.write(
                f"**{labels.get(str(outcome), str(outcome))}:** "
                f"{probability:.1%}"
            )
            st.progress(float(probability))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption(
    "Predictions are estimates, not guarantees. "
    "Accuracy depends on the training data and model."
)

    )
