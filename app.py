



import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #101d33);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div.stButton > button {
    background: #20c997;
    color: #06111f;
    font-weight: bold;
    border-radius: 10px;
    width: 100%;
    min-height: 45px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("multi_league_predictor.joblib")

try:
    bundle = load_model()
    model = bundle["model"]
    league_team_history = bundle["league_team_history"]
    teams_by_league = bundle["teams_by_league"]
    feature_columns = bundle["feature_columns"]
    leagues = bundle["leagues"]
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

st.title("⚽ Football Predictor")
st.write("### Multi-League Match Prediction Dashboard")
st.caption("Machine learning predictions based on historical match results.")

st.divider()

# League selection
league = st.selectbox("Choose a league", leagues)

available_teams = teams_by_league[league]

if len(available_teams) < 2:
    st.warning("Not enough teams are available in this league.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "🏠 Home team",
        available_teams,
        index=0
    )

with col2:
    away_options = [t for t in available_teams if t != home_team]
    away_team = st.selectbox(
        "✈️ Away team",
        away_options,
        index=0
    )

def recent_form(league_name, team):
    history = league_team_history.get(league_name, {})
    recent = history.get(team, [])[-5:]

    if not recent:
        return [0.0, 0.0, 0.0, 0, 0]

    return [
        float(np.mean([m[0] for m in recent])),
        float(np.mean([m[1] for m in recent])),
        float(np.mean([m[2] for m in recent])),
        int(sum(m[0] == 3 for m in recent)),
        int(sum(m[0] == 0 for m in recent))
    ]

def form_summary(league_name, team):
    history = league_team_history.get(league_name, {})
    recent = history.get(team, [])[-5:]

    if not recent:
        return "No recent results"

    results = []
    for points, goals_for, goals_against in recent:
        if points == 3:
            results.append("W")
        elif points == 1:
            results.append("D")
        else:
            results.append("L")

    return " - ".join(results)

st.divider()

if home_team == away_team:
    st.warning("Choose two different teams.")
else:
    if st.button("🔮 Predict Match"):
        home_form = recent_form(league, home_team)
        away_form = recent_form(league, away_team)

        row = pd.DataFrame(
            [home_form + away_form],
            columns=feature_columns
        )

        try:
            probabilities = model.predict_proba(row)[0]
            classes = model.classes_

            probability_map = {
                str(label): float(prob)
                for label, prob in zip(classes, probabilities)
            }

            # FTR labels: H = home win, D = draw, A = away win
            home_prob = probability_map.get("H", 0.0)
            draw_prob = probability_map.get("D", 0.0)
            away_prob = probability_map.get("A", 0.0)

            st.subheader("📊 Match Prediction")

            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric("🏠 Home Win", f"{home_prob:.1%}")
                st.progress(min(max(home_prob, 0.0), 1.0))

            with p2:
                st.metric("🤝 Draw", f"{draw_prob:.1%}")
                st.progress(min(max(draw_prob, 0.0), 1.0))

            with p3:
                st.metric("✈️ Away Win", f"{away_prob:.1%}")
                st.progress(min(max(away_prob, 0.0), 1.0))

            outcomes = {
                "H": f"{home_team} win",
                "D": "Draw",
                "A": f"{away_team} win"
            }

            predicted_label = str(classes[np.argmax(probabilities)])

            st.success(
                "Most likely outcome: "
                + outcomes.get(predicted_label, predicted_label)
            )

            st.info(
                "These are estimates from historical results, "
                "not guarantees. The model does not use live odds, "
                "confirmed lineups, or current injury information."
            )

            st.subheader("📈 Recent Form")

            f1, f2 = st.columns(2)

            with f1:
                st.write(f"**{home_team}**")
                st.write(form_summary(league, home_team))

            with f2:
                st.write(f"**{away_team}**")
                st.write(form_summary(league, away_team))

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "FOOTBALL PREDICTOR | Built with Python, "
    "Streamlit and Machine Learning"
)

# ============== UPCOMING FIXTURES ==============

import requests
from datetime import date, timedelta

st.divider()
st.header("📅 Upcoming Fixtures")
st.write("Explore scheduled matches across your supported leagues.")

API_TOKEN = st.secrets.get("FOOTBALL_DATA_TOKEN", "")

COMPETITION_CODES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "Ligue 1": "FL1",
}

if not API_TOKEN:
    st.warning(
        "Football API token not found. Add "
        "FOOTBALL_DATA_TOKEN to Streamlit Secrets."
    )
else:
    fixture_league = st.selectbox(
        "Filter fixtures by league",
        ["All leagues"] + list(COMPETITION_CODES.keys()),
        key="fixture_league"
    )

    days_ahead = st.selectbox(
        "Show matches in the next",
        [7, 14, 30],
        index=1,
        format_func=lambda n: f"{n} days",
        key="fixture_days"
    )

    @st.cache_data(ttl=1800)
    def fetch_fixtures(token, competition_code, start_date, end_date):
        url = (
            "https://api.football-data.org/v4/competitions/"
            f"{competition_code}/matches"
        )

        response = requests.get(
            url,
            headers={"X-Auth-Token": token},
            params={
                "dateFrom": start_date,
                "dateTo": end_date
            },
            timeout=20
        )

        response.raise_for_status()
        return response.json().get("matches", [])

    if st.button("🔄 Load Upcoming Fixtures"):
        start = date.today()
        end = start + timedelta(days=days_ahead)

        leagues_to_fetch = (
            list(COMPETITION_CODES.items())
            if fixture_league == "All leagues"
            else [(fixture_league,
                   COMPETITION_CODES[fixture_league])]
        )

        fixtures = []
        failed_leagues = []

        with st.spinner("Fetching fixtures..."):
            for league_name, code in leagues_to_fetch:
                try:
                    matches_data = fetch_fixtures(
                        API_TOKEN,
                        code,
                        start.isoformat(),
                        end.isoformat()
                    )

                    for match in matches_data:
                        if match.get("status") not in [
                            "SCHEDULED", "TIMED"
                        ]:
                            continue

                        fixtures.append({
                            "Date (UTC)": match.get("utcDate", ""),
                            "League": league_name,
                            "Home Team": match.get(
                                "homeTeam", {}
                            ).get("name", "Unknown"),
                            "Away Team": match.get(
                                "awayTeam", {}
                            ).get("name", "Unknown"),
                            "Status": match.get("status", "")
                        })

                except requests.HTTPError as e:
                    failed_leagues.append(
                        f"{league_name}: HTTP {e.response.status_code}"
                    )
                except requests.RequestException:
                    failed_leagues.append(
                        f"{league_name}: connection problem"
                    )
                except Exception:
                    failed_leagues.append(
                        f"{league_name}: unexpected response"
                    )

        if fixtures:
            fixture_df = pd.DataFrame(fixtures)
            fixture_df["Date (UTC)"] = pd.to_datetime(
                fixture_df["Date (UTC)"], utc=True, errors="coerce"
            )
            fixture_df = fixture_df.sort_values("Date (UTC)")

            st.success(f"Found {len(fixture_df)} upcoming matches.")

            st.dataframe(
                fixture_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No upcoming fixtures were returned for "
                "the selected period."
            )

        if failed_leagues:
            st.warning(
                "Some leagues could not be loaded. "
                "Your API plan may not include every competition."
            )
            for problem in failed_leagues:
                st.caption(problem)

    st.caption(
        "Fixture dates are shown in UTC. "
        "Availability depends on your API provider and subscription."
    )
