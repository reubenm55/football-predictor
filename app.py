



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
            
            # ===== AUTOMATIC MATCH PREDICTIONS =====

            import re
            import unicodedata
            from difflib import get_close_matches

            st.subheader("🔮 Predictions for Upcoming Fixtures")

            def normalize_team(name):
                name = unicodedata.normalize("NFKD", str(name))
                name = "".join(
                    c for c in name
                    if not unicodedata.combining(c)
                )
                name = re.sub(r"[^a-zA-Z0-9 ]", " ", name)
                name = re.sub(
                    r"\b(fc|afc|cf|sc|club)\b", "", name,
                    flags=re.IGNORECASE
                )
                return " ".join(name.lower().split())

            def find_team(api_name, league_name):
                known = league_team_history.get(league_name, {})
                names = list(known.keys())

                normalized = {
                    normalize_team(name): name for name in names
                }

                search_name = normalize_team(api_name)

                aliases = {
                    "brighton and hove albion": "brighton",
                    "wolverhampton wanderers": "wolves",
                    "manchester united": "man united",
                    "tottenham hotspur": "tottenham",
                    "west ham united": "west ham",
                    "newcastle united": "newcastle",
                    "nottingham forest": "nottm forest",
                    "paris saint germain": "paris sg",
                    "bayern munchen": "bayern munich"
                }

                search_name = aliases.get(search_name, search_name)

                if search_name in normalized:
                    return normalized[search_name]

                # Only accept a very close name match
                matches_found = get_close_matches(
                    search_name,
                    list(normalized.keys()),
                    n=1,
                    cutoff=0.88
                )

                if matches_found:
                    return normalized[matches_found[0]]

                return None

            def goal_estimate(results):
                if not results:
                    return None, None

                recent = results[-5:]
                scored = np.mean([m[1] for m in recent])
                conceded = np.mean([m[2] for m in recent])

                return scored, conceded

            prediction_rows = []

            for _, match_row in fixture_df.iterrows():
                league_name = match_row["League"]
                home_name = match_row["Home Team"]
                away_name = match_row["Away Team"]

                home_team_key = find_team(home_name, league_name)
                away_team_key = find_team(away_name, league_name)

               

                if not home_team_key or not away_team_key:
                    missing_teams = []
                    if not home_team_key:
                        missing_teams.append(home_name)
                    if not away_team_key:
                        missing_teams.append(away_name)

                    prediction_rows.append({
                        "Date (UTC)": match_row["Date (UTC)"],
                        "League": league_name,
                        "Match": f"{home_name} vs {away_name}",
                        "Prediction": "History missing: " + ", ".join(missing_teams),
                        "Home Win": "—",
                        "Draw": "—",
                        "Away Win": "—",
                        "Estimated Score": "—"
                    })
                    continue


                        "Home Win": "—",
                        "Draw": "—",
                        "Away Win": "—",
                        "Estimated Score": "—"
                    })
                    continue

                home_results = league_team_history[league_name][
                    home_team_key
                ]
                away_results = league_team_history[league_name][
                    away_team_key
                ]

                home_form = recent_form(league_name, home_team_key)
                away_form = recent_form(league_name, away_team_key)

                input_row = pd.DataFrame(
                    [home_form + away_form],
                    columns=feature_columns
                )

                probabilities = model.predict_proba(input_row)[0]
                classes = model.classes_

                probability_map = {
                    str(label): float(prob)
                    for label, prob in zip(classes, probabilities)
                }

                home_prob = probability_map.get("H", 0.0)
                draw_prob = probability_map.get("D", 0.0)
                away_prob = probability_map.get("A", 0.0)

                outcome = max(
                    [
                        (home_prob, f"{home_name} win"),
                        (draw_prob, "Draw"),
                        (away_prob, f"{away_name} win")
                    ],
                    key=lambda item: item[0]
                )[1]

                home_scored, home_conceded = goal_estimate(
                    home_results
                )
                away_scored, away_conceded = goal_estimate(
                    away_results
                )

                if home_scored is not None and away_scored is not None:
                    predicted_home_goals = round(
                        (home_scored + away_conceded) / 2
                    )
                    predicted_away_goals = round(
                        (away_scored + home_conceded) / 2
                    )
                    score = (
                        f"{predicted_home_goals} - "
                        f"{predicted_away_goals}"
                    )
                else:
                    score = "Unavailable"

                prediction_rows.append({
                    "Date (UTC)": match_row["Date (UTC)"],
                    "League": league_name,
                    "Match": f"{home_name} vs {away_name}",
                    "Prediction": outcome,
                    "Home Win": f"{home_prob:.1%}",
                    "Draw": f"{draw_prob:.1%}",
                    "Away Win": f"{away_prob:.1%}",
                    "Estimated Score": score
                })

            predictions_df = pd.DataFrame(prediction_rows)

            st.dataframe(
                predictions_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Probabilities are model estimates. Scores are rough "
                "estimates based on recent goals, not a trained score "
                "prediction model. Unmatched teams require updated "
                "historical data."
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
