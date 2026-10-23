



import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re
import unicodedata
import hashlib
import sqlite3
if not st.session_state.get("authenticated", False):
    authentication_screen()
    st.stop()

st.title("⚽ Football Predictor")

from datetime import date, timedelta
from difflib import get_close_matches


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide"
)


st.markdown("""
<style>
/* Main dashboard background */
.stApp {
    background:
        radial-gradient(ellipse at top left, #173b50 0%, transparent 45%),
        linear-gradient(135deg, #07111f, #0b192b 55%, #102b35);
    color: #f1f5f9;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Headings */
h1 {
    color: #ffffff !important;
    font-size: 2.7rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #dbeafe;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #16a34a, #20c997);
    color: #ffffff;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    min-height: 46px;
    width: 100%;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #15803d, #0d9488);
    color: #ffffff;
    border: none;
    transform: translateY(-1px);
}

/* Dropdowns and input fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #13263a;
    border: 1px solid #29445c;
    border-radius: 10px;
    color: white;
}

/* Prediction metrics */
div[data-testid="stMetric"] {
    background: rgba(19, 38, 58, 0.88);
    border: 1px solid #29445c;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #93c5fd;
}

div[data-testid="stMetricValue"] {
    color: #4ade80;
    font-weight: 800;
}

/* Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #29445c;
    border-radius: 12px;
    overflow: hidden;
}

/* Separators */
hr {
    border-color: #29445c;
}

/* Success messages */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Footer text */
.stCaption {
    color: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)



# ==================================================
# LOAD MACHINE LEARNING MODEL
# ==================================================

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


# ==================================================
# HELPER FUNCTIONS
# ==================================================

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


def goal_estimate(results):
    if not results:
        return None, None

    recent = results[-5:]

    scored = np.mean([m[1] for m in recent])
    conceded = np.mean([m[2] for m in recent])

    return float(scored), float(conceded)


def estimate_score(home_results, away_results):
    home_scored, home_conceded = goal_estimate(home_results)
    away_scored, away_conceded = goal_estimate(away_results)

    if (
        home_scored is None
        or home_conceded is None
        or away_scored is None
        or away_conceded is None
    ):
        return None

    home_goals = max(
        0,
        round((home_scored + away_conceded) / 2)
    )

    away_goals = max(
        0,
        round((away_scored + home_conceded) / 2)
    )

    return f"{home_goals} - {away_goals}"


def normalize_team(name):
    name = unicodedata.normalize("NFKD", str(name))

    name = "".join(
        character for character in name
        if not unicodedata.combining(character)
    )

    name = re.sub(r"[^a-zA-Z0-9 ]", " ", name)

    name = re.sub(
        r"\b(fc|afc|cf|sc|club)\b",
        "",
        name,
        flags=re.IGNORECASE
    )

    return " ".join(name.lower().split())


def find_team(api_name, league_name):
    known = league_team_history.get(league_name, {})

    normalized = {
        normalize_team(name): name
        for name in known.keys()
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

    matches = get_close_matches(
        search_name,
        list(normalized.keys()),
        n=1,
        cutoff=0.88
    )

    if matches:
        return normalized[matches[0]]

    return None


def predict_outcome(league_name, home_team, away_team):
    home_form = recent_form(league_name, home_team)
    away_form = recent_form(league_name, away_team)

    input_row = pd.DataFrame(
        [home_form + away_form],
        columns=feature_columns
    )

    probabilities = model.predict_proba(input_row)[0]
    classes = model.classes_

    probability_map = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    home_prob = probability_map.get("H", 0.0)
    draw_prob = probability_map.get("D", 0.0)
    away_prob = probability_map.get("A", 0.0)

    outcome = max(
        [
            (home_prob, f"{home_team} win"),
            (draw_prob, "Draw"),
            (away_prob, f"{away_team} win")
        ],
        key=lambda item: item[0]
    )[1]

    history = league_team_history.get(league_name, {})

    score = estimate_score(
        history.get(home_team, []),
        history.get(away_team, [])
    )

    return {
        "outcome": outcome,
        "home_prob": home_prob,
        "draw_prob": draw_prob,
        "away_prob": away_prob,
        "score": score
    }


# ==================================================
# MAIN DASHBOARD
# ==================================================


# Show login before opening the dashboard
if not st.session_state.get("authenticated", False):
    authentication_screen()
    st.stop()

# Dashboard starts here
st.title("⚽ Football Predictor")

st.write("### Multi-League Match Prediction Dashboard")

st.caption(
    "Machine learning predictions based on historical "
    "football results."
)

st.divider()

league = st.selectbox(
    "Choose a league",
    leagues
)

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
    away_options = [
        team for team in available_teams
        if team != home_team
    ]

    away_team = st.selectbox(
        "✈️ Away team",
        away_options,
        index=0
    )


# ==================================================
# MANUAL MATCH PREDICTION
# ==================================================

st.divider()

if home_team == away_team:
    st.warning("Choose two different teams.")

elif st.button("🔮 Predict Match", key="manual_prediction"):

    try:
        result = predict_outcome(
            league,
            home_team,
            away_team
        )

        st.subheader("📊 Match Prediction")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric(
                "🏠 Home Win",
                f"{result['home_prob']:.1%}"
            )
            st.progress(
                min(max(result["home_prob"], 0.0), 1.0)
            )

        with p2:
            st.metric(
                "🤝 Draw",
                f"{result['draw_prob']:.1%}"
            )
            st.progress(
                min(max(result["draw_prob"], 0.0), 1.0)
            )

        with p3:
            st.metric(
                "✈️ Away Win",
                f"{result['away_prob']:.1%}"
            )
            st.progress(
                min(max(result["away_prob"], 0.0), 1.0)
            )
            
        st.divider()

        st.subheader("🎯 Prediction Summary")

        st.success(
            f"Most likely outcome: **{result['outcome']}**"
        )

        probabilities = [
            result["home_prob"],
            result["draw_prob"],
            result["away_prob"]
        ]

        probability_gap = max(probabilities) - sorted(probabilities)[-2]

        if probability_gap < 0.10:
            st.warning(
                "⚠️ This match looks difficult to predict. "
                "The leading outcomes have similar probabilities."
            )
        else:
            st.info(
                "The model has a clearer leading outcome, "
                "but the prediction is not guaranteed."
            )


        st.success(
            f"Most likely outcome: {result['outcome']}"
        )

        st.subheader("⚽ Estimated Correct Score")

        if result["score"] is not None:
            st.metric(
                "Predicted score",
                f"{home_team} {result['score']} {away_team}"
            )
        else:
            st.info(
                "Insufficient historical data to estimate the score."
            )

        st.caption(
            "The score is estimated from recent goals. "
            "It is not a separately trained score prediction."
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
    "FOOTBALL PREDICTOR | Python, Streamlit and Machine Learning"
)


# ==================================================
# UPCOMING FIXTURES
# ==================================================

st.divider()

st.header("📅 Upcoming Fixtures")

st.write(
    "Explore scheduled matches and view automatic predictions."
)

API_TOKEN = st.secrets.get("FOOTBALL_DATA_TOKEN", "")

COMPETITION_CODES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "Ligue 1": "FL1"
}


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
        format_func=lambda number: f"{number} days",
        key="fixture_days"
    )

    if st.button("🔄 Load Upcoming Fixtures"):

        start = date.today()
        end = start + timedelta(days=days_ahead)

        if fixture_league == "All leagues":
            leagues_to_fetch = list(COMPETITION_CODES.items())
        else:
            leagues_to_fetch = [
                (
                    fixture_league,
                    COMPETITION_CODES[fixture_league]
                )
            ]

        fixtures = []
        failed_leagues = []

        with st.spinner("Fetching upcoming fixtures..."):

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
                            "SCHEDULED",
                            "TIMED"
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

                except requests.HTTPError as error:

                    status_code = (
                        error.response.status_code
                        if error.response is not None
                        else "unknown"
                    )

                    failed_leagues.append(
                        f"{league_name}: HTTP {status_code}"
                    )

                except requests.RequestException:

                    failed_leagues.append(
                        f"{league_name}: connection problem"
                    )

                except Exception as error:

                    failed_leagues.append(
                        f"{league_name}: {error}"
                    )

        if fixtures:

            fixture_df = pd.DataFrame(fixtures)

            fixture_df["Date (UTC)"] = pd.to_datetime(
                fixture_df["Date (UTC)"],
                utc=True,
                errors="coerce"
            )

            fixture_df = fixture_df.sort_values("Date (UTC)")

            st.success(
                f"Found {len(fixture_df)} upcoming matches."
            )

            st.dataframe(
                fixture_df,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------
            # AUTOMATIC PREDICTIONS
            # ------------------------------------------

            st.subheader("🔮 Predictions for Upcoming Fixtures")

            prediction_rows = []

            for _, match_row in fixture_df.iterrows():

                league_name = match_row["League"]
                home_name = match_row["Home Team"]
                away_name = match_row["Away Team"]

                home_key = find_team(
                    home_name,
                    league_name
                )

                away_key = find_team(
                    away_name,
                    league_name
                )

                if not home_key or not away_key:

                    missing = []

                    if not home_key:
                        missing.append(home_name)

                    if not away_key:
                        missing.append(away_name)

                    prediction_rows.append({
                        "Date (UTC)": match_row["Date (UTC)"],
                        "League": league_name,
                        "Match": f"{home_name} vs {away_name}",
                        "Prediction": (
                            "Historical data missing: "
                            + ", ".join(missing)
                        ),
                        "Home Win": "—",
                        "Draw": "—",
                        "Away Win": "—",
                        "Estimated Score": "—"
                    })

                    continue

                try:

                    result = predict_outcome(
                        league_name,
                        home_key,
                        away_key
                    )

                    prediction_rows.append({
                        "Date (UTC)": match_row["Date (UTC)"],
                        "League": league_name,
                        "Match": f"{home_name} vs {away_name}",
                        "Prediction": result["outcome"],
                        "Home Win": f"{result['home_prob']:.1%}",
                        "Draw": f"{result['draw_prob']:.1%}",
                        "Away Win": f"{result['away_prob']:.1%}",
                        "Estimated Score": (
                            result["score"]
                            if result["score"] is not None
                            else "Insufficient historical data"
                        )
                    })

                except Exception as error:

                    prediction_rows.append({
                        "Date (UTC)": match_row["Date (UTC)"],
                        "League": league_name,
                        "Match": f"{home_name} vs {away_name}",
                        "Prediction": f"Prediction error: {error}",
                        "Home Win": "—",
                        "Draw": "—",
                        "Away Win": "—",
                        "Estimated Score": "—"
                    })

            predictions_df = pd.DataFrame(prediction_rows)

            st.dataframe(
                predictions_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Win probabilities come from the existing machine "
                "learning model. Score estimates use recent historical "
                "goals and are not guaranteed results."
            )

        else:

            st.info(
                "No upcoming fixtures were returned for "
                "the selected period."
            )

        if failed_leagues:

            st.warning(
                "Some leagues could not be loaded. "
                "Check your API plan and token."
            )

            for problem in failed_leagues:
                st.caption(problem)


st.divider()

st.caption(
    "Fixture dates are shown in UTC. "
    "Fixture availability depends on your API provider."
)

# ---------- USER REGISTRATION AND LOGIN ----------

def init_user_database():
    conn = sqlite3.connect("users.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, email, password):
    conn = sqlite3.connect("users.db")
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, hash_password(password))
        )
        conn.commit()
        return True, "Registration successful! You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."
    finally:
        conn.close()


def verify_user(username, password):
    conn = sqlite3.connect("users.db")
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()

    return row is not None and row[0] == hash_password(password)


def authentication_screen():
    init_user_database()

    st.title("⚽ Football Predictor")
    st.subheader("Welcome")

    choice = st.radio(
        "Choose an option",
        ["Login", "Register"],
        horizontal=True
    )

    if choice == "Register":
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account")

        if submitted:
            if not username.strip() or not email.strip() or not password:
                st.error("Please fill in all fields.")
            elif "@" not in email or "." not in email:
                st.error("Enter a valid email address.")
            elif len(password) < 8:
                st.error("Use a password with at least 8 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(
                    username.strip(), email.strip().lower(), password
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)

    else:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            if verify_user(username.strip(), password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")


