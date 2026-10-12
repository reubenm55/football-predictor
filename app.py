
import streamlit as st

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽"
)

st.title("⚽ Football Match Predictor")
st.write("Welcome! Let's build your football prediction system.")

teams = [
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Tottenham"
]

home_team = st.selectbox("Select the home team", teams)
away_team = st.selectbox(
    "Select the away team",
    [team for team in teams if team != home_team]
)

if st.button("Predict Match"):
    st.info(
        f"You selected {home_team} vs {away_team}. "
        "The machine-learning prediction model still needs to be connected."
    )
