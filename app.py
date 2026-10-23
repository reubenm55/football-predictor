



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
