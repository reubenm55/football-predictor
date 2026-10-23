import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re
import unicodedata
import hashlib
import sqlite3
import base64

from datetime import date, timedelta, datetime, timezone
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
# USER REGISTRATION AND LOGIN
# ==================================================

import os

DB_PATH = "users.db"


def init_user_database():
    """Create the user table if it does not already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        conn.commit()


def hash_password(password, salt=None):
    """Hash passwords with PBKDF2 and a random salt."""
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex() + "$" + digest.hex()


def verify_password(password, stored_hash):
    """Verify a PBKDF2 password hash; support older SHA-256 hashes too."""
    if "$" not in stored_hash:
        return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored_hash
    salt_hex, expected_digest = stored_hash.split("$", 1)
    salt = bytes.fromhex(salt_hex)
    actual_digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()
    return hashlib.compare_digest(actual_digest, expected_digest) if hasattr(hashlib, "compare_digest") else __import__("hmac").compare_digest(actual_digest, expected_digest)


def register_user(username, email, password):
    init_user_database()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, hash_password(password))
            )
            conn.commit()
        return True, "Account created successfully. You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered. Please choose another."
    except sqlite3.Error:
        return False, "Could not save the account. Please try again."


def verify_user(username, password):
    init_user_database()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                (username,)
            ).fetchone()
        return row is not None and verify_password(password, row[0])
    except sqlite3.Error:
        return False


def ensure_billing_tables():
    """Create prediction and subscription tables without replacing user accounts."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            match_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            phone TEXT NOT NULL,
            amount INTEGER NOT NULL,
            checkout_request_id TEXT,
            merchant_request_id TEXT,
            status TEXT NOT NULL DEFAULT 'PENDING',
            created_at TEXT NOT NULL,
            confirmed_at TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS subscriptions (
            username TEXT PRIMARY KEY,
            expires_at TEXT NOT NULL,
            payment_id INTEGER,
            FOREIGN KEY(payment_id) REFERENCES payments(id)
        )""")
        conn.commit()


def local_now():
    # East Africa Time, independent of the server's timezone.
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Africa/Nairobi"))


def user_subscription_active(username):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT expires_at FROM subscriptions WHERE username=?", (username,)).fetchone()
    if not row:
        return False
    try:
        expiry = datetime.fromisoformat(row[0])
        return expiry > local_now()
    except (TypeError, ValueError):
        return False


def prediction_count(username, today_only=False):
    with sqlite3.connect(DB_PATH) as conn:
        if today_only:
            day = local_now().date().isoformat()
            row = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE username=? AND substr(created_at,1,10)=?",
                (username, day)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM predictions WHERE username=?", (username,)).fetchone()
    return int(row[0] if row else 0)


def record_prediction(username, match_name, outcome):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions(username,match_name,prediction,created_at) VALUES(?,?,?,?)",
            (username, match_name, outcome, local_now().isoformat(timespec="seconds"))
        )
        conn.commit()


def mpesa_stk_push(username, phone, amount):
    """Start a Daraja STK push. Credentials are read only from Streamlit Secrets."""
    required = ["MPESA_CONSUMER_KEY", "MPESA_CONSUMER_SECRET", "MPESA_SHORTCODE", "MPESA_PASSKEY", "MPESA_CALLBACK_URL"]
    missing = [key for key in required if not st.secrets.get(key, "")]
    if missing:
        return False, "M-Pesa is not configured yet. Missing Streamlit Secrets: " + ", ".join(missing)

    environment = str(st.secrets.get("MPESA_ENVIRONMENT", "sandbox")).lower()
    if environment == "production":
        base_url = "https://api.safaricom.co.ke"
    else:
        base_url = "https://sandbox.safaricom.co.ke"

    consumer_key = st.secrets["MPESA_CONSUMER_KEY"]
    consumer_secret = st.secrets["MPESA_CONSUMER_SECRET"]
    try:
        token_response = requests.get(
            base_url + "/oauth/v1/generate?grant_type=client_credentials",
            auth=(consumer_key, consumer_secret), timeout=20
        )
        token_response.raise_for_status()
        access_token = token_response.json()["access_token"]

        shortcode = str(st.secrets["MPESA_SHORTCODE"])
        passkey = str(st.secrets["MPESA_PASSKEY"])
        timestamp = local_now().strftime("%Y%m%d%H%M%S")
        password = base64.b64encode((shortcode + passkey + timestamp).encode()).decode()
        normalized_phone = re.sub(r"\D", "", phone)
        if normalized_phone.startswith("0") and len(normalized_phone) == 10:
            normalized_phone = "254" + normalized_phone[1:]
        elif normalized_phone.startswith("7") or normalized_phone.startswith("1"):
            normalized_phone = "254" + normalized_phone
        if not (normalized_phone.startswith("254") and len(normalized_phone) == 12):
            return False, "Enter a valid Kenyan mobile number, e.g. 0712345678."

        payload = {
            "BusinessShortCode": shortcode,
            "Password": password,
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": int(amount),
            "PartyA": normalized_phone,
            "PartyB": shortcode,
            "PhoneNumber": normalized_phone,
            "CallBackURL": str(st.secrets["MPESA_CALLBACK_URL"]),
            "AccountReference": "FOOTBALLPREDICTOR",
            "TransactionDesc": "Football Predictor subscription"
        }
        response = requests.post(
            base_url + "/mpesa/stkpush/v1/processrequest",
            json=payload,
            headers={"Authorization": "Bearer " + access_token}, timeout=25
        )
        response.raise_for_status()
        data = response.json()
        if data.get("ResponseCode") != "0":
            return False, data.get("errorMessage", data.get("ResponseDescription", "M-Pesa did not accept the request."))
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""INSERT INTO payments(username,phone,amount,checkout_request_id,merchant_request_id,status,created_at)
                VALUES(?,?,?,?,?,'PENDING',?)""",
                (username, normalized_phone, int(amount), data.get("CheckoutRequestID"), data.get("MerchantRequestID"), local_now().isoformat(timespec="seconds")))
            conn.commit()
        return True, "STK push sent. Enter your M-Pesa PIN on your phone. Access stays pending until the payment is confirmed."
    except requests.RequestException as exc:
        return False, f"Could not contact M-Pesa: {exc}"
    except (KeyError, ValueError) as exc:
        return False, f"M-Pesa response/configuration error: {exc}"


def activate_subscription(username, payment_id):
    now = local_now()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT status FROM payments WHERE id=? AND username=?", (payment_id, username)).fetchone()
        if not row:
            return False, "Payment record not found."
        if row[0] == "CONFIRMED":
            return False, "This payment is already confirmed."
        current = conn.execute("SELECT expires_at FROM subscriptions WHERE username=?", (username,)).fetchone()
        try:
            current_expiry = datetime.fromisoformat(current[0]) if current else now
        except ValueError:
            current_expiry = now
        start = max(now, current_expiry)
        expiry = start + timedelta(days=30)
        conn.execute("UPDATE payments SET status='CONFIRMED', confirmed_at=? WHERE id=?", (now.isoformat(timespec="seconds"), payment_id))
        conn.execute("INSERT INTO subscriptions(username,expires_at,payment_id) VALUES(?,?,?) ON CONFLICT(username) DO UPDATE SET expires_at=excluded.expires_at,payment_id=excluded.payment_id", (username, expiry.isoformat(timespec="seconds"), payment_id))
        conn.commit()
    return True, f"Subscription active until {expiry.strftime('%d %b %Y %H:%M')} EAT."


def authentication_screen():
    st.markdown("<div style='text-align:center;font-size:3rem'>⚽</div>", unsafe_allow_html=True)
    st.title("Football Predictor")
    st.markdown("### Welcome — sign in or create your account")

    choice = st.radio("Choose an option", ["Login", "Register"], horizontal=True)

    if choice == "Register":
        with st.form("register_form", clear_on_submit=False):
            username = st.text_input("Username", max_chars=40).strip()
            email = st.text_input("Email address", max_chars=254).strip().lower()
            password = st.text_input("Password (at least 8 characters)", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account")

        if submitted:
            if not username or not email or not password:
                st.error("Please fill in all fields.")
            elif "@" not in email or "." not in email.rsplit("@", 1)[-1]:
                st.error("Please enter a valid email address.")
            elif len(password) < 8:
                st.error("Your password must contain at least 8 characters.")
            elif password != confirm:
                st.error("The passwords do not match.")
            else:
                success, message = register_user(username, email, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    else:
        with st.form("login_form"):
            username = st.text_input("Username", max_chars=40).strip()
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            if verify_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Incorrect username or password.")


init_user_database()
ensure_billing_tables()

# ==================================================
# MAIN DASHBOARD
# ==================================================


# Show login before opening the dashboard
if not st.session_state.get("authenticated", False):
    authentication_screen()
    st.stop()

# Dashboard starts here
st.title("⚽ Football Predictor")

user_col, logout_col = st.columns([5, 1])
with user_col:
    st.caption(f"Signed in as: {st.session_state.get('username', 'User')}")
with logout_col:
    if st.button("Log out", key="logout_button"):
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        st.rerun()

# ==================================================
# USER METRICS, SUBSCRIPTION AND M-PESA
# ==================================================
current_user = st.session_state.get("username", "User")
subscription_active = user_subscription_active(current_user)
free_daily_limit = int(st.secrets.get("FREE_DAILY_PREDICTIONS", 3))
used_today = prediction_count(current_user, today_only=True)
remaining_today = max(0, free_daily_limit - used_today)

with sqlite3.connect(DB_PATH) as conn:
    registered_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    all_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    my_total_predictions = prediction_count(current_user)
    pending_payment_rows = conn.execute("SELECT id, username, phone, amount, created_at FROM payments WHERE status='PENDING' ORDER BY id DESC").fetchall()

metric1, metric2, metric3 = st.columns(3)
with metric1:
    st.metric("Your predictions today", used_today)
with metric2:
    st.metric("Your total predictions", my_total_predictions)
with metric3:
    st.metric("Subscription", "Active" if subscription_active else "Free plan")

if subscription_active:
    st.success("Your premium subscription is active.")
else:
    st.info(f"Free plan: {remaining_today} of {free_daily_limit} predictions remaining today. Premium costs KSh {int(st.secrets.get('SUBSCRIPTION_PRICE_KES', 300))} for 30 days.")

with st.expander("💳 Upgrade with M-Pesa", expanded=not subscription_active):
    price_kes = int(st.secrets.get("SUBSCRIPTION_PRICE_KES", 300))
    st.write(f"**30-day subscription: KSh {price_kes:,}**")
    st.caption("Use your Kenyan M-Pesa number. A payment request will be sent to your phone.")
    with st.form("mpesa_subscription_form"):
        mpesa_phone = st.text_input("M-Pesa phone number", placeholder="0712345678")
        pay_clicked = st.form_submit_button(f"Pay KSh {price_kes:,} with M-Pesa")
    if pay_clicked:
        if not mpesa_phone.strip():
            st.error("Enter the phone number to receive the M-Pesa prompt.")
        else:
            with st.spinner("Requesting M-Pesa payment..."):
                ok, message = mpesa_stk_push(current_user, mpesa_phone.strip(), price_kes)
            (st.success if ok else st.error)(message)
    with sqlite3.connect(DB_PATH) as conn:
        my_payments = conn.execute("SELECT id, amount, status, created_at FROM payments WHERE username=? ORDER BY id DESC LIMIT 10", (current_user,)).fetchall()
    if my_payments:
        st.write("**Your recent payment requests**")
        st.dataframe(pd.DataFrame(my_payments, columns=["Payment ID", "Amount (KES)", "Status", "Requested at"]), use_container_width=True, hide_index=True)

admin_username = str(st.secrets.get("ADMIN_USERNAME", ""))
if admin_username and current_user == admin_username:
    with st.expander("🛠️ Admin dashboard — users, predictions and payment review"):
        a1, a2, a3 = st.columns(3)
        a1.metric("Registered users", int(registered_users))
        a2.metric("All-time predictions", int(all_predictions))
        with sqlite3.connect(DB_PATH) as conn:
            todays_all = conn.execute("SELECT COUNT(*) FROM predictions WHERE substr(created_at,1,10)=?", (local_now().date().isoformat(),)).fetchone()[0]
        a3.metric("Predictions today", int(todays_all))
        with sqlite3.connect(DB_PATH) as conn:
            user_stats = conn.execute("SELECT u.username, u.email, COUNT(p.id) AS total_predictions, SUM(CASE WHEN substr(p.created_at,1,10)=? THEN 1 ELSE 0 END) AS predictions_today FROM users u LEFT JOIN predictions p ON p.username=u.username GROUP BY u.username,u.email ORDER BY total_predictions DESC", (local_now().date().isoformat(),)).fetchall()
        st.write("**User statistics**")
        st.dataframe(pd.DataFrame(user_stats, columns=["Username", "Email", "Total predictions", "Predictions today"]).fillna(0), use_container_width=True, hide_index=True)
        if pending_payment_rows:
            st.write("**Pending payments** — verify payment in your M-Pesa business account before confirming.")
            st.dataframe(pd.DataFrame(pending_payment_rows, columns=["Payment ID", "Username", "Phone", "Amount (KES)", "Requested at"]), use_container_width=True, hide_index=True)
            with st.form("confirm_payment_form"):
                payment_id_to_confirm = st.number_input("Payment ID to confirm", min_value=1, step=1)
                confirm_payment = st.form_submit_button("Confirm verified payment and activate 30 days")
            if confirm_payment:
                # Admin must independently verify receipt; do not trust the user's claim.
                with sqlite3.connect(DB_PATH) as conn:
                    target = conn.execute("SELECT username FROM payments WHERE id=? AND status='PENDING'", (int(payment_id_to_confirm),)).fetchone()
                if not target:
                    st.error("No pending payment with that ID.")
                else:
                    ok, message = activate_subscription(target[0], int(payment_id_to_confirm))
                    (st.success if ok else st.error)(message)
                    if ok:
                        st.rerun()
        else:
            st.caption("No pending payments.")

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

elif not subscription_active and used_today >= free_daily_limit:
    st.warning("You have used today's free predictions. Upgrade with M-Pesa to continue.")
elif st.button("🔮 Predict Match", key="manual_prediction"):

    try:
        result = predict_outcome(
            league,
            home_team,
            away_team
        )
        record_prediction(current_user, f"{league}: {home_team} vs {away_team}", result["outcome"])

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
            if not subscription_active and prediction_count(current_user, today_only=True) >= free_daily_limit:
                st.warning("Your daily free prediction limit has been reached. Upgrade to run predictions for fixtures.")
                st.stop()

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

                    if not subscription_active and prediction_count(current_user, today_only=True) >= free_daily_limit:
                        prediction_rows.append({
                            "Date (UTC)": match_row["Date (UTC)"],
                            "League": league_name,
                            "Match": f"{home_name} vs {away_name}",
                            "Prediction": "Daily free limit reached — upgrade for more",
                            "Home Win": "—", "Draw": "—", "Away Win": "—", "Estimated Score": "—"
                        })
                        continue
                    result = predict_outcome(
                        league_name,
                        home_key,
                        away_key
                    )
                    record_prediction(current_user, f"{league_name}: {home_name} vs {away_name}", result["outcome"])

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