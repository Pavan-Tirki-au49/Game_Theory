# auth.py — Clean Login/Register Toggle (No Tabs, No Sliding, No HTML)

import hashlib
from datetime import datetime
import streamlit as st

from utils import DB_PATH, get_conn


# ======================================================
# Helpers (DB / Password)
# ======================================================

def ensure_user_table():
    """Create users table if it does not exist."""
    with get_conn(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(username: str, password: str):
    username = username.strip()

    if not username or not password:
        return False, "Username and password cannot be empty."

    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    password_hash = _hash_password(password)

    with get_conn(DB_PATH) as conn:
        cur = conn.execute("SELECT id FROM users WHERE username=?", (username,))
        if cur.fetchone() is not None:
            return False, "Username already taken."

        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.utcnow().isoformat())
        )
        conn.commit()

    return True, "Registration successful! You can now log in."


def authenticate_user(username: str, password: str):
    username = username.strip()
    if not username or not password:
        return False, "Enter both username and password."

    password_hash = _hash_password(password)

    with get_conn(DB_PATH) as conn:
        cur = conn.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()

    if row is None:
        return False, "User not found. Please register."

    if row[0] != password_hash:
        return False, "Incorrect password."

    return True, "Login successful!"


def get_logged_in_username():
    return st.session_state.get("username")


def handle_logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None


# ======================================================
# Authentication UI (Login / Register Toggle)
# ======================================================

def render_auth_gate():
    """Login page with a small Register/Back-to-login link."""

    # Default mode = login
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    st.title("FashionChain")
    st.caption("Secure access to your Game Theory dashboard.")
    st.markdown("---")

    left_col, right_col = st.columns([1, 1])

    # ---------------- LEFT SIDE ----------------
    with left_col:
        st.subheader("About FashionChain")
        st.write(
            """
            • Game Theory–based supplier & retailer analytics  
            • Pure & Mixed strategy simulations  
            • Results stored securely in your local database  
            """
        )
        st.write("Only logged-in users can access the full dashboard.")

    # ---------------- RIGHT SIDE ----------------
    with right_col:

        # ---------------------- LOGIN ----------------------
        if st.session_state["auth_mode"] == "login":
            st.subheader("Login")

            with st.form("login_form"):
                username = st.text_input("Username")
                show_pass = st.checkbox("Show password", key="login_show_pass")
                password = st.text_input(
                    "Password",
                    type="default" if show_pass else "password"
                )
                submit = st.form_submit_button("Login")

            if submit:
                success, msg = authenticate_user(username, password)
                if success:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

            st.write("")
            if st.button("New user? Register →", use_container_width=False):
                st.session_state["auth_mode"] = "register"
                st.rerun()

        # ---------------------- REGISTER ----------------------
        else:
            st.subheader("Register")

            with st.form("reg_form"):
                new_username = st.text_input("Choose username")
                show_pass2 = st.checkbox("Show password", key="register_show_pass")
                new_password = st.text_input(
                    "Choose password (min 4 chars)",
                    type="default" if show_pass2 else "password"
                )
                reg_submit = st.form_submit_button("Create Account")

            if reg_submit:
                success, msg = register_user(new_username, new_password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

            st.write("")
            if st.button("← Already have an account? Login", use_container_width=False):
                st.session_state["auth_mode"] = "login"
                st.rerun()
