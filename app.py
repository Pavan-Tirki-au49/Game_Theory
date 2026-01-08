# app.py — Modern Retail Theme (Fashion Supplier Game Theory)
import streamlit as st
from pathlib import Path
import pandas as pd

# ---------------------------
# Streamlit page config FIRST
# ---------------------------
st.set_page_config(page_title="FashionChain — Game Theory", layout="wide")

# Apply Modern Retail Theme
from theme import apply_modern_retail_theme

# Default logo (from your screenshot)
_default_logo_path = "/mnt/data/Screenshot 2025-11-20 234737.png"
apply_modern_retail_theme(logo_path=_default_logo_path)

# ---------------------------
# Import utilities
# ---------------------------
from utils import (
    load_data_file, DB_PATH, get_conn,
    ensure_results_table, ensure_losses_table,
    clear_results_table, clear_losses_table,
    ensure_generated_strategies_table, fetch_all_generated_strategies,
    ensure_quarantine_table, fetch_quarantine_generated_strategies
)

# ---------------------------
# Import auth module (NEW)
# ---------------------------
from auth import (
    ensure_user_table,
    render_auth_gate,
    get_logged_in_username,
    handle_logout,
)

# ---------------------------
# Import feature modules
# ---------------------------
import pure_strategy
import mixed_strategy

# ======================================================
# Authentication Gate (NEW)
# ======================================================

# Make sure the users table exists
ensure_user_table()

# Initialize auth-related session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

# If not logged in → show Login / Register and stop
if not st.session_state["logged_in"]:
    render_auth_gate()
    st.stop()

# If logged in, show who is logged in + Logout in sidebar header area
username = get_logged_in_username()

# ======================================================
# Sidebar Navigation
# ======================================================
st.sidebar.title("🧭 Navigation Menu")

# Show user info + logout button (NEW)
if username:
    st.sidebar.markdown(f"👤 **User:** `{username}`")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        handle_logout()
        st.rerun()
else:
    st.sidebar.warning("No user logged in (this should not happen).")

section = st.sidebar.radio(
    "Choose section:",
    ["Home", "Dashboard", "Welcome to the Game", "Results", "Settings"]
)

st.sidebar.subheader("📂 Data Source")
uploaded = st.sidebar.file_uploader(
    "Upload Excel/CSV (Supplier_Name, Supplier_Profit)",
    type=["xlsx", "csv"]
)

# ======================================================
# Load dataset (uploaded or fallback) + New Member Handling
# ======================================================
# Base dataset from file
if uploaded is not None:
    base_df = load_data_file(uploaded)
else:
    default_path = Path("data_samples_with_real_supplier_names.xlsx")
    base_df = load_data_file(default_path if default_path.exists() else None)

# If load_data_file returned None, make an empty frame with standard columns
if base_df is None:
    base_df = pd.DataFrame(columns=["Supplier_Name", "Supplier_Profit"])

# Ensure minimal expected columns exist
if "Supplier_Name" not in base_df.columns:
    base_df["Supplier_Name"] = ""
if "Supplier_Profit" not in base_df.columns:
    base_df["Supplier_Profit"] = 0.0

# Session-based "new members" (extra rows added from 'Welcome to the Game' page)
if "new_members_df" not in st.session_state:
    st.session_state["new_members_df"] = pd.DataFrame(columns=base_df.columns)

# Combine base dataset + any new members
data_df = pd.concat([base_df, st.session_state["new_members_df"]], ignore_index=True)

st.sidebar.caption(f"✅ Loaded rows (including new members): {len(data_df)}")

# ======================================================
# Page: Home
# ======================================================
if section == "Home":
    st.title("🎯 Welcome to FashionChain")
    st.markdown("""
    Enter FashionChain, a strategic analytics environment where Game Theory meets retail supply-chain performance—helping businesses evaluate scenarios, anticipate partner decisions, and refine commercial strategy.
    """)
    st.image(
        "https://tse4.mm.bing.net/th/id/OIP.RgBHENWD8655MzqmyRS_gwHaHa?pid=Api&P=0&h=180",
        width=180
    )
    st.info("Use the **Dashboard** to generate payoff matrices and run simulations. Results are stored in SQLite automatically.")

# ======================================================
# Page: Dashboard
# ======================================================
elif section == "Dashboard":
    st.title("📊 Supplier and Retailer Strategy Dashboard")

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        # handle case where Supplier_Name might be empty
        if "Supplier_Name" in data_df.columns and not data_df.empty:
            st.metric("Suppliers in Dataset", len(data_df["Supplier_Name"].dropna().unique()))
        else:
            st.metric("Suppliers in Dataset", 0)
    with c2:
        if "Supplier_Profit" in data_df.columns and not data_df.empty:
            st.metric("Average Profit", f"{data_df['Supplier_Profit'].mean():.2f}")
        else:
            st.metric("Average Profit", "0.00")
    with c3:
        st.metric("Database File", str(DB_PATH.name if hasattr(DB_PATH, "name") else DB_PATH))

    st.divider()
    st.markdown("### 🎮 Choose Strategy Mode")

    # Choose Pure or Mixed strategy
    left, right = st.columns(2)
    if "selected_strategy" not in st.session_state:
        st.session_state["selected_strategy"] = None

    with left:
        if st.button("🟢 Pure Strategy", use_container_width=True):
            st.session_state["selected_strategy"] = "pure"
    with right:
        if st.button("🔵 Mixed Strategy", use_container_width=True):
            st.session_state["selected_strategy"] = "mixed"

    st.divider()

    # Show dataset sample
    st.subheader("📑 Data Preview")
    max_rows = len(data_df) if len(data_df) > 0 else 5
    rows_to_show = st.slider(
        "Rows to display",
        min_value=1,
        max_value=min(200, max_rows),
        value=min(15, max_rows)
    )

    st.dataframe(
        data_df.head(rows_to_show).style.map(
            lambda v: "color:red" if isinstance(v, (int, float)) and v < 0 else ""
        ),
        width="stretch"
    )

    # Conditional display
    if st.session_state.get("selected_strategy") == "pure":
        choice = st.radio("Select Pure Strategy Mode:", ["Payoff Matrix", "Simulation"], horizontal=True)
        if choice == "Payoff Matrix":
            pure_strategy.render_payoff_matrix_ui(data_df, DB_PATH)
        else:
            pure_strategy.render_simulation_ui(data_df, DB_PATH)

    elif st.session_state.get("selected_strategy") == "mixed":
        mixed_strategy.render_mixed_strategy_ui(data_df)

# ======================================================
# Page: Welcome to the Game (New Member Join)
# ======================================================
elif section == "Welcome to the Game":
    st.title("🤝 Welcome to the Game — Add New Supplier")

    st.markdown("""
    New suppliers can **join the game** here.  
    When you add a new member, they are appended to the **current dataset** and can participate in all simulations.
    """)

    with st.form("new_member_form"):
        name = st.text_input("Supplier Name", placeholder="e.g., Alpha Trends Pvt Ltd")
        profit = st.number_input(
            "Initial Supplier_Profit value for this member (can be 0)",
            value=0.0,
            step=1.0,
            format="%.2f"
        )
        submitted = st.form_submit_button("➕ Add Member to Dataset")

    if submitted:
        if not name.strip():
            st.error("Please enter a valid supplier name.")
        else:
            # Build a new row matching base_df columns
            row_dict = {}
            for col in base_df.columns:
                if col == "Supplier_Name":
                    row_dict[col] = name.strip()
                elif col == "Supplier_Profit":
                    row_dict[col] = float(profit)
                else:
                    # Other columns (if any) default to None
                    row_dict[col] = None

            new_row_df = pd.DataFrame([row_dict])

            # Append to session-based "new members" table
            st.session_state["new_members_df"] = pd.concat(
                [st.session_state["new_members_df"], new_row_df],
                ignore_index=True
            )

            st.success(f"🎉 New supplier **{name}** added to the dataset!")

    # Show a small preview of new members + final combined dataset info
    st.divider()
    st.subheader("🆕 New Members Added This Session")
    if st.session_state["new_members_df"].empty:
        st.info("No new members added yet.")
    else:
        st.dataframe(st.session_state["new_members_df"], width="stretch")

    st.subheader("📦 Combined Dataset (File + New Members)")
    combined_df = pd.concat([base_df, st.session_state["new_members_df"]], ignore_index=True)
    st.write(f"Total rows (including new members): **{len(combined_df)}**")
    st.dataframe(combined_df.head(50), width="stretch")

# ======================================================
# Page: Results
# ======================================================
elif section == "Results":
    st.title("📈 Simulation Results")
    pure_strategy.render_results_ui(DB_PATH)

# ======================================================
# Page: Settings
# ======================================================
elif section == "Settings":
    st.title("⚙️ Settings & Database Tools")

    with st.expander("🗄️ Manage Database Tables"):
        if st.button("Create/Ensure Results Table"):
            with get_conn(DB_PATH) as conn:
                ensure_results_table(conn)
            st.success("✅ Results table ensured.")

        if st.button("Create/Ensure Losses Table"):
            with get_conn(DB_PATH) as conn:
                ensure_losses_table(conn)
            st.success("✅ Losses table ensured.")

        if st.button("Create/Ensure Generated Strategies Table"):
            with get_conn(DB_PATH) as conn:
                ensure_generated_strategies_table(conn)
            st.success("✅ generated_strategies table ensured.")

        if st.button("Create/Ensure Quarantine Table"):
            with get_conn(DB_PATH) as conn:
                ensure_quarantine_table(conn)
            st.success("✅ quarantine_generated_strategies table ensured.")

        if st.button("Clear Results Table"):
            with get_conn(DB_PATH) as conn:
                clear_results_table(conn)
            st.warning("🧹 All results cleared.")

        if st.button("Clear Losses Table"):
            with get_conn(DB_PATH) as conn:
                clear_losses_table(conn)
            st.warning("🧹 All losses cleared.")

        if st.button("Clear Generated Strategies Table"):
            with get_conn(DB_PATH) as conn:
                ensure_generated_strategies_table(conn)
                conn.execute("DELETE FROM generated_strategies;")
                conn.commit()
            st.warning("🧹 All generated strategies cleared.")

        if st.button("Clear Quarantine Table"):
            with get_conn(DB_PATH) as conn:
                ensure_quarantine_table(conn)
                conn.execute("DELETE FROM quarantine_generated_strategies;")
                conn.commit()
            st.warning("🧹 All quarantined strategies cleared.")

    st.markdown("""
    Upload your supplier dataset in the sidebar  
    to analyze **Fashion Supplier Profit Strategies** using Game Theory.
    """)
