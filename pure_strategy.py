# pure_strategy.py — Final Fixed Version (Clothing Supplier Theme) + SVM classifier UI
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from typing import List, Dict, Any

# Developer helper: local uploaded file path (from workspace).
# Use this for quick testing if you want to point to a local file.
DEFAULT_UPLOADED_FILE = "/mnt/data/Screenshot 2025-11-20 234737.png"

# -------------------------------------------------------------------
# Import shared utilities (with optional ML helpers)
# -------------------------------------------------------------------
try:
    from utils import (
        generate_payoff_matrix,
        find_saddle_points,  # NOTE: kept for compatibility but we override with our own logic below
        get_conn,
        save_results_df,
        save_losses,
        fetch_all_results,
        fetch_all_losses,
        ensure_generated_strategies_table,
        fetch_all_generated_strategies,
        ensure_quarantine_table,
        fetch_all_quarantine,
        move_rows_to_quarantine,
        DB_PATH,
        A_LABELS,
        B_LABELS,
        highlight_losses,
        # ML helpers & flags
        train_strategy_svm_from_dataframe,
        predict_strategy_labels,
        load_strategy_model,
        auto_label_from_texts,
        SKLEARN_AVAILABLE,
    )
    UTIL_ML = bool(SKLEARN_AVAILABLE)
except Exception:
    from utils import (
        generate_payoff_matrix,
        find_saddle_points,
        get_conn,
        save_results_df,
        save_losses,
        fetch_all_results,
        fetch_all_losses,
        ensure_generated_strategies_table,
        fetch_all_generated_strategies,
        ensure_quarantine_table,
        fetch_all_quarantine,
        move_rows_to_quarantine,
        DB_PATH,
        A_LABELS,
        B_LABELS,
        highlight_losses,
    )
    UTIL_ML = False

# ======================================================
# Strategy Labels (for UI explanation)
# ======================================================
SUPPLIER1_STRATEGIES = [
    "Interested in business",
    "Not interested in business",
    "50-50 choice",
]
SUPPLIER2_STRATEGIES = [
    "Interested in business",
    "Not interested in business",
    "50-50 choice",
]

# ======================================================
# Custom saddle-point logic (Row-Min / Col-Max rule)
# ======================================================
def find_saddle_points_rowmin_colmax(matrix: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Compute saddle points strictly using the classical rule:

      - Compute row minima for each row.
      - Compute column maxima for each column.
      - Let V_r = max(row minima), V_c = min(column maxima).
      - If V_r == V_c, every intersection (row, col) where matrix[row,col] == V_r
        and that row's minimum == V_r and that column's maximum == V_c
        is a saddle point.

    This guarantees that the saddle point ALWAYS lies at the intersection
    of a Row-Min and a Col-Max (exactly what ma'am asked).
    """
    if matrix is None or matrix.empty:
        return []

    # Ensure we only use the core A×B matrix (3x3) with proper labels
    mat = matrix.loc[A_LABELS, B_LABELS].astype(float)

    row_mins = mat.min(axis=1)
    col_maxs = mat.max(axis=0)

    max_row_min = row_mins.max()
    min_col_max = col_maxs.min()

    # No saddle if these don't match
    if not np.isclose(max_row_min, min_col_max):
        return []

    saddle_value = max_row_min

    # Candidate rows: those whose row-min equals max_row_min
    candidate_rows = [r for r in A_LABELS if np.isclose(row_mins[r], max_row_min)]
    # Candidate cols: those whose col-max equals min_col_max
    candidate_cols = [c for c in B_LABELS if np.isclose(col_maxs[c], min_col_max)]

    sps: List[Dict[str, Any]] = []
    for r in candidate_rows:
        for c in candidate_cols:
            if np.isclose(mat.loc[r, c], saddle_value):
                sps.append(
                    {
                        "rows": r,
                        "cols": c,
                        "value": float(saddle_value),
                    }
                )
    return sps

# ======================================================
# Single Simulation Round
# ======================================================
def simulate_one(data_df: pd.DataFrame, sl_no: int):
    """
    Run one simulation round:
    - Randomly pick two supplier names
    - Generate a 3×3 payoff matrix
    - Identify saddle points (if any) using Row-Min / Col-Max rule
    - Sample descriptive strategies for each supplier
    - Record all negative payoffs as 'loss' events
    """
    if (
        data_df is not None
        and not data_df.empty
        and "Supplier_Name" in data_df.columns
    ):
        names = data_df["Supplier_Name"].dropna().unique()
    else:
        names = np.array(["Supplier X", "Supplier Y"])

    if len(names) < 2:
        names = np.array(["Supplier X", "Supplier Y"])

    s1 = np.random.choice(names)
    s2 = np.random.choice(names)
    while s2 == s1:
        s2 = np.random.choice(names)

    payoff_df = generate_payoff_matrix(data_df)
    core_matrix = payoff_df.loc[A_LABELS, B_LABELS].copy()
    sps = find_saddle_points_rowmin_colmax(core_matrix)

    s1_strategy = np.random.choice(SUPPLIER1_STRATEGIES)
    s2_strategy = np.random.choice(SUPPLIER2_STRATEGIES)

    losses_list: List[List[Any]] = []
    for r in A_LABELS:
        for c in B_LABELS:
            val = int(core_matrix.loc[r, c])
            if val < 0:
                losses_list.append(
                    [sl_no, s1, s2, float(val), "Simulation"]
                )

    row = [
        sl_no,
        s1,
        s2,
        "Yes" if sps else "No",
        s1_strategy,
        s2_strategy,
    ]
    return row, payoff_df, sps, losses_list

# ======================================================
# Run Multiple Simulations
# ======================================================
def simulate_many(data_df: pd.DataFrame, rounds: int):
    """
    Execute multiple simulation rounds and aggregate:
    - All simulation results
    - Last payoff matrix and its saddle points
    - All loss events across simulations
    """
    rows = []
    all_losses: List[List[Any]] = []
    last_payoff = None
    last_sps = None

    for i in range(1, rounds + 1):
        row, payoff_df, sps, losses = simulate_one(data_df, i)
        rows.append(row)
        all_losses.extend(losses)
        last_payoff, last_sps = payoff_df, sps

    results_df = pd.DataFrame(
        rows,
        columns=[
            "SL_No",
            "Supplier1",
            "Supplier2",
            "Saddle_Point",
            "Supplier1_Strategy",
            "Supplier2_Strategy",
        ],
    )
    return results_df, last_payoff, last_sps, all_losses

# ======================================================
# Payoff Matrix Page (ENHANCED — Payoff → Saddle Point → Graph → Heatmap)
# ======================================================
def render_payoff_matrix_ui(data_df: pd.DataFrame, DB_PATH: Path):

    # -----------------------
    # Page Title
    # -----------------------
    st.title("📊 Supplier Payoff Matrix (3×3)")
    st.caption(
        "This view compares Supplier-1 (rows) and Supplier-2 (columns) strategic decisions "
        "under a competitive business environment."
    )

    st.markdown("---")

    # -----------------------------------------
    # Generate New Matrix Button
    # -----------------------------------------
    if st.button("🔄 Generate New Random Payoff Matrix", use_container_width=True):
        st.session_state["payoff_df"] = generate_payoff_matrix(data_df)

    if "payoff_df" not in st.session_state:
        st.session_state["payoff_df"] = generate_payoff_matrix(data_df)

    payoff_df = st.session_state["payoff_df"].copy()

    # Core 3×3 matrix (before adding Row Min / Col Max extras)
    core_matrix = payoff_df.loc[A_LABELS, B_LABELS].copy()

    # Add Row-Min & Column-Max for clearer analysis
    payoff_df["Row Min"] = core_matrix.min(axis=1)
    col_max_row = core_matrix.max(axis=0).to_frame().T.rename(index={0: "Col Max"})
    payoff_df_display = pd.concat([payoff_df, col_max_row])

    # ✅ Compute saddle points using ONLY Row-Min & Col-Max logic
    sps = find_saddle_points_rowmin_colmax(core_matrix)

    # ======================================================
    # 🟦 1) PAYOFF MATRIX (FIRST)
    # ======================================================
    st.subheader("📘 1) Payoff Matrix (with Row-Min and Column-Max)")

    st.markdown(
        """
The matrix below summarises the **payoff outcomes** for each combination of supplier strategies.

- **Row Min** helps identify the minimum guaranteed payoff for each Supplier-1 strategy.  
- **Col Max** highlights the maximum payoff available to Supplier-2 across each column.  

This structure supports systematic evaluation of conservative and aggressive strategy choices.
        """
    )

    styled_matrix = (
        payoff_df_display.style
        .applymap(highlight_losses)
        .format("{:,.0f}")
    )

    st.dataframe(styled_matrix, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # 🟩 2) SADDLE POINT ANALYSIS (SECOND)
    # ======================================================
    st.subheader("🎯 2) Saddle Point Analysis (Equilibrium Detection)")

    saddle_exists = bool(sps)
    saddle_value = None
    chosen_row_label = None
    chosen_col_label = None
    s1_choice = None
    s2_choice = None

    if saddle_exists:
        # Take the first saddle point for interpretation (there can be more than one)
        sp0 = sps[0]
        chosen_row_label = sp0["rows"]   # e.g., "A2"
        chosen_col_label = sp0["cols"]   # e.g., "B1"
        saddle_value = float(sp0["value"])

        row_idx = A_LABELS.index(chosen_row_label)
        col_idx = B_LABELS.index(chosen_col_label)

        s1_choice = SUPPLIER1_STRATEGIES[row_idx]
        s2_choice = SUPPLIER2_STRATEGIES[col_idx]

        # Display each saddle point (usually one) – ALL of these satisfy Row-Min & Col-Max rule
        for sp in sps:
            r_lbl = sp["rows"]
            c_lbl = sp["cols"]
            st.success(
                f"""
### ✔ Saddle Point Identified at **Row `{r_lbl}` × Column `{c_lbl}`**  

A saddle point represents a **strategic equilibrium** in the payoff matrix, obtained
strictly from the intersection of **Row-Min** and **Column-Max** values.

At this position, neither supplier can improve their payoff by changing their own strategy alone,
while the other supplier keeps their decision fixed.

#### Strategic Interpretation
- **Supplier-1 Strategy:** *{s1_choice}*  
- **Supplier-2 Strategy:** *{s2_choice}*  

This outcome can be interpreted as a **mutually stable business decision**,  
where both parties accept the trade-off between risk and expected profit.
                """
            )
    else:
        st.warning(
            "⚠ No saddle point exists for the current payoff matrix using the Row-Min / Col-Max rule. "
            "This indicates that there is no single strategy pair that is simultaneously "
            "optimal and stable for both suppliers."
        )

    st.markdown("---")

    # ======================================================
    # 🟥 3) PURE STRATEGY PREDICTION LINE GRAPH & STRENGTH
    # ======================================================
    st.subheader("📈 3) Pure Strategy Prediction Line Graph & Strength")

    try:
        # Row minima (Supplier A “worst–case”) and column maxima (Supplier B “best–case”)
        row_mins = core_matrix.min(axis=1)
        col_maxs = core_matrix.max(axis=0)

        # X-axis: strategy indices (1, 2, 3 …)
        x_idx = np.arange(1, len(A_LABELS) + 1)

        line_df = pd.DataFrame(
            {
                "Strategy Index": x_idx,
                "Supplier A (Row Min)": row_mins.values,
                "Supplier B (Col Max)": col_maxs.values,
            }
        )

        line_melt = line_df.melt(
            id_vars="Strategy Index",
            var_name="Series",
            value_name="Payoff",
        )

        fig_line = px.line(
            line_melt,
            x="Strategy Index",
            y="Payoff",
            color="Series",
            markers=True,
        )

        # Horizontal line at saddle point (if it exists)
        if saddle_exists and saddle_value is not None:
            fig_line.add_hline(
                y=saddle_value,
                line_dash="dash",
                annotation_text=f"Saddle point = {saddle_value:.2f}",
                annotation_position="top left",
            )

        fig_line.update_layout(
            title="Pure Strategy Prediction Level",
            xaxis_title="Supplier A strategy 1   |   Supplier B strategy 2",
            yaxis_title="Saddle point / payoff level",
        )

        st.plotly_chart(fig_line, use_container_width=True)

        # ----------------------------
        # Pure Strategy Prediction Percentage
        # ----------------------------
        if saddle_exists and saddle_value is not None:
            global_min = float(core_matrix.values.min())
            global_max = float(core_matrix.values.max())

            if np.isclose(global_min, global_max):
                prediction_percent = 100.0
            else:
                # Centre-based scaling:
                #  - Saddle at middle of range → 50%
                #  - Saddle near extremes → closer to 100%
                mid = (global_min + global_max) / 2.0
                spread_half = (global_max - global_min) / 2.0
                dist_from_mid = abs(saddle_value - mid) / spread_half
                dist_from_mid = float(np.clip(dist_from_mid, 0.0, 1.0))
                prediction_percent = 50.0 + 50.0 * dist_from_mid

            prediction_percent = round(prediction_percent, 2)

            st.markdown("### 📊 Pure Strategy Prediction Strength (%)")
            st.success(f"**Pure Strategy Prediction Power:** {prediction_percent} %")

            st.markdown(
                f"""
The **prediction strength** measures how dominant the pure-strategy equilibrium is
within the entire payoff range.

- Values closer to **100%** indicate a **strong and stable** pure-strategy prediction.  
- Values around **50%** indicate a **moderate** prediction strength.  

For this payoff matrix, the pure strategy shows a prediction strength of  
**`{prediction_percent} %`**, meaning the saddle-point level is comparatively
strong within the observed profit range.
                """
            )
        else:
            st.info(
                "Since no exact saddle point exists (by Row-Min / Col-Max rule), a pure-strategy "
                "prediction strength cannot be calculated for this matrix."
            )

    except Exception as e:
        st.warning(f"Could not draw pure strategy line graph / prediction strength: {e}")

    st.markdown("---")

    # ======================================================
    # 🟧 4) HEATMAP VISUALIZATION (FOURTH)
    # ======================================================
    st.subheader("🌈 4) Heatmap Visualization of Payoffs")

    st.caption(
        "The heatmap below provides a visual perspective of payoff intensity. "
        "Darker shades indicate higher payoff values for the corresponding strategy pair."
    )

    fig = px.imshow(
        core_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # Negative Payoff Table
    # ---------------------------------------------------
    losses_local = [
        {"Row": r, "Col": c, "Value": int(core_matrix.loc[r, c])}
        for r in A_LABELS
        for c in B_LABELS
        if int(core_matrix.loc[r, c]) < 0
    ]

    if losses_local:
        st.subheader("💔 Negative Payoff Instances")
        st.caption(
            "These cells represent scenarios where the resulting profit is negative, "
            "indicating potential loss or unfavourable contract terms."
        )
        st.table(pd.DataFrame(losses_local).style.applymap(highlight_losses))

    st.markdown("---")

# ======================================================
# Simulation Page
# ======================================================
def render_simulation_ui(data_df: pd.DataFrame, DB_PATH: Path):
    st.title("🧵 Pure-Strategy Simulation (Clothing Suppliers)")

    st.caption(
        "Simulate multiple rounds of supplier interactions to study how often saddle points arise "
        "and how frequently suppliers incur losses."
    )

    rounds = st.slider("Select number of simulation iterations", 1, 200, 10)

    col_run1, col_run2 = st.columns([1, 1])
    with col_run1:
        run_sim = st.button("▶️ Run Simulation", type="primary")
    with col_run2:
        save_after = st.checkbox(
            "💾 Auto-save results to database",
            value=False,
            help="If enabled, all simulation outcomes and loss events will be stored in SQLite.",
        )

    if not run_sim:
        return

    results_df, last_payoff, last_sps, losses_all = simulate_many(data_df, rounds)
    st.session_state["sim_results_df"] = results_df
    st.session_state["last_payoff_df"] = last_payoff
    st.session_state["last_sps"] = last_sps
    st.session_state["last_losses"] = losses_all

    st.success(f"Simulation completed for **{rounds}** iteration(s). ✅")

    payoff_df = last_payoff.copy()
    core_matrix = payoff_df.loc[A_LABELS, B_LABELS].copy()
    payoff_df["Row Min"] = core_matrix.min(axis=1)
    col_max_row = core_matrix.max(axis=0).to_frame().T.rename(index={0: "Col Max"})
    payoff_df_display = pd.concat([payoff_df, col_max_row])

    st.subheader("🧾 Last Payoff Matrix (from Simulation)")
    st.dataframe(
        payoff_df_display.style.applymap(highlight_losses).format("{:,.0f}"),
        use_container_width=True,
    )

    if not results_df.empty:
        st.subheader("📊 Simulation Results (All Rounds in This Run)")
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            "⬇️ Download Simulation Results (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="simulation_results.csv",
        )

    if losses_all:
        losses_df = pd.DataFrame(
            losses_all,
            columns=["SL_No", "Supplier1", "Supplier2", "Loss_Value", "Source"],
        )
        st.subheader("💔 Loss Events Observed in Simulation")
        st.dataframe(
            losses_df.style.applymap(highlight_losses),
            use_container_width=True,
        )

    if save_after and not results_df.empty:
        with get_conn(DB_PATH) as conn:
            save_results_df(results_df, conn)
            save_losses(losses_all, conn)
        st.success("✅ Simulation results and losses have been saved to the database.")

# ======================================================
# Results Page
# ======================================================
def render_results_ui(DB_PATH: Path):
    st.title("📑 Stored Simulation & Strategy Results")

    with get_conn(DB_PATH) as conn:
        df_db = fetch_all_results(conn)
        df_losses = fetch_all_losses(conn)
        try:
            ensure_generated_strategies_table(conn)
            df_generated = fetch_all_generated_strategies(conn)
        except Exception:
            df_generated = pd.DataFrame()

    # ---------------- Simulation Results ----------------
    if df_db.empty:
        st.info("No stored pure-strategy results are available yet. Please run a simulation first.")
    else:
        st.subheader("📈 Pure-Strategy Simulation Results (All Stored Runs)")
        st.dataframe(df_db, use_container_width=True)
        st.download_button(
            "⬇️ Download stored pure-strategy results (CSV)",
            data=df_db.to_csv(index=False).encode("utf-8"),
            file_name="stored_simulation_results.csv",
        )

    # ---------------- Recorded Losses ----------------
    if df_losses.empty:
        st.info("No recorded loss events have been stored in the database.")
    else:
        st.subheader("💔 Recorded Losses Across All Simulations")
        st.dataframe(
            df_losses.style.applymap(highlight_losses),
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download recorded losses (CSV)",
            data=df_losses.to_csv(index=False).encode("utf-8"),
            file_name="stored_losses.csv",
        )

    # ---------------- Mixed Strategy Results ----------------
    st.divider()
    st.subheader("📚 Mixed-Strategy Results (Generated Strategies from Solver)")
    if df_generated is None or df_generated.empty:
        st.info("No mixed-strategy records have been saved in the database yet.")
    else:
        display_df = df_generated.copy()
        if "probability" in display_df.columns:
            display_df["probability"] = display_df["probability"].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "-"
            )
        if "mapped_value" in display_df.columns:
            display_df["mapped_value"] = display_df["mapped_value"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "-"
            )
        st.dataframe(display_df, use_container_width=True)
        st.download_button(
            "⬇️ Download saved mixed-strategy records (CSV)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="db_saved_generated_strategies.csv",
        )

    # ======================================================
    # Classifier Section
    # ======================================================
    st.divider()
    st.subheader("🧹 Classifier — Flag & Quarantine 'I won't do business' Rows")

    if not UTIL_ML:
        st.info(
            "Machine-learning classifier helpers are not available. "
            "Please install `scikit-learn` and `joblib` and ensure `utils.SKLEARN_AVAILABLE` is True "
            "to enable this feature."
        )
        st.caption(f"Example local file for testing: {DEFAULT_UPLOADED_FILE}")
        return

    # ---- Training / Loading Controls ----
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Model Training / Loading**")
        upload_train = st.file_uploader(
            "Upload a labelled CSV for training (columns: text, label)",
            type=["csv"],
            key="train_csv",
        )
        do_auto_label = st.checkbox(
            "Auto-label existing generated_strategies (rule-based) and train",
            value=False,
        )

    with col2:
        if st.button("Train model from uploaded CSV"):
            if upload_train is None:
                st.error("Please upload a labelled CSV file before training.")
            else:
                try:
                    train_df = pd.read_csv(upload_train)
                    if (
                        "text" not in train_df.columns
                        and "strategy_label" in train_df.columns
                    ):
                        train_df = train_df.rename(columns={"strategy_label": "text"})
                    if "label" not in train_df.columns:
                        st.error("The training CSV must contain both 'text' and 'label' columns.")
                    else:
                        model_path, _ = train_strategy_svm_from_dataframe(
                            train_df,
                            text_col="text",
                            label_col="label",
                        )
                        st.success(f"Trained and saved model → {model_path}")
                except Exception as e:
                    st.error(f"Training failed: {e}")

        if do_auto_label and st.button("Auto-label & Train"):
            try:
                with get_conn(DB_PATH) as conn:
                    ensure_generated_strategies_table(conn)
                    df_gen = fetch_all_generated_strategies(conn)
                if df_gen.empty:
                    st.error("No generated strategies are available in the database for auto-labelling.")
                else:
                    if "mapped_strategy" in df_gen.columns:
                        texts = df_gen["mapped_strategy"].astype(str).tolist()
                    else:
                        texts = df_gen.iloc[:, 0].astype(str).tolist()

                    labels = auto_label_from_texts(texts)

                    train_rows = [
                        {"text": t, "label": int(lab)}
                        for t, lab in zip(texts, labels)
                        if lab in (0, 1)
                    ]
                    train_df = pd.DataFrame(train_rows)

                    if train_df.empty:
                        st.error(
                            "The auto-labelling step did not produce enough clear 0/1 rows for training."
                        )
                    else:
                        model_path, _ = train_strategy_svm_from_dataframe(
                            train_df, text_col="text", label_col="label"
                        )
                        st.success(f"Trained classifier model → {model_path.name}")
            except Exception as e:
                st.error(f"Auto-label/train failed: {e}")

        if st.button("Load existing model"):
            try:
                _ = load_strategy_model()
                st.success("Existing model loaded successfully.")
            except Exception as e:
                st.error(f"Model load failed: {e}")

    # ---- Prediction / Quarantine Flow ----
    st.markdown("**Run classifier on database rows and quarantine flagged entries**")
    conf_thresh = st.slider(
        "Confidence threshold for quarantining (lower → more rows flagged)",
        0.0,
        1.0,
        0.6,
        0.05,
    )

    pred_col1, pred_col2 = st.columns([2, 1])
    with pred_col1:
        source_field = st.selectbox(
            "Select text column for classification:",
            options=["mapped_strategy", "strategy_label"],
            index=0,
        )
    with pred_col2:
        run_predict = st.button("🔎 Run classifier on generated_strategies")

    if run_predict:
        try:
            with get_conn(DB_PATH) as conn:
                ensure_generated_strategies_table(conn)
                df_gen = fetch_all_generated_strategies(conn)

            if df_gen.empty:
                st.error("The database does not contain any generated strategies to classify.")
            else:
                if source_field not in df_gen.columns:
                    source_field = "strategy_label"

                texts = df_gen[source_field].astype(str).tolist()
                preds = predict_strategy_labels(texts)

                pred_rows = []
                flagged_ids = []

                for i, p in enumerate(preds):
                    pred_label = p.get("pred")
                    prob = float(p.get("prob") or 0.0)

                    orig_row = df_gen.iloc[i].to_dict()

                    is_remove = False
                    if isinstance(pred_label, (int, float)) and int(pred_label) == 0:
                        is_remove = True
                    elif isinstance(pred_label, str) and pred_label.strip().lower() in (
                        "0",
                        "remove",
                        "i won't do business",
                        "no",
                        "n",
                    ):
                        is_remove = True

                    flagged = is_remove and prob >= conf_thresh
                    if flagged and "id" in df_gen.columns:
                        flagged_ids.append(int(df_gen.iloc[i]["id"]))

                    pred_rows.append(
                        {
                            **orig_row,
                            "pred_label": pred_label,
                            "pred_prob": prob,
                            "flagged": flagged,
                        }
                    )

                df_preds = pd.DataFrame(pred_rows)
                n_flagged = df_preds["flagged"].sum()

                st.success(
                    f"Classifier execution completed — **{n_flagged}** row(s) flagged "
                    f"for potential quarantine at threshold {conf_thresh:.2f}."
                )
                st.dataframe(df_preds.head(200), use_container_width=True)

                if flagged_ids:
                    if st.button("🛡️ Move flagged rows to quarantine table"):
                        try:
                            with get_conn(DB_PATH) as conn:
                                moved = move_rows_to_quarantine(conn, flagged_ids)
                            st.success(f"Successfully moved {moved} row(s) into the quarantine table.")
                        except Exception as e:
                            st.error(f"Move to quarantine failed: {e}")

        except Exception as e:
            st.error(f"Classifier run failed: {e}")

    # ---------------- Quarantine Table Review ----------------
    st.divider()
    st.subheader("🧳 Quarantine Table Review")

    try:
        with get_conn(DB_PATH) as conn:
            ensure_quarantine_table(conn)
            df_quar = fetch_all_quarantine(conn)

        if df_quar.empty:
            st.info("No rows have been quarantined yet.")
        else:
            st.dataframe(df_quar, use_container_width=True)
            st.caption(
                "Quarantined rows are retained for audit and can be safely reviewed, "
                "restored or permanently removed from within the Mixed-Strategy interface."
            )
    except Exception as e:
        st.error(f"Could not access quarantine table: {e}")
