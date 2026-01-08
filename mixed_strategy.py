# mixed_strategy.py — Final Themed Version (Fashion Supplier Strategy Dashboard)
# UPDATED:
# - Retailer-based payoff matrix for mixed strategy
# - SVM-based Mixed-Strategy prediction line graph
# - Prediction Strength = average SVM confidence × 100
# - Restore / Permanently Delete actions for quarantine rows with safe confirmations
# - Auto-quarantine on Train default SVM button
# - SVM mixed-strategy line graph & prediction appear ONLY AFTER pressing
#   "Train default SVM (quick)" in the Mixed-Strategy section.

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sqlite3
import plotly.express as px
from typing import List, Tuple, Dict, Any, Optional

# Developer-provided local file path (from workspace) — used in info text
LOCAL_UPLOADED_FILE_URL = "/mnt/data/Screenshot 2025-11-20 234737.png"

# Utilities from utils
from utils import (
    A_LABELS,
    B_LABELS,
    highlight_losses,
    generate_payoff_matrix,
    get_conn,
    DB_PATH,
    ensure_generated_strategies_table,
    fetch_all_generated_strategies,
    save_generated_strategies,
    ensure_quarantine_table,
    fetch_quarantine_generated_strategies,
    move_rows_to_quarantine,
    train_default_svm,
    load_strategy_svm,
    predict_strategy_texts,   # used for SVM probabilities
    preview_removals_from_db,
    SKLEARN_AVAILABLE,  # ML availability flag
)

# Flag: are ML helpers + scikit-learn/joblib available?
UTIL_ML = bool(SKLEARN_AVAILABLE)

# Fallback DB path if DB_PATH is None (should already be defined in utils)
FALLBACK_DB_PATH = os.path.join(os.path.dirname(__file__), "game_theory.db")


def conn_factory():
    if DB_PATH is not None:
        return lambda: get_conn(DB_PATH)
    return lambda: sqlite3.connect(FALLBACK_DB_PATH)


# --- Strategy helpers ---
STRATEGY_COLUMN_CANDIDATES = [
    "strategy",
    "strategies",
    "choice",
    "choices",
    "decision",
    "response",
    "action",
    "answer",
]

_STR_DO_BUSINESS = [
    "do business",
    "will do business",
    "i will do business",
    "i'll do business",
    "ill do business",
    "do it",
    "yes",
    "y",
    "agree",
    "accept",
    "business",
]

_STR_WONT_BUSINESS = [
    "won't do business",
    "won't do",
    "will not do business",
    "will not do",
    "do not business",
    "do not",
    "dont do business",
    "don't do business",
    "no",
    "n",
    "decline",
    "reject",
    "refuse",
    "not interested",
]

_STR_FIFTY = [
    "50-50",
    "50/50",
    "50 50",
    "50%",
    "50",
    "half",
    "50–50",
    "fifty fifty",
    "mixed",
    "maybe",
    "neutral",
]


def detect_strategy_col(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in STRATEGY_COLUMN_CANDIDATES:
        if candidate in cols_lower:
            return cols_lower[candidate]
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


def normalize_text(x: Any) -> str:
    try:
        return str(x).strip().lower()
    except Exception:
        return ""


def map_strategy_text_to_category(text: Any) -> Optional[str]:
    t = normalize_text(text)
    for patt in _STR_DO_BUSINESS:
        if patt in t:
            return "I will do business"
    for patt in _STR_WONT_BUSINESS:
        if patt in t:
            return "I won't do business"
    for patt in _STR_FIFTY:
        if patt in t:
            return "50-50"
    try:
        num = float(t)
        if num == 1.0:
            return "I will do business"
        if num == 0.0:
            return "I won't do business"
        if abs(num - 0.5) < 1e-8:
            return "50-50"
    except Exception:
        pass
    return None


def strategy_label_to_value(label: str) -> float:
    if label == "I will do business":
        return 1.0
    if label == "I won't do business":
        return 0.0
    if label == "50-50":
        return 0.5
    return np.nan


def map_prob_to_strategy_label(prob: Optional[float]) -> str:
    """
    Map a probability in [0,1] to the nearest of:
      0.0 -> "I won't do business"
      0.5 -> "50-50"
      1.0 -> "I will do business"
    """
    try:
        if prob is None or (isinstance(prob, float) and np.isnan(prob)):
            return "Unknown"
        candidates = {
            "I will do business": 1.0,
            "50-50": 0.5,
            "I won't do business": 0.0,
        }
        best = min(candidates.items(), key=lambda kv: abs(prob - kv[1]))
        return best[0]
    except Exception:
        return "Unknown"


def extract_strategy_probabilities(data_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect a dataframe and try to map a strategy-like text column into counts/probabilities.
    Returns:
      {
        "found_column": str or None,
        "counts": {label -> count},
        "probs": {label -> probability},
        "expected_value": float or None,
        "total": int,
        "per_row_mapped_values": list
      }
    """
    result = {
        "found_column": None,
        "counts": {
            "I will do business": 0,
            "I won't do business": 0,
            "50-50": 0,
            "unknown": 0,
        },
        "probs": {},
        "expected_value": None,
        "total": 0,
        "per_row_mapped_values": [],
    }
    if data_df is None or data_df.shape[0] == 0:
        return result
    col = detect_strategy_col(data_df)
    if col is None:
        return result
    result["found_column"] = col
    total = 0
    mapped_values = []
    for raw in data_df[col].dropna().values:
        total += 1
        cat = map_strategy_text_to_category(raw)
        if cat is None:
            result["counts"]["unknown"] += 1
            mapped_values.append(np.nan)
        else:
            result["counts"][cat] = result["counts"].get(cat, 0) + 1
            mapped_values.append(strategy_label_to_value(cat))
    result["total"] = total
    probs = {}
    for k, v in result["counts"].items():
        probs[k] = (v / total) if total > 0 else 0.0
    result["probs"] = probs
    mapped_array = np.array(
        [v for v in mapped_values if not (isinstance(v, float) and np.isnan(v))],
        dtype=float,
    )
    if mapped_array.size > 0:
        result["expected_value"] = float(np.nanmean(mapped_array))
    else:
        result["expected_value"] = None
    result["per_row_mapped_values"] = mapped_values
    return result


def is_row_dominated(df: pd.DataFrame, r_dom: str, r_by: str, cols: List[str]) -> bool:
    a = df.loc[r_dom, cols].astype(float).values
    b = df.loc[r_by, cols].astype(float).values
    return np.all(a <= b) and np.any(a < b)


def is_col_dominated(df: pd.DataFrame, c_dom: str, c_by: str, rows: List[str]) -> bool:
    a = df.loc[rows, c_dom].astype(float).values
    b = df.loc[rows, c_by].astype(float).values
    return np.all(a >= b) and np.any(a > b)


def reduce_by_dominance(df: pd.DataFrame):
    working_rows = list(df.index)
    working_cols = list(df.columns)
    steps, removed_rows, removed_cols = [], [], []
    while True:
        changed = False
        for r_dom in list(working_rows):
            for r_by in list(working_rows):
                if r_dom == r_by:
                    continue
                if is_row_dominated(df, r_dom, r_by, working_cols):
                    working_rows.remove(r_dom)
                    removed_rows.append((r_dom, r_by))
                    steps.append({"type": "row", "removed": r_dom, "by": r_by})
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue
        for c_dom in list(working_cols):
            for c_by in list(working_cols):
                if c_dom == c_by:
                    continue
                if is_col_dominated(df, c_dom, c_by, working_rows):
                    working_cols.remove(c_dom)
                    removed_cols.append((c_dom, c_by))
                    steps.append({"type": "col", "removed": c_dom, "by": c_by})
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    final_df = df.loc[working_rows, working_cols].copy()
    info = {
        "removed_rows": removed_rows,
        "removed_cols": removed_cols,
        "steps": steps,
        "final_rows": working_rows,
        "final_cols": working_cols,
    }
    return final_df, info


def solve_algebraic_2x2(matrix: pd.DataFrame):
    a11, a12 = matrix.iloc[0, 0], matrix.iloc[0, 1]
    a21, a22 = matrix.iloc[1, 0], matrix.iloc[1, 1]
    denom = a11 + a22 - (a12 + a21)
    if denom == 0:
        return {"error": "Degenerate case (denominator = 0)"}
    p1 = (a22 - a21) / denom
    p2 = 1 - p1
    q1 = (a22 - a12) / denom
    q2 = 1 - q1
    V = (a11 * a22 - a12 * a21) / denom
    return {"p1": p1, "p2": p2, "q1": q1, "q2": q2, "V": V}


def build_payoff_from_data(
    data_df: pd.DataFrame = None, rows: int = 3, cols: int = 3, seed=None
):
    """
    Build a payoff matrix for the mixed-strategy solver.

    IMPORTANT:
    - For MIXED STRATEGY we prefer to use Retailer profits from the dataset.
    - If no retailer-profit column is available, we fall back to Supplier_Profit.
    """
    if seed is not None:
        np.random.seed(seed)

    profit_col: Optional[str] = None

    if data_df is not None and not data_df.empty:
        cols_lower = {c.lower(): c for c in data_df.columns}

        retailer_candidates_lower = [
            "retailer_profit",
            "retailer profit",
            "retailer_margin",
            "retailer margin",
            "retailer_payoff",
            "retailer payoff",
            "retailer_profit_value",
        ]
        for cand in retailer_candidates_lower:
            if cand in cols_lower:
                profit_col = cols_lower[cand]
                break

        if profit_col is None:
            if "Retailer_Profit" in data_df.columns:
                profit_col = "Retailer_Profit"
            elif "Retailer Profit" in data_df.columns:
                profit_col = "Retailer Profit"

        if profit_col is None:
            if "Supplier_Profit" in data_df.columns:
                profit_col = "Supplier_Profit"
            elif "Supplier Profit" in data_df.columns:
                profit_col = "Supplier Profit"

    if (
        data_df is None
        or profit_col is None
        or profit_col not in data_df.columns
    ):
        mat = np.random.randint(-10, 25, size=(rows, cols))
    else:
        pool = pd.to_numeric(data_df[profit_col], errors="coerce").dropna().values
        if pool.size == 0:
            pool = np.array([10, -5, 20, -15])
        sampled = np.random.choice(pool, size=(rows * cols), replace=True)
        mat = np.round(sampled.reshape(rows, cols)).astype(int)

    row_labels = A_LABELS[:rows]
    col_labels = B_LABELS[:cols]
    return pd.DataFrame(mat, index=row_labels, columns=col_labels)


# ======================================================
# Helper: Mixed Strategy Prediction Strength
# ======================================================
def _entropy_based_strength(probs: np.ndarray) -> float:
    """
    Mixed-strategy prediction strength (0–100 %).

    Here `probs` are P(class = 0) = probability of "I won't do business".
    We convert them into a **confidence score**:

        confidence = max(p, 1 - p)

    So:
      - p = 0.95 → very sure negative → confidence = 0.95
      - p = 0.05 → very sure positive → confidence = 0.95
      - p = 0.50 → unsure → confidence = 0.50

    Final strength = average(confidence) × 100.
    """
    # Filter out invalid values
    valid = [p for p in probs if p is not None and not np.isnan(p)]
    if not valid:
        return np.nan

    # Convert raw probabilities into symmetric confidence scores
    confidences = [max(p, 1.0 - p) for p in valid]

    mean_conf = float(np.mean(confidences))
    strength = mean_conf * 100.0
    return float(np.clip(strength, 0.0, 100.0))


# ======================================================
# Streamlit UI — Mixed Strategy Solver (with classifier controls)
# ======================================================
def render_mixed_strategy_ui(data_df: pd.DataFrame = None):
    st.title("🧮 Mixed Strategy Solver (Fashion Supplier Analysis)")
    st.caption("Reduce by dominance → Simplify → Solve algebraically if 2×2")
    st.divider()

    # Auto-save checkbox (like simulation)
    save_after = st.checkbox(
        "💾 Auto-save generated strategies to DB",
        value=False,
        help=(
            "If checked, solver outputs will be saved automatically to the "
            "generated_strategies table when produced."
        ),
    )

    # Use last payoff matrix (from Pure Strategy) if available
    use_last = "payoff_df" in st.session_state and st.checkbox(
        "Use last payoff matrix (from Pure Strategy page)", value=True
    )

    if use_last and "payoff_df" in st.session_state:
        matrix_df = st.session_state["payoff_df"].copy()
        matrix_df = matrix_df.loc[A_LABELS, B_LABELS]
    else:
        matrix_df = build_payoff_from_data(data_df)

    # Original payoff matrix display
    st.subheader("👚 Original Payoff Matrix (Retailer Profits for Mixed Strategy)")
    st.dataframe(
        matrix_df.style.map(highlight_losses).format("{:,.0f}"),
        width="stretch",
    )

    # Strategy probabilities from dataset
    """if data_df is not None:
        strat_info = extract_strategy_probabilities(data_df)
        col_name = strat_info["found_column"]
        if col_name is not None:
            st.divider()
            st.subheader("📊 Strategy Probabilities (from dataset)")
            st.markdown(f"**Detected strategy column:** `{col_name}`")
            display_rows = []
            for label in ["I will do business", "50-50", "I won't do business", "unknown"]:
                prob = strat_info["probs"].get(label, 0.0)
                mapped_val = (
                    strategy_label_to_value(label)
                    if label in ["I will do business", "50-50", "I won't do business"]
                    else None
                )
                if mapped_val is not None and not np.isnan(mapped_val):
                    display_rows.append((label, f"{mapped_val:.1f}", f"{prob:.3f}"))
                else:
                    display_rows.append((label, "-", f"{prob:.3f}"))
            df_display = pd.DataFrame(
                display_rows, columns=["Strategy", "Mapped Value", "Dataset Probability"]
            ).set_index("Strategy")
            st.table(df_display)

            known_probs = {k: v for k, v in strat_info["probs"].items() if k != "unknown"}
            if known_probs:
                max_prob = max(known_probs.values())
                mapped_best = map_prob_to_strategy_label(max_prob)
                st.markdown(
                    f"**Most common strategy (by nearby value):** "
                    f"`{mapped_best}` with probability **{max_prob:.3f}**"
                )
            else:
                st.info("No recognizable strategies found in dataset.")
        else:
            st.info(
                "No strategy-like column detected in dataset. Provide a column named "
                "'strategy', 'choice', or similar."
            )
"""
    # Reduction button & process
    if st.button("🧾 Reduce Matrix by Dominance", use_container_width=True):
        reduced_df, info = reduce_by_dominance(matrix_df)
        st.divider()
        st.subheader("🧩 Step-by-Step Reduction Process")
        if not info["steps"]:
            st.info("✅ No dominated strategies found. Matrix is already optimal.")
        else:
            for i, step in enumerate(info["steps"], start=1):
                emoji = "🧺" if step["type"] == "row" else "🎽"
                st.markdown(
                    f"{emoji} **Step {i}:** Removed `{step['removed']}` "
                    f"dominated by `{step['by']}`"
                )
        st.subheader("📉 Reduced Payoff Matrix")
        st.dataframe(
            reduced_df.style.map(highlight_losses).format("{:,.0f}"),
            width="stretch",
        )

        # Solve if 2x2
        if reduced_df.shape == (2, 2):
            st.success("🎯 Reduced matrix is 2×2 — solving algebraically.")
            res = solve_algebraic_2x2(reduced_df)
            if "error" in res:
                st.error(res["error"])
            else:
                st.markdown(
                    f"""
                ### 🧵 Mixed Strategy Solution
                - **Supplier A probabilities:** p₁ = {res['p1']:.3f}, p₂ = {res['p2']:.3f}  
                - **Supplier B probabilities:** q₁ = {res['q1']:.3f}, q₂ = {res['q2']:.3f}  
                """
                )
                # Build generated strategies dataframe from solver probabilities
                try:
                    a_labels = list(reduced_df.index)
                    p_vals = [res.get("p1", np.nan), res.get("p2", np.nan)]
                    b_labels = list(reduced_df.columns)
                    q_vals = [res.get("q1", np.nan), res.get("q2", np.nan)]
                    rows = []
                    for lbl, prob in zip(a_labels, p_vals):
                        prob_float = float(prob) if prob is not None else float("nan")
                        mapped_label = map_prob_to_strategy_label(prob_float)
                        rows.append(
                            {
                                "Player": "A",
                                "Strategy_Label": lbl,
                                "Probability": prob_float,
                                "Mapped_Strategy": mapped_label,
                                "Mapped_Value": strategy_label_to_value(mapped_label)
                                if mapped_label != "Unknown"
                                else np.nan,
                            }
                        )
                    for lbl, prob in zip(b_labels, q_vals):
                        prob_float = float(prob) if prob is not None else float("nan")
                        mapped_label = map_prob_to_strategy_label(prob_float)
                        rows.append(
                            {
                                "Player": "B",
                                "Strategy_Label": lbl,
                                "Probability": prob_float,
                                "Mapped_Strategy": mapped_label,
                                "Mapped_Value": strategy_label_to_value(mapped_label)
                                if mapped_label != "Unknown"
                                else np.nan,
                            }
                        )
                    gen_df = pd.DataFrame(rows)
                    st.session_state["generated_strategies"] = gen_df

                    # Auto-save if requested
                    if save_after:
                        try:
                            with conn_factory()() as conn:
                                ensure_generated_strategies_table(conn)
                                db_rows = []
                                for r in gen_df.to_dict(orient="records"):
                                    db_rows.append(
                                        {
                                            "player": r.get("Player"),
                                            "strategy_label": r.get("Strategy_Label"),
                                            "mapped_strategy": r.get("Mapped_Strategy"),
                                            "mapped_value": (
                                                float(r.get("Mapped_Value"))
                                                if not pd.isna(r.get("Mapped_Value"))
                                                else None
                                            ),
                                            "probability": (
                                                float(r.get("Probability"))
                                                if not pd.isna(r.get("Probability"))
                                                else None
                                            ),
                                        }
                                    )
                                inserted = save_generated_strategies(db_rows, conn)
                                if inserted:
                                    st.success(
                                        f"Auto-saved {inserted} generated strategy "
                                        f"row(s) to DB."
                                    )
                                    df_saved = fetch_all_generated_strategies(conn)
                                    st.session_state["db_saved_strategies"] = df_saved
                                else:
                                    st.info("Auto-save: no rows inserted.")
                        except Exception as e:
                            st.error(f"Auto-save to DB failed: {e}")

                except Exception as e:
                    st.error(f"Failed to build generated strategy table: {e}")
        else:
            st.warning("⚠️ Reduced matrix is not 2×2. Algebraic solution not applicable.")

    # ----------------------------
    # Show generated strategies (in-session) + save/download/load DB
    # ----------------------------
    if "generated_strategies" in st.session_state:
        try:
            st.divider()
            st.subheader("🗂️ Generated Strategies (from solver probabilities)")
            gen_df = st.session_state["generated_strategies"].copy()

            # ==================================================
            # 🤖 SVM-based Mixed-Strategy Prediction Level
            # (Shown ONLY after SVM is trained in this section)
            # ==================================================
            st.divider()
            st.subheader("🤖 SVM-based Mixed-Strategy Prediction Level")

            if not UTIL_ML:
                st.info(
                    "SVM helpers are not available. Install scikit-learn & joblib and "
                    "ensure SKLEARN_AVAILABLE=True in utils.py to enable classifier-based prediction."
                )
            else:
                # Session flag for "has user trained SVM for mixed strategy?"
                if "svm_ms_trained" not in st.session_state:
                    st.session_state["svm_ms_trained"] = False

                if st.button("🔧 Train default SVM (quick)", key="train_svm_ms"):
                    try:
                        train_default_svm()
                        st.session_state["svm_ms_trained"] = True
                        st.success(
                            "✅ Trained and saved default SVM pipeline "
                            "(strategy_svm_model.joblib)."
                        )
                    except Exception as e:
                        st.session_state["svm_ms_trained"] = False
                        st.error(f"Training failed: {e}")

                if st.session_state.get("svm_ms_trained", False):
                    try:
                        _ = load_strategy_svm()

                        # Use mapped_strategy text for classification
                        if "Mapped_Strategy" in gen_df.columns:
                            texts = gen_df["Mapped_Strategy"].fillna("").astype(str).tolist()
                        else:
                            texts = gen_df["Strategy_Label"].fillna("").astype(str).tolist()

                        raw_res = predict_strategy_texts(texts)

                        # Normalize prediction outputs
                        if isinstance(raw_res, pd.DataFrame):
                            svm_results = raw_res.to_dict(orient="records")
                        elif isinstance(raw_res, list):
                            svm_results = raw_res
                        elif isinstance(raw_res, tuple) and len(raw_res) > 0:
                            if isinstance(raw_res[0], pd.DataFrame):
                                svm_results = raw_res[0].to_dict(orient="records")
                            else:
                                svm_results = list(raw_res)
                        elif raw_res is None:
                            svm_results = []
                        else:
                            svm_results = [raw_res]

                        graph_rows: List[Dict[str, Any]] = []
                        probs_A: List[float] = []
                        probs_B: List[float] = []

                        n = min(len(gen_df), len(svm_results))
                        for idx in range(n):
                            row_dict = gen_df.iloc[idx].to_dict()
                            pred = svm_results[idx] if idx < len(svm_results) else {}

                            player = row_dict.get("Player", "Unknown")
                            label = row_dict.get("Strategy_Label", "")
                            raw_prob = (
                                (pred or {}).get("prob")
                                or (pred or {}).get("probability")
                                or (pred or {}).get("pred_prob")
                                or 0.0
                            )
                            try:
                                prob_val = float(raw_prob)
                            except Exception:
                                prob_val = 0.0

                            if player == "A":
                                probs_A.append(prob_val)
                                strat_index = len(probs_A)
                            elif player == "B":
                                probs_B.append(prob_val)
                                strat_index = len(probs_B)
                            else:
                                strat_index = idx + 1

                            graph_rows.append(
                                {
                                    "Player": player,
                                    "Strategy_Index": strat_index,
                                    "Strategy_Label": label,
                                    "SVM_Probability": prob_val,
                                }
                            )

                        graph_df = pd.DataFrame(graph_rows)

                        if graph_df.empty:
                            st.info(
                                "SVM could not produce probabilities for generated strategies."
                            )
                        else:
                            fig_ms = px.line(
                                graph_df,
                                x="SVM_Probability",
                                y="Strategy_Index",
                                color="Player",
                                markers=True,
                                hover_data=["Strategy_Label"],
                            )
                            fig_ms.update_layout(
                                title="Mixed-Strategy Prediction Level (after SVM classifier)",
                                xaxis_title="SVM predicted probability",
                                yaxis_title="Strategy (index per player)",
                            )
                            st.plotly_chart(fig_ms, use_container_width=True)

                            # ---------------------------------------
                            # 📊 Mixed-Strategy SVM Prediction Strength (%)
                            # ---------------------------------------
                            strength_A = _entropy_based_strength(np.array(probs_A))
                            strength_B = _entropy_based_strength(np.array(probs_B))
                            all_strengths = [
                                s for s in [strength_A, strength_B] if not np.isnan(s)
                            ]
                            overall_strength = (
                                float(np.nanmean(all_strengths)) if all_strengths else np.nan
                            )

                            st.markdown("### 📊 Mixed-Strategy SVM Prediction Strength (%)")
                            if not np.isnan(overall_strength):
                                st.success(
                                    f"**Overall SVM analysis power on strategies:** "
                                    f"{overall_strength:.2f} %"
                                )
                                if not np.isnan(strength_A):
                                    st.markdown(
                                        f"- Player A (rows) analysed strength: "
                                        f"**{strength_A:.2f}%**"
                                    )
                                if not np.isnan(strength_B):
                                    st.markdown(
                                        f"- Player B (columns) analysed strength: "
                                        f"**{strength_B:.2f}%**"
                                    )

                                st.markdown(
                                    """
This percentage tells **how strongly the SVM classifier is analysing your strategies**,  
computed as the **average classifier confidence across all mixed strategies**:

- Higher % → on average SVM is more confident about its predictions.  
- Lower % → predictions are more uncertain and less separable.  
                                    """
                                )
                            else:
                                st.info(
                                    "Not enough classifier probability information to compute "
                                    "a meaningful prediction strength."
                                )
                    except Exception as e:
                        st.warning(f"SVM-based prediction graph could not be built: {e}")
                else:
                    st.info("Train the SVM first (using the button above) to view prediction graph.")

            # ==================================================
            # Existing table display of generated strategies
            # ==================================================
            gen_df_display = gen_df[
                ["Player", "Strategy_Label", "Mapped_Strategy", "Mapped_Value", "Probability"]
            ].copy()
            gen_df_display = gen_df_display.rename(
                columns={
                    "Strategy_Label": "Strategy (Matrix Label)",
                    "Mapped_Strategy": "Mapped Strategy (Text)",
                    "Mapped_Value": "Mapped Value (0/0.5/1)",
                }
            )
            gen_df_display["Probability"] = gen_df_display["Probability"].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "-"
            )
            gen_df_display["Mapped Value (0/0.5/1)"] = gen_df_display[
                "Mapped Value (0/0.5/1)"
            ].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

            st.table(gen_df_display)

            # Download CSV of in-session generated strategies
            try:
                csv_buf = io.StringIO()
                gen_df.to_csv(csv_buf, index=False)
                csv_bytes = csv_buf.getvalue().encode("utf-8")
                st.download_button(
                    label="⬇️ Download generated strategies as CSV",
                    data=csv_bytes,
                    file_name="generated_strategies.csv",
                    mime="text/csv",
                    help="Download the solver-generated strategies table as CSV",
                )
            except Exception as e:
                st.warning(f"Could not prepare CSV for download: {e}")

            # DB operations: Save, Load
            if st.button(
                "💾 Save generated strategies to database", use_container_width=False
            ):
                try:
                    rows_to_insert = []
                    for r in gen_df.to_dict(orient="records"):
                        player = r.get("Player") or r.get("player")
                        strategy_label = (
                            r.get("Strategy_Label")
                            or r.get("strategy_label")
                            or r.get("Strategy (Matrix Label)")
                        )
                        mapped_strategy = (
                            r.get("Mapped_Strategy")
                            or r.get("mapped_strategy")
                            or r.get("Mapped Strategy (Text)")
                        )
                        mapped_value = (
                            r.get("Mapped_Value")
                            if "Mapped_Value" in r
                            else r.get("Mapped Value (0/0.5/1)")
                        )
                        prob_raw = r.get("Probability") or r.get("probability")
                        try:
                            prob_val = float(prob_raw)
                        except Exception:
                            prob_val = None
                        rows_to_insert.append(
                            {
                                "player": player,
                                "strategy_label": strategy_label,
                                "mapped_strategy": mapped_strategy,
                                "mapped_value": float(mapped_value)
                                if mapped_value is not None
                                and not (
                                    isinstance(mapped_value, float)
                                    and np.isnan(mapped_value)
                                )
                                else None,
                                "probability": prob_val,
                            }
                        )

                    with conn_factory()() as conn:
                        ensure_generated_strategies_table(conn)
                        inserted = save_generated_strategies(rows_to_insert, conn)
                        if inserted:
                            st.success(
                                f"Saved {inserted} row(s) to the database "
                                f"(generated_strategies)."
                            )
                            df_saved = fetch_all_generated_strategies(conn)
                            st.session_state["db_saved_strategies"] = df_saved
                        else:
                            st.info("No rows to insert.")
                except Exception as e:
                    st.error(f"Failed to save generated strategies to DB: {e}")

            if st.button("🔁 Load saved strategies from DB", use_container_width=False):
                try:
                    with conn_factory()() as conn:
                        ensure_generated_strategies_table(conn)
                        df_saved = fetch_all_generated_strategies(conn)
                        st.session_state["db_saved_strategies"] = df_saved
                        st.success(f"Loaded {len(df_saved)} row(s) from DB.")
                except Exception as e:
                    st.error(f"Failed to load saved strategies from DB: {e}")

            # ---------------------------
            # Classifier / Cleanup controls
            # ---------------------------
            st.divider()
            st.subheader("🧹 Cleanup: Remove 'I won't do business' rows (Safe Flow)")

            # If ML not available → show info card only
            if not UTIL_ML:
                st.info(
                    "ML classifier helpers are not available in `utils.py`. "
                    "To enable this section install **scikit-learn** and **joblib** "
                    "and use the updated `utils.py` (it provides training/predict helpers)."
                )
                st.caption("Required packages: scikit-learn, joblib")
                st.caption(f"Local test file (example): {LOCAL_UPLOADED_FILE_URL}")
            else:
                col1, col2, col3 = st.columns([1, 1, 1])

                # Slider in the middle
                with col2:
                    threshold = st.slider(
                        "Detection threshold (probability)",
                        0.5,
                        0.99,
                        0.80,
                        0.01,
                        help="Higher → stricter, fewer rows quarantined.",
                    )

                # Train button for cleanup / quarantine flow
                with col1:
                    if st.button(
                        "🔧 Train default SVM (quick) for cleanup", use_container_width=True
                    ):
                        try:
                            train_default_svm()
                            st.success(
                                "✅ Trained and saved default SVM pipeline "
                                "(strategy_svm_model.joblib)."
                            )

                            # After training, immediately run classifier and move flagged rows.
                            with conn_factory()() as conn:
                                ensure_generated_strategies_table(conn)
                                ensure_quarantine_table(conn)
                                preview_df = preview_removals_from_db(
                                    conn, threshold=threshold
                                )
                                if preview_df is None or preview_df.empty:
                                    st.info(
                                        f"No 'I won't do business' style rows "
                                        f"flagged at threshold {threshold:.2f}."
                                    )
                                else:
                                    ids = list(preview_df["id"].astype(int).tolist())
                                    moved = move_rows_to_quarantine(conn, ids)
                                    st.success(
                                        f"Automatically moved {moved} row(s) to "
                                        f"`quarantine_generated_strategies` "
                                        f"(threshold {threshold:.2f})."
                                    )
                                    # Refresh DB view in session
                                    df_saved = fetch_all_generated_strategies(conn)
                                    st.session_state["db_saved_strategies"] = df_saved
                        except Exception as e:
                            st.error(f"Failed to train model / auto-quarantine: {e}")

                # Manual preview button (if you want to inspect before moving)
                with col3:
                    if st.button("🔍 Preview flagged rows", use_container_width=True):
                        try:
                            with conn_factory()() as conn:
                                ensure_generated_strategies_table(conn)
                                preview_df = preview_removals_from_db(
                                    conn, threshold=threshold
                                )
                                st.session_state["preview_quarantine_df"] = preview_df
                                st.success(
                                    f"Previewed {len(preview_df)} flagged row(s)."
                                )
                        except Exception as e:
                            st.error(f"Preview failed: {e}")

                # Show preview if available and ML enabled
                if UTIL_ML and "preview_quarantine_df" in st.session_state:
                    st.markdown(
                        "**Preview of rows flagged for quarantine** "
                        "(inspect before confirming)"
                    )
                    df_prev = st.session_state["preview_quarantine_df"]
                    if df_prev.empty:
                        st.info(
                            "No rows flagged for quarantine with current threshold."
                        )
                    else:
                        st.dataframe(df_prev, width="stretch")
                        if st.button(
                            "✅ Confirm: Move flagged rows to quarantine",
                            use_container_width=False,
                        ):
                            try:
                                ids = list(df_prev["id"].astype(int).tolist())
                                with conn_factory()() as conn:
                                    ensure_quarantine_table(conn)
                                    moved = move_rows_to_quarantine(conn, ids)
                                    st.success(
                                        f"Moved {moved} row(s) to "
                                        f"`quarantine_generated_strategies`."
                                    )
                                    # refresh saved display
                                    df_saved = fetch_all_generated_strategies(conn)
                                    st.session_state["db_saved_strategies"] = df_saved
                                    # clear preview
                                    st.session_state.pop("preview_quarantine_df", None)
                            except Exception as e:
                                st.error(
                                    f"Failed to move rows to quarantine: {e}"
                                )

            # ---------------------------
            # Quarantine display + Restore / Delete UI
            # ---------------------------
            st.divider()
            st.subheader("🚨 Quarantine — Review, Restore or Permanently Delete")

            try:
                with conn_factory()() as conn:
                    ensure_quarantine_table(conn)
                    df_quar = fetch_quarantine_generated_strategies(conn)
            except Exception as e:
                st.error(f"Could not read quarantine table: {e}")
                df_quar = pd.DataFrame([])

            if df_quar is None or df_quar.empty:
                st.info("Quarantine is empty (no flagged rows).")
            else:
                display_quar = df_quar.copy()
                display_quar["selector_label"] = display_quar.apply(
                    lambda r: (
                        f"{int(r['id'])}: player={r.get('player','')}, "
                        f"label={r.get('strategy_label','')}, "
                        f"mapped={r.get('mapped_strategy','')}"
                    ),
                    axis=1,
                )
                st.dataframe(
                    display_quar.drop(columns=["selector_label"]),
                    width="stretch",
                )

                option_map = dict(
                    zip(
                        display_quar["selector_label"].tolist(),
                        display_quar["id"].tolist(),
                    )
                )
                selected_labels = st.multiselect(
                    "Select quarantined rows (by id) to restore or delete:",
                    options=list(option_map.keys()),
                    key="quar_select",
                )
                selected_ids = (
                    [int(option_map[s]) for s in selected_labels]
                    if selected_labels
                    else []
                )

                if selected_ids:
                    st.warning(
                        f"Selected quarantined IDs: {selected_ids} — choose an action "
                        "below. Restoring will insert rows back into "
                        "`generated_strategies`; deleting removes quarantine rows "
                        "permanently."
                    )

                    # Restore flow
                    with st.expander(
                        "Restore selected rows back into generated_strategies"
                    ):
                        st.markdown(
                            "This will re-insert the selected rows into "
                            "`generated_strategies` (columns: player, strategy_label, "
                            "mapped_strategy, mapped_value, probability) and remove "
                            "them from quarantine. This is **non-destructive**."
                        )
                        confirm_restore = st.text_input(
                            "Type RESTORE to confirm restore of selected rows",
                            key="confirm_restore",
                        )
                        if st.button("🔁 Restore selected rows (safe)"):
                            if confirm_restore != "RESTORE":
                                st.error(
                                    "Type exactly RESTORE in the confirmation box to "
                                    "proceed. No action taken."
                                )
                            else:
                                try:
                                    restored = 0
                                    with conn_factory()() as conn:
                                        ensure_generated_strategies_table(conn)
                                        ensure_quarantine_table(conn)
                                        cur = conn.cursor()
                                        for qid in selected_ids:
                                            cur.execute(
                                                """
                                                SELECT original_id, player,
                                                       strategy_label, mapped_strategy,
                                                       mapped_value, probability
                                                FROM quarantine_generated_strategies
                                                WHERE id = ?
                                                """,
                                                (int(qid),),
                                            )
                                            r = cur.fetchone()
                                            if not r:
                                                continue
                                            (
                                                _orig_id,
                                                player,
                                                strat_label,
                                                mapped_strategy,
                                                mapped_value,
                                                prob,
                                            ) = r
                                            cur.execute(
                                                """
                                                INSERT INTO generated_strategies
                                                    (player, strategy_label,
                                                     mapped_strategy, mapped_value,
                                                     probability)
                                                VALUES (?, ?, ?, ?, ?)
                                                """,
                                                (
                                                    player,
                                                    strat_label,
                                                    mapped_strategy,
                                                    mapped_value,
                                                    prob,
                                                ),
                                            )
                                            cur.execute(
                                                "DELETE FROM "
                                                "quarantine_generated_strategies "
                                                "WHERE id = ?",
                                                (int(qid),),
                                            )
                                            restored += 1
                                        conn.commit()
                                    st.success(
                                        f"Restored {restored} row(s) to "
                                        "generated_strategies and removed them from "
                                        "quarantine."
                                    )
                                    with conn_factory()() as conn2:
                                        st.session_state[
                                            "db_saved_strategies"
                                        ] = fetch_all_generated_strategies(conn2)
                                    st.session_state.pop("quar_select", None)
                                except Exception as e:
                                    st.error(f"Restore failed: {e}")

                    # Permanent delete flow
                    with st.expander(
                        "Permanently delete selected quarantined rows"
                    ):
                        st.markdown(
                            "This is **destructive**. It will permanently remove the "
                            "quarantined rows. If original `generated_strategies` "
                            "rows still exist (unlikely), they will also be removed "
                            "if present. Proceed carefully."
                        )
                        confirm_delete = st.text_input(
                            "Type DELETE to confirm permanent deletion of selected rows",
                            key="confirm_delete_quar",
                        )
                        if st.button(
                            "🗑️ Permanently delete selected quarantined rows"
                        ):
                            if confirm_delete != "DELETE":
                                st.error(
                                    "Type exactly DELETE in the confirmation box to "
                                    "proceed. No action taken."
                                )
                            else:
                                try:
                                    deleted = 0
                                    with conn_factory()() as conn:
                                        cur = conn.cursor()
                                        for qid in selected_ids:
                                            cur.execute(
                                                "SELECT original_id FROM "
                                                "quarantine_generated_strategies "
                                                "WHERE id = ?",
                                                (int(qid),),
                                            )
                                            row = cur.fetchone()
                                            if row and row[0] is not None:
                                                orig_id = int(row[0])
                                                try:
                                                    cur.execute(
                                                        "DELETE FROM "
                                                        "generated_strategies "
                                                        "WHERE id = ?",
                                                        (orig_id,),
                                                    )
                                                except Exception:
                                                    pass
                                            cur.execute(
                                                "DELETE FROM "
                                                "quarantine_generated_strategies "
                                                "WHERE id = ?",
                                                (int(qid),),
                                            )
                                            deleted += 1
                                        conn.commit()
                                    st.success(
                                        f"Permanently deleted {deleted} "
                                        "quarantine row(s)."
                                    )
                                    st.session_state.pop("quar_select", None)
                                except Exception as e:
                                    st.error(f"Permanent deletion failed: {e}")
                else:
                    st.info(
                        "Select one or more quarantined rows to enable "
                        "Restore / Delete actions."
                    )

        except Exception as e:
            st.error(f"Error showing generated strategies: {e}")

    st.caption(
        "💡 Tip: Use 'Preview flagged rows' and inspect before confirming "
        "quarantine. Quarantine keeps a trace (original_id) for auditing."
    )
