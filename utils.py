# utils.py — High-Confidence SVM Version (Option-C2, FINAL)

from pathlib import Path
import os
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Iterable, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Try to import ML tools (optional)
# -------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from joblib import dump, load
    SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    Pipeline = None
    SVC = None
    dump = None
    load = None
    SKLEARN_AVAILABLE = False


# -------------------------------------------------------------------
# Paths & DB connection
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "game_theory.db"

RESULTS_TABLE = "results"
LOSSES_TABLE = "losses"
GEN_STRATS_TABLE = "generated_strategies"
QUARANTINE_TABLE = "quarantine_generated_strategies"

TRAIN_CSV_PATH = BASE_DIR / "svm_training_data.csv"
MODEL_PATH = BASE_DIR / "strategy_svm_model.joblib"

A_LABELS = ["A1", "A2", "A3"]
B_LABELS = ["B1", "B2", "B3"]


@contextmanager
def get_conn(db_path: Path = DB_PATH):
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


# -------------------------------------------------------------------
# Data Loading — Excel / CSV
# -------------------------------------------------------------------
def load_data_file(file_or_path) -> pd.DataFrame:
    """
    Load Excel/CSV into DataFrame.
    Normalizes to: ["Supplier_Name", "Supplier_Profit"].
    """
    if file_or_path is None:
        return pd.DataFrame(columns=["Supplier_Name", "Supplier_Profit"])

    try:
        # Streamlit Upload
        if hasattr(file_or_path, "read"):
            name = getattr(file_or_path, "name", "")
            if name.lower().endswith(".csv"):
                df = pd.read_csv(file_or_path)
            else:
                df = pd.read_excel(file_or_path)
        else:
            # File path
            p = Path(file_or_path)
            if not p.exists():
                return pd.DataFrame(columns=["Supplier_Name", "Supplier_Profit"])

            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
    except Exception:
        return pd.DataFrame(columns=["Supplier_Name", "Supplier_Profit"])

    cols = {c.lower(): c for c in df.columns}

    # Supplier Name
    name_col = cols.get("supplier_name") or cols.get("name")
    if name_col is None:
        df["Supplier_Name"] = df.iloc[:, 0].astype(str)
    else:
        df.rename(columns={name_col: "Supplier_Name"}, inplace=True)

    # Supplier Profit
    profit_col = cols.get("supplier_profit") or cols.get("profit")
    if profit_col is None:
        if df.shape[1] > 1:
            df["Supplier_Profit"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        else:
            df["Supplier_Profit"] = 0.0
    else:
        df.rename(columns={profit_col: "Supplier_Profit"}, inplace=True)
        df["Supplier_Profit"] = pd.to_numeric(df["Supplier_Profit"], errors="coerce")

    return df[["Supplier_Name", "Supplier_Profit"]].copy()


# -------------------------------------------------------------------
# Payoff Matrix + Saddle points
# -------------------------------------------------------------------
def generate_payoff_matrix(
    data_df: pd.DataFrame, rows: int = 3, cols: int = 3
) -> pd.DataFrame:
    """
    Generate a 3×3 payoff matrix using Supplier_Profit values.
    If no profit column → random.
    """
    if data_df is None or "Supplier_Profit" not in data_df.columns:
        mat = np.random.randint(-20, 60, size=(rows, cols))
    else:
        pool = data_df["Supplier_Profit"].dropna().values
        if pool.size == 0:
            pool = np.array([10, -5, 20, -15])

        sampled = np.random.choice(pool, size=(rows * cols), replace=True)
        mat = sampled.reshape(rows, cols).astype(int)

    return pd.DataFrame(mat, index=A_LABELS[:rows], columns=B_LABELS[:cols])


def find_saddle_points(payoff_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Find maximin = minimax saddle points.
    """
    df = payoff_df
    min_in_rows = df.min(axis=1)
    maximin = min_in_rows.max()

    max_in_cols = df.max(axis=0)
    minimax = max_in_cols.min()

    sps: List[Dict[str, Any]] = []
    if maximin == minimax:
        val = maximin
        for r in df.index:
            for c in df.columns:
                if df.loc[r, c] == val:
                    sps.append({"rows": r, "cols": c, "value": val})
    return sps


def highlight_losses(val: Any) -> str:
    """
    Red color for negative integers.
    """
    try:
        if float(val) < 0:
            return "color: red; font-weight: bold;"
    except Exception:
        return ""
    return ""


# -------------------------------------------------------------------
# DB Table Creation Helpers
# -------------------------------------------------------------------
def ensure_results_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sl_no INTEGER,
            supplier1 TEXT,
            supplier2 TEXT,
            saddle_point TEXT,
            supplier1_strategy TEXT,
            supplier2_strategy TEXT,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def ensure_losses_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LOSSES_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sl_no INTEGER,
            supplier1 TEXT,
            supplier2 TEXT,
            loss_value REAL,
            source TEXT,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def ensure_generated_strategies_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {GEN_STRATS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player TEXT,
            strategy_label TEXT,
            mapped_strategy TEXT,
            mapped_value REAL,
            probability REAL,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def ensure_quarantine_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id INTEGER,
            player TEXT,
            strategy_label TEXT,
            mapped_strategy TEXT,
            mapped_value REAL,
            probability REAL,
            flagged_reason TEXT,
            confidence REAL,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


# -------------------------------------------------------------------
# Insert Results & Losses
# -------------------------------------------------------------------
def save_results_df(results_df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    ensure_results_table(conn)
    rows = []
    for _, r in results_df.iterrows():
        rows.append(
            (
                int(r.get("SL_No", 0)),
                str(r.get("Supplier1", "")),
                str(r.get("Supplier2", "")),
                str(r.get("Saddle_Point", "")),
                str(r.get("Supplier1_Strategy", "")),
                str(r.get("Supplier2_Strategy", "")),
            )
        )
    conn.executemany(
        f"""
        INSERT INTO {RESULTS_TABLE}
            (sl_no, supplier1, supplier2, saddle_point,
             supplier1_strategy, supplier2_strategy)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def save_losses(losses_list: List[List[Any]], conn: sqlite3.Connection) -> int:
    ensure_losses_table(conn)
    if not losses_list:
        return 0
    rows = []
    for sl_no, s1, s2, loss_val, source in losses_list:
        rows.append((int(sl_no), str(s1), str(s2), float(loss_val), str(source)))
    conn.executemany(
        f"""
        INSERT INTO {LOSSES_TABLE}
            (sl_no, supplier1, supplier2, loss_value, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


# -------------------------------------------------------------------
# Fetch DB Data
# -------------------------------------------------------------------
def fetch_all_results(conn: sqlite3.Connection) -> pd.DataFrame:
    ensure_results_table(conn)
    return pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE} ORDER BY id", conn)


def fetch_all_losses(conn: sqlite3.Connection) -> pd.DataFrame:
    ensure_losses_table(conn)
    return pd.read_sql_query(f"SELECT * FROM {LOSSES_TABLE} ORDER BY id", conn)


def fetch_all_generated_strategies(conn: sqlite3.Connection) -> pd.DataFrame:
    ensure_generated_strategies_table(conn)
    return pd.read_sql_query(f"SELECT * FROM {GEN_STRATS_TABLE} ORDER BY id", conn)


def fetch_quarantine_generated_strategies(conn: sqlite3.Connection) -> pd.DataFrame:
    ensure_quarantine_table(conn)
    return pd.read_sql_query(f"SELECT * FROM {QUARANTINE_TABLE} ORDER BY id", conn)


def fetch_all_quarantine(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Alias expected by pure_strategy to fetch quarantine table.
    """
    return fetch_quarantine_generated_strategies(conn)


# -------------------------------------------------------------------
# Clear Tables
# -------------------------------------------------------------------
def clear_results_table(conn: sqlite3.Connection) -> None:
    ensure_results_table(conn)
    conn.execute(f"DELETE FROM {RESULTS_TABLE}")
    conn.commit()


def clear_losses_table(conn: sqlite3.Connection) -> None:
    ensure_losses_table(conn)
    conn.execute(f"DELETE FROM {LOSSES_TABLE}")
    conn.commit()


def clear_quarantine(conn: sqlite3.Connection) -> None:
    ensure_quarantine_table(conn)
    conn.execute(f"DELETE FROM {QUARANTINE_TABLE}")
    conn.commit()


# -------------------------------------------------------------------
# Generated Strategies Save
# -------------------------------------------------------------------
def save_generated_strategies(
    rows: Iterable[Dict[str, Any]], conn: sqlite3.Connection
) -> int:
    ensure_generated_strategies_table(conn)
    prepared_rows = []
    for r in rows:
        prepared_rows.append(
            (
                r.get("player"),
                r.get("strategy_label"),
                r.get("mapped_strategy"),
                r.get("mapped_value"),
                r.get("probability"),
            )
        )
    conn.executemany(
        f"""
        INSERT INTO {GEN_STRATS_TABLE}
            (player, strategy_label, mapped_strategy, mapped_value, probability)
        VALUES (?, ?, ?, ?, ?)
        """,
        prepared_rows,
    )
    conn.commit()
    return len(prepared_rows)


# -------------------------------------------------------------------
# Move flagged rows to quarantine table
# -------------------------------------------------------------------
def move_rows_to_quarantine(conn: sqlite3.Connection, ids: List[int]) -> int:
    """
    Move rows from generated_strategies → quarantine_generated_strategies.
    """
    if not ids:
        return 0

    ensure_generated_strategies_table(conn)
    ensure_quarantine_table(conn)

    cur = conn.cursor()
    moved = 0

    for rid in ids:
        cur.execute(
            f"""
            SELECT id, player, strategy_label, mapped_strategy,
                   mapped_value, probability
            FROM {GEN_STRATS_TABLE}
            WHERE id = ?
            """,
            (int(rid),),
        )
        row = cur.fetchone()
        if not row:
            continue

        (
            original_id,
            player,
            strategy_label,
            mapped_strategy,
            mapped_value,
            prob,
        ) = row

        cur.execute(
            f"""
            INSERT INTO {QUARANTINE_TABLE}
                (original_id, player, strategy_label,
                 mapped_strategy, mapped_value, probability,
                 flagged_reason, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                original_id,
                player,
                strategy_label,
                mapped_strategy,
                mapped_value,
                prob,
                "auto",
                1.0,
            ),
        )

        cur.execute(
            f"DELETE FROM {GEN_STRATS_TABLE} WHERE id = ?",
            (int(rid),),
        )

        moved += 1

    conn.commit()
    return moved


# -------------------------------------------------------------------
# EXTRA: BOOSTED default dataset (large) for strong margin
# -------------------------------------------------------------------
def _build_boosted_training_df() -> pd.DataFrame:
    """
    Large boosted dataset to achieve high-confidence separation.
    Positive = 1 (will work with you / business)
    Negative = 0 (won't do business)
    """
    pos = [
        "I will do business", "We will do business", "Yes we will do business",
        "Interested in business", "Very much interested in business",
        "Happy to collaborate", "Ready to collaborate", "Open to doing business",
        "We accept the business", "We are ready to start business",
        "I agree to business", "We agree to work with you",
        "We accept your proposal", "Business partnership accepted",
        "50-50 choice", "maybe 50 50", "neutral 50-50", "mixed strategy 50-50",
        "partially interested", "slightly interested",
        "positive business decision", "likely to do business",
        "interested to move forward", "willing to take next step"
    ] * 6  # repeat to strengthen margin

    neg = [
        "I won't do business", "We won't do business", "we will not do business",
        "Not interested in business", "Do not want business",
        "We reject the business proposal", "We refuse the proposal",
        "No we are not ready", "Business collaboration not possible",
        "We decline your proposal", "We do not accept the business offer",
        "We are not interested to work with you", "do not contact us",
        "We cannot do business", "Willingness = no", "no business",
        "we are not moving forward", "we reject collaboration",
        "business refused", "completely not interested"
    ] * 6  # repeat

    df = pd.DataFrame({
        "text": pos + neg,
        "label": [1] * len(pos) + [0] * len(neg)
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# TRAIN: High-confidence Hybrid SVM
# -------------------------------------------------------------------
def train_strategy_svm_from_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label"
) -> Tuple[Path, Path]:
    """
    High-margin boosted SVM.
    Uses:
    - TFIDF 1–3 grams
    - Linear SVC with C=15
    - probability=True
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for SVM training.")

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("svc", SVC(kernel="linear", C=15, probability=True))
    ])

    pipeline.fit(texts, labels)
    dump(pipeline, MODEL_PATH)
    return MODEL_PATH, MODEL_PATH


# -------------------------------------------------------------------
# TRAIN DEFAULT SVM (uses CSV if exists OR boosted dataset)
# -------------------------------------------------------------------
def train_default_svm() -> None:
    """
    Trains the SVM using the external CSV if available.
    Otherwise, uses large boosted dataset.
    """
    if TRAIN_CSV_PATH.exists():
        df = pd.read_csv(TRAIN_CSV_PATH, encoding="utf-8")

        cols = {c.lower(): c for c in df.columns}
        if "text" not in cols or "label" not in cols:
            raise ValueError("svm_training_data.csv must have 'text' and 'label' columns.")

        df.rename(columns={cols["text"]: "text", cols["label"]: "label"}, inplace=True)

        df["label"] = (
            df["label"].astype(str)
            .str.replace('"', "")
            .str.strip()
        )
        df["label"] = df["label"].replace({"0": 0, "1": 1})
        df["label"] = df["label"].astype(int)

        df = df.dropna(subset=["text", "label"])
    else:
        df = _build_boosted_training_df()

    train_strategy_svm_from_dataframe(df)


# -------------------------------------------------------------------
# LOAD TRAINED SVM MODEL
# -------------------------------------------------------------------
def load_strategy_model():
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn not installed.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return load(MODEL_PATH)


def load_strategy_svm():
    """
    Alias expected by mixed_strategy.py
    """
    return load_strategy_model()


# -------------------------------------------------------------------
# PREDICT WITH SVM
# -------------------------------------------------------------------
def predict_strategy_labels(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Predict probabilities and labels.
    prob = P(class = 0) → probability of "won't do business".
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn not installed.")

    model = load_strategy_model()
    preds = model.predict(texts)

    try:
        proba = model.predict_proba(texts)
        classes = list(model.classes_)
        idx0 = classes.index(0)
        probs0 = proba[:, idx0]
    except Exception:
        probs0 = np.array([1.0 if p == 0 else 0.0 for p in preds])

    out = []
    for t, p, pr in zip(texts, preds, probs0):
        out.append({"text": t, "pred": int(p), "prob": float(pr)})
    return out


def predict_strategy_texts(texts: List[str]) -> pd.DataFrame:
    """
    Wrapper returning DataFrame for mixed_strategy.py
    """
    return pd.DataFrame(predict_strategy_labels(texts))


# -------------------------------------------------------------------
# AUTO-LABEL (fallback)
# -------------------------------------------------------------------
def auto_label_from_texts(texts: List[str]) -> List[int]:
    """
    Rule-based fallback classifier.
    0 = won't do business
    1 = will / 50-50 / keep
    2 = unknown
    """
    labels: List[int] = []
    for t in texts:
        s = str(t).lower()

        if any(x in s for x in ["won't", "will not", "do not", "dont do", "don't do",
                                "no business", "reject", "refuse", "decline",
                                "not interested"]):
            labels.append(0)
        elif any(x in s for x in ["do business", "interested", "yes", "accept",
                                  "50-50", "50 50", "50/50", "neutral"]):
            labels.append(1)
        else:
            labels.append(2)
    return labels


# -------------------------------------------------------------------
# PREVIEW: Which rows should go to quarantine?
#   • Strong SVM (prob >= threshold)
#   • Rule-based "won't do business" detector
# -------------------------------------------------------------------
def preview_removals_from_db(conn: sqlite3.Connection, threshold: float = 0.80) -> pd.DataFrame:
    """
    Reads generated_strategies and returns rows to be quarantined.

    A row is flagged if:
      - SVM predicts class=0 (won't do business) AND prob >= threshold
      - OR rule-based text inspection detects negative pattern
    """
    ensure_generated_strategies_table(conn)
    df = fetch_all_generated_strategies(conn)

    if df is None or df.empty:
        return pd.DataFrame([])

    # Choose text source
    if "mapped_strategy" in df.columns:
        texts = df["mapped_strategy"].fillna("").astype(str)
    elif "strategy_label" in df.columns:
        texts = df["strategy_label"].fillna("").astype(str)
    else:
        texts = df.iloc[:, 0].astype(str).fillna("")

    combined = df.copy()

    combined["pred_label"] = -1
    combined["pred_prob"] = 0.0

    ml_mask = pd.Series(False, index=combined.index)

    # ML prediction
    if SKLEARN_AVAILABLE and MODEL_PATH.exists():
        try:
            pred_df = predict_strategy_texts(texts.tolist())
            combined["pred_label"] = pred_df["pred"].values
            combined["pred_prob"] = pred_df["prob"].values

            ml_mask = (combined["pred_label"] == 0) & (combined["pred_prob"] >= threshold)
        except Exception:
            pass  # Ignore SVM failure, rely only on rules

    # RULE-BASED negative text check
    lower = texts.str.lower()

    rule_mask = (
        lower.str.contains("won't do", na=False)
        | lower.str.contains("will not", na=False)
        | lower.str.contains("do not", na=False)
        | lower.str.contains("dont do", na=False)
        | lower.str.contains("don't do", na=False)
        | lower.str.contains("no business", na=False)
        | lower.str.contains("reject", na=False)
        | lower.str.contains("refuse", na=False)
        | lower.str.contains("decline", na=False)
        | lower.str.contains("not interested", na=False)
    )

    final_mask = ml_mask | rule_mask

    flagged = combined.loc[final_mask].copy()
    return flagged
