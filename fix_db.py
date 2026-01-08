# fix_db.py — Safe Database Reset Script (for Clothing Supplier Game Theory Project)
"""
Safely ensure / reset database tables including generated_strategies and quarantine.
Usage:
    python fix_db.py --ensure-only
    python fix_db.py --drop-and-reset
    python fix_db.py --ensure-only --seed-sample
"""

import sqlite3
import argparse
import os
from typing import List, Dict, Any, Iterable

# Try to import helpers from utils; provide sensible fallbacks if missing.
try:
    from utils import (
        get_conn,
        ensure_results_table,
        ensure_losses_table,
        ensure_generated_strategies_table,
        save_generated_strategies,
        ensure_quarantine_table,
    )
except Exception:
    get_conn = None
    ensure_results_table = None
    ensure_losses_table = None
    ensure_generated_strategies_table = None
    save_generated_strategies = None
    ensure_quarantine_table = None

# Fallback DB path (used when utils.get_conn is not available)
FALLBACK_DB_PATH = os.path.join(os.path.dirname(__file__), "game_theory.db")


def fallback_get_conn() -> sqlite3.Connection:
    return sqlite3.connect(FALLBACK_DB_PATH)


def drop_table_if_exists(conn: sqlite3.Connection, table_name: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    if cur.fetchone():
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()


def ensure_generated_strategies_table_local(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS generated_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player TEXT NOT NULL,
            strategy_label TEXT NOT NULL,
            mapped_strategy TEXT,
            mapped_value REAL,
            probability REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_generated_strategies_player_created
        ON generated_strategies (player, created_at);
        """
    )
    conn.commit()


def save_generated_strategies_local(
    conn: sqlite3.Connection, rows: Iterable[Dict[str, Any]]
) -> int:
    cur = conn.cursor()
    inserted = 0
    sql = """
        INSERT INTO generated_strategies
            (player, strategy_label, mapped_strategy, mapped_value, probability)
        VALUES (?, ?, ?, ?, ?);
    """
    try:
        for r in rows:
            if isinstance(r, dict):
                player = r.get("player")
                strategy_label = r.get("strategy_label")
                mapped_strategy = r.get("mapped_strategy")
                mapped_value = r.get("mapped_value")
                probability = r.get("probability")
            else:
                player, strategy_label, mapped_strategy, mapped_value, probability = r
            if player is None or strategy_label is None:
                continue
            cur.execute(
                sql,
                (player, strategy_label, mapped_strategy, mapped_value, probability),
            )
            inserted += 1
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise
    return inserted


def ensure_quarantine_table_local(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quarantine_generated_strategies (
            qid INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id INTEGER,
            player TEXT,
            strategy_label TEXT,
            mapped_strategy TEXT,
            mapped_value REAL,
            probability REAL,
            quarantined_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()


def reset_database(drop_and_reset: bool = False, seed_sample: bool = False) -> None:
    conn_factory = get_conn if get_conn is not None else fallback_get_conn

    with conn_factory() as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cur.fetchall()]
        print("🧩 Checking existing tables in database.")
        print(f"    Existing tables: {existing_tables}")

        if drop_and_reset:
            for t in (
                "results_table",
                "losses_table",
                "generated_strategies",
                "quarantine_generated_strategies",
            ):
                if t in existing_tables:
                    print(f"🗑️ Dropping old '{t}'.")
                    drop_table_if_exists(conn, t)
            print("🔁 Dropped selected tables (drop_and_reset requested).")

        # Ensure results_table and losses_table (via utils or fallback)
        print("🔧 Ensuring 'results_table' and 'losses_table' exist (utils or minimal).")
        if ensure_results_table is not None:
            try:
                ensure_results_table(conn)
                print("✅ ensure_results_table executed (from utils).")
            except Exception as e:
                print(f"⚠️ utils.ensure_results_table failed: {e}")
        else:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS results_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    result_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

        if ensure_losses_table is not None:
            try:
                ensure_losses_table(conn)
                print("✅ ensure_losses_table executed (from utils).")
            except Exception as e:
                print(f"⚠️ utils.ensure_losses_table failed: {e}")
        else:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS losses_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    loss_value REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

        # Ensure generated_strategies
        print("🔧 Ensuring 'generated_strategies' table exists.")
        if ensure_generated_strategies_table is not None:
            try:
                ensure_generated_strategies_table(conn)
                print("✅ ensure_generated_strategies_table executed (from utils).")
            except Exception as e:
                print(
                    f"⚠️ utils.ensure_generated_strategies_table failed: {e} — using local creation."
                )
                ensure_generated_strategies_table_local(conn)
        else:
            ensure_generated_strategies_table_local(conn)

        # Ensure quarantine table
        print("🔧 Ensuring 'quarantine_generated_strategies' table exists.")
        if ensure_quarantine_table is not None:
            try:
                ensure_quarantine_table(conn)
                print("✅ ensure_quarantine_table executed (from utils).")
            except Exception as e:
                print(
                    f"⚠️ utils.ensure_quarantine_table failed: {e} — using local creation."
                )
                ensure_quarantine_table_local(conn)
        else:
            ensure_quarantine_table_local(conn)

        print("\n✅ Database tables ensured/recreated successfully.")
        print("   - results_table (stores simulation outcomes)")
        print("   - losses_table (stores negative payoff/loss records)")
        print("   - generated_strategies (stores solver-generated strategy distributions)")
        print("   - quarantine_generated_strategies (stores quarantined rows)\n")

        if seed_sample:
            sample_rows = [
                {
                    "player": "A",
                    "strategy_label": "A1",
                    "mapped_strategy": "I will do business",
                    "mapped_value": 1.0,
                    "probability": 0.823,
                },
                {
                    "player": "A",
                    "strategy_label": "A2",
                    "mapped_strategy": "50-50",
                    "mapped_value": 0.5,
                    "probability": 0.177,
                },
                {
                    "player": "B",
                    "strategy_label": "B1",
                    "mapped_strategy": "I won't do business",
                    "mapped_value": 0.0,
                    "probability": 0.400,
                },
                {
                    "player": "B",
                    "strategy_label": "B2",
                    "mapped_strategy": "I will do business",
                    "mapped_value": 1.0,
                    "probability": 0.600,
                },
            ]
            if save_generated_strategies is not None:
                try:
                    inserted = save_generated_strategies(conn, sample_rows)
                    print(
                        f"🧪 Seeded {inserted} sample rows into 'generated_strategies' (utils.save_generated_strategies)."
                    )
                except Exception as e:
                    print(
                        f"⚠️ utils.save_generated_strategies failed: {e} — falling back to local saver."
                    )
                    inserted = save_generated_strategies_local(conn, sample_rows)
                    print(
                        f"🧪 Seeded {inserted} sample rows into 'generated_strategies' (local)."
                    )
            else:
                inserted = save_generated_strategies_local(conn, sample_rows)
                print(
                    f"🧪 Seeded {inserted} sample rows into 'generated_strategies' (local)."
                )


def main():
    parser = argparse.ArgumentParser(
        description="Reset/ensure the project database tables safely."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ensure-only",
        action="store_true",
        help="Ensure tables exist (don't drop existing).",
    )
    group.add_argument(
        "--drop-and-reset",
        action="store_true",
        help="Drop known tables and recreate (destructive).",
    )
    parser.add_argument(
        "--seed-sample",
        action="store_true",
        help="Seed a small sample in generated_strategies after ensuring.",
    )
    args = parser.parse_args()

    try:
        if args.drop_and_reset:
            print("➡️ Running in drop-and-reset mode (destructive).")
            reset_database(drop_and_reset=True, seed_sample=args.seed_sample)
        else:
            print("➡️ Running in ensure-only mode (non-destructive).")
            reset_database(drop_and_reset=False, seed_sample=args.seed_sample)
    except sqlite3.Error as e:
        print(f"❌ SQLite Error: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected Error: {e}")


if __name__ == "__main__":
    main()
