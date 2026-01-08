# move_flagged_to_quarantine.py  — rule-based version
import sqlite3
import pandas as pd
import utils


def get_connection():
    """Match app logic: use utils.DB_PATH if set, else local game_theory.db."""
    if getattr(utils, "DB_PATH", None) is not None:
        return utils.get_conn(utils.DB_PATH)
    return sqlite3.connect("game_theory.db")


def main():
    with get_connection() as conn:
        # make sure tables exist
        utils.ensure_generated_strategies_table(conn)
        utils.ensure_quarantine_table(conn)

        # Select all rows where mapped_strategy = "I won't do business"
        query = """
        SELECT *
        FROM generated_strategies
        WHERE mapped_strategy = 'I won''t do business';
        """
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("No 'I won't do business' rows found in generated_strategies.")
            return

        ids = list(df["id"].astype(int).tolist())
        print("Moving these ids to quarantine:", ids)

        moved = utils.move_rows_to_quarantine(conn, ids)
        print("Moved", moved, "rows to quarantine_generated_strategies")


if __name__ == "__main__":
    main()
