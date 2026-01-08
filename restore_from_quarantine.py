# restore_from_quarantine.py
import sqlite3
import utils

# list quarantine row ids (the quarantine table primary-key id values)
quarantine_ids_to_restore = [7,6,5,4,3,2,1]  # example - use the ids shown in your verify output or pick subset

with utils.get_conn() as conn:
    cur = conn.cursor()
    restored = 0
    for qid in quarantine_ids_to_restore:
        cur.execute("""
            SELECT original_id, player, strategy_label, mapped_strategy, mapped_value, probability
            FROM quarantine_generated_strategies
            WHERE id = ?
        """, (int(qid),))
        row = cur.fetchone()
        if not row:
            continue
        original_id, player, strategy_label, mapped_strategy, mapped_value, prob = row
        # Insert back into generated_strategies
        cur.execute("""
            INSERT INTO generated_strategies (player, strategy_label, mapped_strategy, mapped_value, probability)
            VALUES (?, ?, ?, ?, ?)
        """, (player, strategy_label, mapped_strategy, mapped_value, prob))
        # delete from quarantine
        cur.execute("DELETE FROM quarantine_generated_strategies WHERE id = ?", (int(qid),))
        restored += 1
    conn.commit()
print(f"Restored {restored} rows from quarantine.")
