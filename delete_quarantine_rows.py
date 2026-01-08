# delete_quarantine_rows.py
import utils

# List of quarantine row ids (primary keys in quarantine_generated_strategies)
quarantine_ids = [7, 6, 5, 4, 3, 2, 1]  # adjust as needed

with utils.get_conn() as conn:
    cur = conn.cursor()
    cur.executemany(
        "DELETE FROM quarantine_generated_strategies WHERE id = ?",
        [(int(i),) for i in quarantine_ids]
    )
    conn.commit()
    print("Deleted rows:", quarantine_ids)
