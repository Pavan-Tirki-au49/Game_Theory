# verify.py — Quick DB inspection helper for game_theory.db

import utils


def main():
    with utils.get_conn() as conn:
        print("\nRemaining generated_strategies:")
        print(utils.fetch_all_generated_strategies(conn).to_string(index=False))

        print("\nQuarantine table:")
        print(utils.fetch_quarantine_generated_strategies(conn).to_string(index=False))


if __name__ == "__main__":
    main()
