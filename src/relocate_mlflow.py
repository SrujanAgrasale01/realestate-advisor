"""Re-point MLflow artifact paths after the project folder has been moved.

MLflow stores absolute artifact locations in mlflow.db.  If you copy or unzip
this project somewhere else, run once:   python -m src.relocate_mlflow
(only needed to open model artefacts in the MLflow UI / load models from the
registry; metrics, parameters and the Streamlit app work regardless).
"""
import sqlite3

from . import config as C


def main() -> None:
    if not C.MLFLOW_DB.exists():
        print("No mlflow.db found - run `python -m src.train` first.")
        return
    new_root = C.MLFLOW_ARTIFACTS.as_uri()
    con = sqlite3.connect(C.MLFLOW_DB)
    cur = con.cursor()
    cur.execute("SELECT artifact_location FROM experiments WHERE artifact_location LIKE '%mlartifacts%' LIMIT 1")
    row = cur.fetchone()
    if row is None:
        print("No experiment with a local artifact location found.")
        return
    loc = row[0]
    old_root = loc[: loc.index("mlartifacts") + len("mlartifacts")]
    if old_root == new_root:
        print("MLflow paths already point at this folder.")
        return
    changed = 0
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    for table in tables:
        info = cur.execute(f'PRAGMA table_info("{table}")').fetchall()
        for col in [c[1] for c in info if any(t in (c[2] or "").upper() for t in ("CHAR", "TEXT", "CLOB"))]:
            cur.execute(f'UPDATE "{table}" SET "{col}" = REPLACE("{col}", ?, ?) WHERE instr("{col}", ?) > 0',
                        (old_root, new_root, old_root))
            changed += cur.rowcount
    con.commit()
    con.close()
    print(f"Updated {changed} MLflow records: {old_root} -> {new_root}")


if __name__ == "__main__":
    main()
