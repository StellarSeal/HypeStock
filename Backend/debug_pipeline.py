import numpy as np
import pandas as pd

from feature_pipeline import FEATURE_SCHEMA, normalize_features


def compare_csv_vs_db(csv_df: pd.DataFrame, db_df: pd.DataFrame) -> None:
    csv_norm = normalize_features(csv_df)
    db_norm = normalize_features(db_df)

    overlap = min(len(csv_norm), len(db_norm))
    if overlap == 0:
        print("No overlap between CSV and DB frames.")
        return

    csv_tail = csv_norm.tail(overlap).reset_index(drop=True)
    db_tail = db_norm.tail(overlap).reset_index(drop=True)

    for col in FEATURE_SCHEMA:
        diff = np.abs(csv_tail[col].to_numpy() - db_tail[col].to_numpy())
        print(f"{col}: mean={diff.mean()}, max={diff.max()}")

    print("CSV tail:\n", csv_tail.tail(10))
    print("DB tail:\n", db_tail.tail(10))
