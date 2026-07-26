import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _add_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = (
        out["medication"].fillna("").astype(str)
        + " || "
        + out["conditions"].fillna("").astype(str)
        + " || "
        + out["allergies"].fillna("").astype(str)
    )
    return out


def train_and_save(data_dir: Path, out_path: Path) -> None:
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run prepare_synthea_dataset.py in simulate mode first.")

    df_train = pd.read_csv(train_path)
    y_train = df_train.pop("label")
    X_train = _add_text_feature(df_train)

    preprocess = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(min_df=2, ngram_range=(1, 2)), "text"),
            ("gender", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
            ("age", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), ["age"]),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    clf.fit(X_train, y_train)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/synthea_processed", help="Folder containing train.csv")
    p.add_argument("--out", default="models/alert_model.joblib", help="Output model file")
    args = p.parse_args()

    train_and_save(Path(args.data_dir), Path(args.out))


if __name__ == "__main__":
    main()
