import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _clean_text(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_condition(s: str) -> str:
    s = _clean_text(s)
    s = s.lower()
    s = re.sub(r"\busers\b", "", s)
    s = re.sub(r"[^a-z0-9\s\-_/]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_drug_name(s: str) -> str:
    s = _clean_text(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_kaggle_drug_reviews(train_csv: Path, test_csv: Path) -> pd.DataFrame:
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    required = {"drugName", "condition", "review", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["drug"] = out["drugName"].map(_clean_drug_name)
    out["condition_clean"] = out["condition"].map(_clean_condition)
    out["review_clean"] = out["review"].map(_clean_text)

    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")

    out = out.dropna(subset=["drug", "condition_clean", "review_clean", "rating"])
    out = out[(out["drug"] != "") & (out["condition_clean"] != "") & (out["review_clean"] != "")]

    out["label"] = pd.NA
    out.loc[out["rating"] <= 4, "label"] = 1
    out.loc[out["rating"] >= 8, "label"] = 0
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)

    out = out[["drug", "condition_clean", "review_clean", "rating", "label"]]
    out = out.rename(columns={"condition_clean": "condition", "review_clean": "review"})

    out = out.drop_duplicates(subset=["drug", "condition", "review"])
    return out


def split_and_save(df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    y = df["label"]
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=y)
    y_temp = temp_df["label"]
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=y_temp)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    summary = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "rows": [len(train_df), len(val_df), len(test_df)],
            "risk_rate(label=1)": [
                float(train_df["label"].mean()),
                float(val_df["label"].mean()),
                float(test_df["label"].mean()),
            ],
        }
    )
    summary.to_csv(out_dir / "summary.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing Kaggle CSVs: drugsComTrain_raw.csv and drugsComTest_raw.csv",
    )
    p.add_argument("--out-dir", default="data/processed", help="Output folder")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    train_csv = input_dir / "drugsComTrain_raw.csv"
    test_csv = input_dir / "drugsComTest_raw.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Expected drugsComTrain_raw.csv and drugsComTest_raw.csv in --input-dir")

    df = load_kaggle_drug_reviews(train_csv, test_csv)
    df_labeled = make_labels(df)

    split_and_save(df_labeled, Path(args.out_dir), seed=args.seed)


if __name__ == "__main__":
    main()
