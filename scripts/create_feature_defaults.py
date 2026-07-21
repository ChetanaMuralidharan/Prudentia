from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import add_credit_risk_features  # noqa: E402


def json_safe(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def apply_caps(frame: pd.DataFrame, caps: dict) -> pd.DataFrame:
    result = frame.copy()
    for column, limits in caps.items():
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").clip(
                lower=limits.get("lower_p01"), upper=limits.get("upper_p99")
            )
    return result


def main() -> None:
    train_path = ROOT / "data" / "processed" / "loan_train.parquet"
    preprocessor_path = ROOT / "artifacts" / "preprocessor.joblib"
    caps_path = ROOT / "artifacts" / "outlier_caps.json"
    output_path = ROOT / "artifacts" / "feature_defaults.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Training population not found: {train_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    train = pd.read_parquet(train_path)
    train = train.drop(
        columns=[column for column in ["target", "loan_status", "issue_d", "issue_date", "earliest_cr_line"] if column in train.columns],
        errors="ignore",
    )
    train = add_credit_risk_features(train)
    caps = json.loads(caps_path.read_text(encoding="utf-8")) if caps_path.exists() else {}
    train = apply_caps(train, caps)

    preprocessor = joblib.load(preprocessor_path)
    defaults = {}
    for column in preprocessor.feature_names_in_:
        if column not in train.columns:
            defaults[column] = None
            continue
        series = train[column]
        if pd.api.types.is_numeric_dtype(series):
            value = pd.to_numeric(series, errors="coerce").median()
        else:
            mode = series.dropna().mode()
            value = mode.iloc[0] if not mode.empty else None
        defaults[column] = json_safe(value)

    payload = {
        "source": str(train_path),
        "strategy": {"numeric": "training median", "categorical": "training mode"},
        "feature_count": len(defaults),
        "defaults": defaults,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(defaults)} defaults to {output_path}")


if __name__ == "__main__":
    main()
