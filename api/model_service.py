from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import add_credit_risk_features  # noqa: E402


LGD = 0.60
RISK_TIERS = {
    "A": "0.00-0.05",
    "B": "0.05-0.10",
    "C": "0.10-0.15",
    "D": "0.15-0.20",
    "E": "0.20-0.30",
    "F": "0.30-1.00",
}
RISK_LABELS = {
    "A": "Very low risk",
    "B": "Low risk",
    "C": "Moderate risk",
    "D": "Elevated risk",
    "E": "High risk",
    "F": "Very high risk",
}


def sanitize_xgboost_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = (
        result.columns.astype(str)
        .str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "lt_", regex=False)
        .str.replace(">", "gt_", regex=False)
    )
    return result


def json_safe(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class ModelService:
    def __init__(self) -> None:
        self.artifacts_dir = ROOT / "artifacts"
        self.models_dir = self.artifacts_dir / "models"
        self.processed_dir = ROOT / "data" / "processed"

        self.preprocessor = joblib.load(self.artifacts_dir / "preprocessor.joblib")
        self.calibrated_model = joblib.load(self.models_dir / "calibrated_model.joblib")
        self.feature_names = self._load_json(self.artifacts_dir / "feature_names.json", [])
        self.outlier_caps = self._load_json(self.artifacts_dir / "outlier_caps.json", {})

        model_meta = self._load_json(self.models_dir / "best_model_metadata.json", {})
        calibration_meta = self._load_json(self.models_dir / "calibration_metadata.json", {})
        self.model_name = model_meta.get("best_model_name", calibration_meta.get("base_model_name", "unknown"))
        self.calibration_method = calibration_meta.get("selected_calibration_method", "unknown")
        self.threshold = self._load_threshold()
        self.defaults = self._load_defaults()

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @property
    def required_columns(self) -> list[str]:
        return list(self.preprocessor.feature_names_in_)

    def _load_defaults(self) -> dict[str, Any]:
        path = self.artifacts_dir / "feature_defaults.json"
        payload = self._load_json(path, {})
        defaults = payload.get("defaults", payload)
        if defaults:
            return defaults

        # Fallback: recover fitted imputer statistics from the saved preprocessor.
        defaults: dict[str, Any] = {column: None for column in self.required_columns}
        for transformer_name, transformer, columns in self.preprocessor.transformers_:
            if transformer_name == "remainder" or transformer == "drop":
                continue
            if not hasattr(transformer, "named_steps") or "imputer" not in transformer.named_steps:
                continue
            statistics = transformer.named_steps["imputer"].statistics_
            for column, statistic in zip(list(columns), statistics):
                defaults[str(column)] = json_safe(statistic)
        return defaults

    def _load_threshold(self) -> float:
        candidates = [
            self.processed_dir / "phase6_decisioning" / "cost_threshold_optimization.csv",
            self.processed_dir / "phase6_decisioning" / "threshold_optimization.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            table = pd.read_csv(path)
            if not {"threshold", "total_cost"}.issubset(table.columns):
                continue
            table["threshold"] = pd.to_numeric(table["threshold"], errors="coerce")
            table["total_cost"] = pd.to_numeric(table["total_cost"], errors="coerce")
            table = table.dropna(subset=["threshold", "total_cost"])
            if not table.empty:
                return float(table.loc[table["total_cost"].idxmin(), "threshold"])
        return 0.08

    @staticmethod
    def _normalize_term(value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"36", "36 month", "36 months"}:
            return " 36 months"
        if text in {"60", "60 month", "60 months"}:
            return " 60 months"
        return value

    def _apply_outlier_caps(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, limits in self.outlier_caps.items():
            if column not in result.columns:
                continue
            result[column] = pd.to_numeric(result[column], errors="coerce")
            result[column] = result[column].clip(
                lower=limits.get("lower_p01"),
                upper=limits.get("upper_p99"),
            )
        return result

    def prepare_application(self, application: dict[str, Any]) -> pd.DataFrame:
        record = dict(self.defaults)
        record.update({key: value for key, value in application.items() if value is not None})
        record["term"] = self._normalize_term(record.get("term"))
        if record.get("funded_amnt") in (None, 0):
            record["funded_amnt"] = record.get("loan_amnt")

        frame = pd.DataFrame([record])
        frame = add_credit_risk_features(frame)
        frame = self._apply_outlier_caps(frame)
        for column in self.required_columns:
            if column not in frame.columns:
                frame[column] = self.defaults.get(column)
        return frame[self.required_columns]

    def transform(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        encoded = self.preprocessor.transform(raw_frame)
        feature_names = self.feature_names or self.preprocessor.get_feature_names_out().tolist()
        encoded_frame = pd.DataFrame(encoded, columns=feature_names, index=raw_frame.index)
        if self.model_name == "xgboost_sampled":
            encoded_frame = sanitize_xgboost_columns(encoded_frame)
        return encoded_frame

    @staticmethod
    def risk_tier(probability: float) -> str:
        if probability < 0.05:
            return "A"
        if probability < 0.10:
            return "B"
        if probability < 0.15:
            return "C"
        if probability < 0.20:
            return "D"
        if probability < 0.30:
            return "E"
        return "F"

    def reason_codes(self, record: dict[str, Any], probability: float) -> list[str]:
        reasons: list[tuple[float, str]] = []

        def number(name: str) -> float | None:
            value = record.get(name)
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        dti = number("dti")
        revol_util = number("revol_util")
        annual_inc = number("annual_inc")
        loan_amnt = number("loan_amnt")
        delinq = number("delinq_2yrs")
        inquiries = number("inq_last_6mths")
        public_records = number("pub_rec")
        recent_accounts = number("acc_open_past_24mths")
        available = number("bc_open_to_buy")
        total_limit = number("total_bc_limit")
        percent_high = number("percent_bc_gt_75")

        if dti is not None and dti >= 35:
            reasons.append((4.0, "Very high debt-to-income ratio"))
        elif dti is not None and dti >= 25:
            reasons.append((3.0, "High debt-to-income ratio"))

        if revol_util is not None and revol_util >= 90:
            reasons.append((4.0, "Very high revolving credit utilization"))
        elif revol_util is not None and revol_util >= 70:
            reasons.append((3.0, "High revolving credit utilization"))

        if annual_inc and annual_inc > 0 and loan_amnt is not None:
            ratio = loan_amnt / annual_inc
            if ratio >= 0.50:
                reasons.append((4.0, "Requested loan is high relative to annual income"))
            elif ratio >= 0.30:
                reasons.append((3.0, "Loan amount is elevated relative to income"))

        if str(record.get("term", "")).strip() == "60 months":
            reasons.append((2.5, "Longer requested repayment term"))
        if delinq is not None and delinq > 0:
            reasons.append((3.5, "Recent delinquency history"))
        if public_records is not None and public_records > 0:
            reasons.append((3.5, "Public-record history contributes to risk"))
        if inquiries is not None and inquiries >= 3:
            reasons.append((2.5, "Multiple recent credit inquiries"))
        if recent_accounts is not None and recent_accounts >= 6:
            reasons.append((2.5, "Numerous recently opened credit accounts"))
        if available is not None and total_limit is not None and total_limit > 0:
            if available / total_limit < 0.10:
                reasons.append((3.0, "Limited available bankcard credit"))
        if percent_high is not None and percent_high >= 75:
            reasons.append((3.0, "Large share of bankcards is highly utilized"))
        if str(record.get("verification_status", "")).lower() == "not verified":
            reasons.append((1.5, "Income was not independently verified"))

        unique: list[str] = []
        for _, reason in sorted(reasons, key=lambda item: item[0], reverse=True):
            if reason not in unique:
                unique.append(reason)
        if not unique:
            unique = [
                "Combined credit profile exceeds the approval threshold"
                if probability >= self.threshold
                else "Applicant profile remains within the approval threshold"
            ]
        return unique[:4]

    def predict_one(self, application: dict[str, Any]) -> dict[str, Any]:
        raw = self.prepare_application(application)
        encoded = self.transform(raw)
        probability = float(self.calibrated_model.predict_proba(encoded)[0, 1])
        approved = probability < self.threshold
        tier = self.risk_tier(probability)
        loan_amount = float(application.get("loan_amnt", raw.iloc[0].get("loan_amnt", 0)) or 0)
        reasons = self.reason_codes(raw.iloc[0].to_dict(), probability)
        return {
            "probability_of_default": probability,
            "probability_percent": probability * 100,
            "decision": "APPROVED" if approved else "DENIED",
            "approved": approved,
            "decision_threshold": self.threshold,
            "risk_tier": tier,
            "risk_label": RISK_LABELS[tier],
            "expected_loss": probability * LGD * loan_amount,
            "adverse_action_reasons": reasons,
            "model_name": self.model_name,
            "calibration_method": self.calibration_method,
        }

    def predict_many(self, applications: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict_one(application) for application in applications]

    def model_info(self) -> dict[str, Any]:
        encoded_count = len(self.feature_names) or len(self.preprocessor.get_feature_names_out())
        return {
            "model_name": self.model_name,
            "calibration_method": self.calibration_method,
            "decision_threshold": self.threshold,
            "raw_input_feature_count": len(self.required_columns),
            "encoded_feature_count": encoded_count,
            "risk_tiers": RISK_TIERS,
            "extra": {
                "loss_given_default": LGD,
                "decision_rule": "Approve when calibrated PD is below the threshold",
            },
        }
