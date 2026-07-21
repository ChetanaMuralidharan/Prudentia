from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    """A compact application schema. Extra raw Lending Club fields are accepted."""

    model_config = ConfigDict(extra="allow")

    loan_amnt: float = Field(..., gt=0)
    annual_inc: float = Field(..., gt=0)
    dti: float = Field(..., ge=0)

    funded_amnt: float | None = Field(default=None, gt=0)
    term: str = "36 months"
    revol_util: float | None = Field(default=None, ge=0)
    revol_bal: float | None = Field(default=None, ge=0)
    fico_range_low: float | None = Field(default=None, ge=300, le=850)
    fico_range_high: float | None = Field(default=None, ge=300, le=850)
    emp_length: str | None = None
    home_ownership: str | None = None
    verification_status: str | None = None
    purpose: str | None = None
    addr_state: str | None = None
    zip_code: str | None = None
    initial_list_status: str | None = None
    disbursement_method: str | None = None
    delinq_2yrs: float | None = Field(default=None, ge=0)
    inq_last_6mths: float | None = Field(default=None, ge=0)
    open_acc: float | None = Field(default=None, ge=0)
    pub_rec: float | None = Field(default=None, ge=0)
    total_acc: float | None = Field(default=None, ge=0)
    mort_acc: float | None = Field(default=None, ge=0)
    bc_open_to_buy: float | None = None
    total_bc_limit: float | None = None
    acc_open_past_24mths: float | None = None
    num_tl_op_past_12m: float | None = None
    percent_bc_gt_75: float | None = None


class PredictionResponse(BaseModel):
    probability_of_default: float
    probability_percent: float
    decision: str
    approved: bool
    decision_threshold: float
    risk_tier: str
    risk_label: str
    expected_loss: float
    adverse_action_reasons: list[str]
    model_name: str
    calibration_method: str


class BatchPredictionRequest(BaseModel):
    applications: list[LoanApplication]


class BatchPredictionResponse(BaseModel):
    prediction_count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    defaults_loaded: bool
    decision_threshold: float


class ModelInfoResponse(BaseModel):
    model_name: str
    calibration_method: str
    decision_threshold: float
    raw_input_feature_count: int
    encoded_feature_count: int
    risk_tiers: dict[str, str]
    extra: dict[str, Any]
