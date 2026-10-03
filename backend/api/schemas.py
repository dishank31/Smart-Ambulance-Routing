from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientVitals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heart_rate: float = Field(gt=0, le=250)
    bp_systolic: float = Field(gt=0, le=300)
    bp_diastolic: float = Field(gt=0, le=200)
    spo2: float = Field(ge=50, le=100)
    respiratory_rate: float = Field(gt=0, le=60)
    temperature: float = Field(ge=30, le=45)
    gcs_score: int = Field(ge=3, le=15)
    pain_scale: int = Field(ge=0, le=10)
    age: int = Field(ge=0, le=120)
    gender: str
    has_chronic_condition: bool
    chief_complaint: str = Field(min_length=2, max_length=100)

    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"male", "m"}:
            return "Male"
        if normalized in {"female", "f"}:
            return "Female"
        raise ValueError("gender must be Male/Female or M/F")


class EmergencyLocation(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class EmergencyRequest(BaseModel):
    location: EmergencyLocation
    patient_vitals: PatientVitals
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    use_ml_eta: bool = True


class SeverityResponse(BaseModel):
    severity: int
    severity_label: str
    confidence: float
    required_department: str


class HospitalInfo(BaseModel):
    hospital_id: int
    name: str
    location: EmergencyLocation
    department: str
    predicted_eta_minutes: float
    predicted_beds_available: int
    score: float
    route_geometry: Optional[dict] = None
    route_steps: List[dict] = Field(default_factory=list)


class NaiveHospitalInfo(HospitalInfo):
    why_not_optimal: str


class RecommendationResponse(BaseModel):
    severity_info: SeverityResponse
    optimal_hospital: HospitalInfo
    alternative_hospitals: List[HospitalInfo]
    naive_choice: NaiveHospitalInfo
    recommendation_reason: str
    recommendation_breakdown: List[str] = Field(default_factory=list)
    processing_time_ms: float
    emergency_location: Optional[EmergencyLocation] = None


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None


class ModelComparisonResponse(BaseModel):
    severity_models: List[ModelMetrics]
    eta_models: List[ModelMetrics]
    bed_models: List[ModelMetrics]
