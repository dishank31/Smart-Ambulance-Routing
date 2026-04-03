from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional

class TriageRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    heart_rate: float
    bp_systolic: float
    bp_diastolic: float
    spo2: float
    respiratory_rate: float
    temperature: float
    gcs_score: int = Field(ge=3, le=15)
    pain_scale: int = Field(ge=0, le=10)
    age: int = Field(ge=0, le=120)
    has_chronic_condition: int = Field(ge=0, le=1)
    gender: str = Field(description="M or F")
    chief_complaint: str

class TriageResponse(BaseModel):
    severity_level: int
    severity_label: str
    recommended_department: str
    confidence: float
    emoji: str

class HospitalRecommendation(BaseModel):
    hospital_id: int
    name: str
    eta_min: float
    beds_available: int
    department: str
    score: float
    lat: float
    lon: float

class LocationContext(BaseModel):
    lat: float
    lon: float
    hour: int
    day_of_week: int
    month: int

class DispatchRequest(BaseModel):
    triage: TriageRequest
    location: LocationContext

class DispatchResponse(BaseModel):
    triage_result: TriageResponse
    recommendations: List[HospitalRecommendation]
