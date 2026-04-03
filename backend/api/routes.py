from fastapi import APIRouter, HTTPException, Depends
from typing import List
from .schemas import DispatchRequest, DispatchResponse, TriageResponse, HospitalRecommendation
from ..services.prediction_service import PredictionService
from ..services.hospital_service import HospitalService
from src.models.decision_engine import HospitalRecommender
import pandas as pd

router = APIRouter()

pred_service = None
hosp_service = None

def get_pred_service():
    global pred_service
    if pred_service is None:
        pred_service = PredictionService()
    return pred_service

def get_hosp_service():
    global hosp_service
    if hosp_service is None:
        hosp_service = HospitalService()
    return hosp_service

@router.post("/dispatch", response_model=DispatchResponse)
async def dispatch_ambulance(
    request: DispatchRequest, 
    preds: PredictionService = Depends(get_pred_service),
    hosps: HospitalService = Depends(get_hosp_service)
):
    if not preds.is_ready():
        raise HTTPException(status_code=503, detail="Machine Learning models are still training. Please wait.")
        
    try:
        req_dict = request.triage.model_dump()
        triage_result = preds.predict_severity(req_dict)
        
        loc_ctx = request.location.model_dump()
        
        nearby_hospitals = hosps.get_nearby_hospitals(loc_ctx['lat'], loc_ctx['lon'])
        
        if not nearby_hospitals:
            raise HTTPException(status_code=404, detail="No hospitals found within the search radius.")
            
        coords = [(h['latitude'], h['longitude']) for h in nearby_hospitals]
        eta_preds = preds.batch_predict_eta(coords, loc_ctx)
        
        dept = triage_result['recommended_department']
        bed_preds = preds.batch_predict_beds(nearby_hospitals, loc_ctx, req_dict, department=dept)
        
        eta_dict = {h['id']: eta for h, eta in zip(nearby_hospitals, eta_preds)}
        bed_dict = {h['id']: beds for h, beds in zip(nearby_hospitals, bed_preds)}
        
        df_hosps = pd.DataFrame(nearby_hospitals)
        recommender = HospitalRecommender(df_hosps)
        
        raw_recs = recommender.recommend(triage_result, eta_dict, bed_dict)
        
        recommendations = [HospitalRecommendation(**r) for r in raw_recs]
        
        return DispatchResponse(
            triage_result=TriageResponse(**triage_result),
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/status")
async def get_status(preds: PredictionService = Depends(get_pred_service)):
    return {
        "status": "online" if preds.is_ready() else "training",
        "models_ready": preds.is_ready()
    }
