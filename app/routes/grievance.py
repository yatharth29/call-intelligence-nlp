# app/routes/grievance.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.nlp_service import nlp_service
import logging
from typing import List

router = APIRouter()

# --- Data Models ---
class GrievanceRequest(BaseModel):
    grievance_text: str
    customer_id: str

class GrievanceResponse(BaseModel):
    classification: str
    suggested_routing: List[str]
    priority: str

@router.post("/grievance", response_model=GrievanceResponse)
async def manage_grievance(request: GrievanceRequest):
    """
    Handles Smart Grievance Management.
    Classifies complaints, suggests routing to multiple departments if needed, and assigns priority based on content.
    """
    try:
        logging.info(f"Received grievance from customer {request.customer_id}: '{request.grievance_text}'")

        # Complaint Classification & Routing
        classification, routing, priority = nlp_service.classify_grievance(request.grievance_text)
        logging.info(f"Grievance classified as '{classification}', routed to '{routing}' with priority '{priority}'.")

        return GrievanceResponse(
            classification=classification,
            suggested_routing=routing,
            priority=priority
        )
    except Exception as e:
        logging.error(f"Error in grievance endpoint for customer {request.customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing grievance: {e}")