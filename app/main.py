# app/main.py

from fastapi import FastAPI
from app.routes import chatbot, grievance, call_analysis # Import all routers
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Unified AI Customer Care System Backend",
    description="Integrated APIs for L1 Automation, Grievance Management, and Call Intelligence via NLP.",
    version="1.0.0"
)

# Include routers with prefixes and tags
app.include_router(chatbot.router, prefix="/l1_automation", tags=["L1 Automation (Chat/Voice)"])
app.include_router(grievance.router, prefix="/grievance_management", tags=["Smart Grievance Management"])
app.include_router(call_analysis.router, prefix="/call_intelligence", tags=["Call Intelligence via NLP"]) 

@app.get("/")
async def root():
    """Root endpoint for basic health check."""
    return {"message": "Welcome to the Unified AI Customer Care System Backend! Visit /docs for API documentation."}
