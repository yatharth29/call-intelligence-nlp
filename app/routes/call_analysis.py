# app/routes/call_analysis.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.nlp_service import nlp_service
import logging
from typing import List

router = APIRouter()

# --- Data Models ---
class CallAnalysisRequest(BaseModel):
    transcript_text: str
    call_id: str

class CallAnalysisResponse(BaseModel):
    summary: str
    tags: List[str]
    sentiment_overall: dict

# New Data Model for Q&A Request
class CallQuestionRequest(BaseModel):
    question: str
    transcript_text: str # Need the full transcript for context

# New Data Model for Q&A Response
class CallQuestionResponse(BaseModel):
    answer: str

@router.post("/call_nlp", response_model=CallAnalysisResponse)
async def analyze_call(request: CallAnalysisRequest):
    """
    Analyzes call transcripts for summary, tags, and overall sentiment.
    """
    try:
        logging.info(f"Received call analysis request for call {request.call_id}")

        # 1. Summarization
        summary = nlp_service.summarize_text(request.transcript_text)
        logging.info("Transcript summarized.")

        # 2. Tag and Entity Extraction
        tags = nlp_service.extract_tags_and_entities(request.transcript_text)
        logging.info(f"Tags extracted: {tags}")

        # 3. Overall Sentiment
        sentiment_overall = nlp_service.get_sentiment(request.transcript_text)
        logging.info(f"Overall sentiment: {sentiment_overall['label']}")

        return CallAnalysisResponse(
            summary=summary,
            tags=tags,
            sentiment_overall=sentiment_overall
        )
    except Exception as e:
        logging.error(f"Error in call analysis endpoint for call {request.call_id}: {e}", exc_info=True)
        # Raise HTTPException for client to see the error details
        raise HTTPException(status_code=500, detail=f"Internal server error processing call analysis: {e}")

@router.post("/ask_question", response_model=CallQuestionResponse) # NEW ENDPOINT
async def ask_call_question(request: CallQuestionRequest):
    """
    Answers a question based on the provided call transcript.
    """
    try:
        logging.info(f"Received question: '{request.question}' for call Q&A.")
        answer = nlp_service.answer_question_from_transcript(request.question, request.transcript_text)
        logging.info(f"Answer generated: '{answer[:50]}...'")
        return CallQuestionResponse(answer=answer)
    except Exception as e:
        logging.error(f"Error in ask_question endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing question: {e}")

