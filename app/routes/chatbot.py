# ai-customer-care/app/routes/chatbot.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.nlp_service import nlp_service
from app.services.speech_service import speech_service
import logging
from typing import List, Dict, Any

router = APIRouter()

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str # Main text input from text area
    user_id: str = "guest_user"
    is_voice_input: bool = False
    simulated_voice_text: str = "" # Text from simulated voice input
    conversation_history: List[Dict[str, Any]] = [] # Crucial for conversational context, flexible Any for content

class ChatResponse(BaseModel):
    response: str
    sentiment: Dict[str, Any]
    escalate_to_human: bool
    detected_intent: str
    generative_refinement_notes: str
    processed_message: str
    options: List[str] = [] # To send options back to frontend
    context: Dict[str, Any] = {} # To send context back to frontend
    action_type: str # To indicate bot's internal action

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Handles AI-powered L1 automation (chatbot).
    Performs sentiment detection, intent recognition, generates adaptive responses,
    and predicts auto-escalation. Supports conceptual voice input.
    """
    try:
        # Determine the message to process, prioritizing simulated_voice_text if voice input is selected
        if request.is_voice_input:
            if request.simulated_voice_text:
                 processed_message = request.simulated_voice_text
                 logging.info(f"Using simulated voice text: '{processed_message}'")
            else:
                 # Fallback to default simulation from speech_service if no simulated_voice_text is provided
                 processed_message = speech_service.transcribe_audio("dummy_audio.wav") 
                 logging.warning(f"Simulated voice input was empty, using default transcription: '{processed_message}'")
        else:
            processed_message = request.message # Use regular text message

        # Ensure processed_message is not empty before proceeding
        if not processed_message.strip():
            logging.warning("Received empty or whitespace-only message, returning generic response.")
            return ChatResponse(
                response="I didn't receive a clear message. Could you please try again?",
                sentiment={"label": "NEUTRAL", "score": 1.0},
                escalate_to_human=False,
                detected_intent="general_query",
                generative_refinement_notes="Empty message received.",
                processed_message="",
                options=[],
                context={},
                action_type="clarify_request"
            )


        logging.info(f"Processing chat request from user {request.user_id}: '{processed_message}' (Voice input: {request.is_voice_input})")

        # 1. Emotional Tone Detection (via sentiment analysis)
        sentiment_result = nlp_service.get_sentiment(processed_message)
        logging.info(f"Sentiment detected: {sentiment_result['label']} (Score: {sentiment_result['score']})")

        # 2. Intent Recognition and Specialization
        detected_intent, specialization = nlp_service.get_intent_and_specialization(processed_message) # UPDATED CALL
        logging.info(f"Intent detected: {detected_intent}, Specialization: {specialization}")

        # 3. Generative Response & Refinement
        # Pass specialization to the generative response for dynamic prompting
        bot_response_content, refinement_notes, options_from_llm, new_context_state, bot_action = nlp_service.get_generative_response(
            detected_intent,
            specialization, # Pass specialization
            sentiment_result['label'],
            processed_message, 
            request.user_id,
            request.conversation_history 
        )
        logging.info(f"Bot response generated: '{bot_response_content}'")
        logging.info(f"Options from LLM: {options_from_llm}")
        logging.info(f"New context state: {new_context_state}")
        logging.info(f"Bot's internal action: {bot_action}")

        # 4. Auto-escalation Prediction (now considers bot_action_type)
        escalate = nlp_service.predict_escalation(
            sentiment_result['score'],
            sentiment_result['label'],
            detected_intent,
            request.conversation_history,
            bot_action 
        )
        
        # If escalation is triggered, modify the response and clear options
        if escalate:
            bot_response_content += "\n\n**AI Escalation:** It seems your query requires human assistance. I'm escalating this to a human agent now and providing them with our conversation history."
            options_from_llm = [] # Clear options if escalating
            logging.warning("Escalation triggered based on detected conditions.")

        return ChatResponse(
            response=bot_response_content,
            sentiment=sentiment_result,
            escalate_to_human=escalate,
            detected_intent=detected_intent,
            generative_refinement_notes=refinement_notes,
            processed_message=processed_message,
            options=options_from_llm, # Return options to frontend
            context=new_context_state, # Return context to frontend
            action_type=bot_action # Return action type
        )
    except Exception as e:
        logging.error(f"Error in chat endpoint for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing chat: {e}")

