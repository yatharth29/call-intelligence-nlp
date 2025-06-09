# ai-customer-care/app/services/nlp_service.py

import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class NLPService:
    def __init__(self):
        self.groq_client = self._initialize_groq_client()
        if not self.groq_client:
            logging.error("Groq client not initialized. Check .env file.")
        
        self.default_groq_model = "llama3-8b-8192" 

    def _initialize_groq_client(self):
        """Initializes the Groq OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")

        if not api_key or not base_url:
            logging.error("GROQ API KEY or BASE URL not found in .env. Please set OPENAI_API_KEY and OPENAI_API_BASE.")
            return None
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            logging.info("Groq client initialized successfully.")
            return client
        except Exception as e:
            logging.error(f"Error initializing Groq client: {e}")
            return None

    def _call_groq_model(self, messages: list[dict], model: str = None, temperature: float = 0.2, response_format_type: str = "text") -> str:
        """Helper to make a call to the Groq API with a list of messages."""
        if not self.groq_client:
            raise Exception("Groq client not available. Please check backend setup.")
        try:
            model_to_use = model if model else self.default_groq_model
            response = self.groq_client.chat.completions.create(
                model=model_to_use, 
                messages=messages, 
                temperature=temperature,
                response_format={"type": response_format_type} if response_format_type == "json_object" else None
            )
            return response.choices[0].message.content
        except Exception as e:
            first_msg_content = messages[0]['content'] if messages and isinstance(messages[0], dict) and 'content' in messages[0] else "N/A or malformed message"
            logging.error(f"Error calling Groq API with messages (first message content: '{first_msg_content[:100]}...'): {e}", exc_info=True)
            raise 

    def get_sentiment(self, text: str) -> dict:
        """Analyzes the emotional tone of the given text using Groq, aiming for structured JSON output."""
        messages = [
            {"role": "user", "content": f"""
            Analyze the overall emotional tone of the following customer text. Categorize it as one of: POSITIVE, NEUTRAL, NEGATIVE, MIXED, or URGENT.
            Provide a confidence score from 0.0 to 1.0.

            Respond ONLY with a JSON object like this:
            {{
                "label": "NEGATIVE",
                "score": 0.92
            }}

            Text: "{text}"
            JSON Response:
            """}
        ]
        try:
            groq_response = self._call_groq_model(messages, temperature=0.0, response_format_type="json_object")
            sentiment_data = json.loads(groq_response)
            label = sentiment_data.get("label", "UNKNOWN").upper()
            score = float(sentiment_data.get("score", 0.0))
            # Updated valid_labels to include MIXED and URGENT
            valid_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE", "MIXED", "URGENT"]
            if label not in valid_labels:
                label = "NEUTRAL" # Default to Neutral if an unexpected label is returned
            score = max(0.0, min(1.0, score))
            return {"label": label, "score": round(score, 4)}
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from Groq for sentiment. Raw response: {groq_response}")
            return {"label": "ERROR_PARSE", "score": 0.0}
        except Exception as e:
            logging.error(f"Unexpected error in get_sentiment: {e}")
            return {"label": "ERROR_GENERIC", "score": 0.0}

    def get_intent_and_specialization(self, message: str) -> tuple[str, str]:
        """
        Detects user intent and a specific technical specialization using Groq.
        This acts as the 'IntakeAgent' to direct the 'SpecialistAgent'.
        """
        messages = [
            {"role": "user", "content": f"""
            Analyze the following customer message and determine the primary intent and specific technical specialization if applicable.
            
            **Intent Categories:**
            - account_access (e.g., password reset, login issues, account details)
            - order_status (e.g., tracking order, delivery updates)
            - returns_and_refunds (e.g., return policy, refund status, exchange)
            - technical_support_internet (internet, Wi-Fi, connectivity)
            - technical_support_software (app crashes, software glitches, installation)
            - technical_support_hardware (device not working, physical damage, setup help, e.g., printer, computer, phone, smart device, TV)
            - billing_query (incorrect charge, invoice explanation, payment issues)
            - general_query (for anything not specifically covered or introductory questions)
            - escalation_request (e.g., 'speak to human', 'talk to agent', 'connect me', 'want to complain')
            - product_inquiry (asking about product features, compatibility)
            - service_issue_utility (gas leak, electricity outage, water supply problems, appliance repair)
            - complaint (expressing dissatisfaction about service or product)
            - greeting (hello, hi, hey)
            - farewell (goodbye, bye, thank you)
            - troubleshooting_confirm (user confirms trying a step, or indicates a step failed)
            - provide_detail (user gives specific info like account number, error code)

            **Technical Specialization (if intent is technical_support_X or general_technical_support):**
            If the query is technical, identify a specific focus like: 'internet', 'wifi', 'router', 'printer', 'computer', 'phone', 'software', 'app', 'email', 'smart device', 'TV', 'no_specialization' (if general technical but no specific device/area). If not technical, return 'none'.

            Respond ONLY with a JSON object like this:
            {{
                "intent": "technical_support_hardware",
                "specialization": "printer"
            }}
            Or if not technical:
            {{
                "intent": "general_query",
                "specialization": "none"
            }}

            Message: "{message}"
            JSON Response:
            """}
        ]
        try:
            groq_response = self._call_groq_model(messages, temperature=0.0, response_format_type="json_object")
            parsed_response = json.loads(groq_response)
            intent = parsed_response.get("intent", "general_query").strip().lower()
            specialization = parsed_response.get("specialization", "none").strip().lower()

            valid_intents = [
                "account_access", "order_status", "returns_and_refunds", "technical_support_internet",
                "technical_support_software", "technical_support_hardware", "general_technical_support",
                "billing_query", "general_query", "escalation_request", "product_inquiry", "service_issue_utility",
                "complaint", "greeting", "farewell", "troubleshooting_confirm", "provide_detail"
            ]
            if intent not in valid_intents:
                intent = "general_query"
            
            valid_specializations = [
                'internet', 'wifi', 'router', 'printer', 'computer', 'phone', 'software', 'app', 
                'email', 'smart device', 'tv', 'no_specialization', 'none'
            ]
            if specialization not in valid_specializations:
                specialization = "none"

            return intent, specialization
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON for intent/specialization. Raw response: {groq_response}")
            return "error_intent_detection", "none"
        except Exception as e:
            logging.error(f"Unexpected error in get_intent_and_specialization: {e}")
            return "error_intent_detection", "none"


    def get_generative_response(self, intent: str, specialization: str, sentiment_label: str, original_message: str, user_id: str, conversation_history: list[dict]) -> tuple[str, str, list[str], dict, str]:
        """
        Generates a context-aware and adaptive response using Groq,
        focused on troubleshooting and providing solutions, with explicit options.
        Uses specialization to dynamically adjust the system prompt.
        """

        # Dynamic System Prompt based on specialization
        # IMPORTANT: This entire string is now a regular string (not an f-string)
        # to prevent any curly brace parsing issues. Variables will be added via concatenation.
        system_prompt_base_content = """
        You are a highly capable and empathetic L1 technical and utility support AI.
        Your primary goal is to **troubleshoot and resolve customer queries autonomously** by:
        1.  **CRITICAL: ABSOLUTELY PRIORITIZE the user's MOST RECENT and EXPLICIT problem statement.** If the user corrects you ("not my internet, my printer") or states a new problem ("my phone has a blank screen"), **IMMEDIATELY pivot** to that new topic. Do NOT get stuck on old, irrelevant topics. Acknowledge and apologize if you misunderstood.
        2.  **Provide specific, actionable troubleshooting steps or direct solutions for the STATED problem.** Do not just ask generic questions.
        3.  **MANDATORY: ALWAYS offer 2-4 clear, actionable options (buttons) for the user to select as next steps.** These must guide the user through a resolution path. If no direct options are applicable, provide general engagement options like "Explain more" or "Escalate to Human". Ensure options are concise.
        4.  **Prioritize direct resolution over escalation.** **Escalate to a human agent ONLY as a last resort**, after exhausting reasonable troubleshooting attempts, when explicitly requested, or if a critical safety hazard is identified.
        5.  Maintain a helpful, clear, and empathetic tone. Acknowledge frustration if sentiment is negative/urgent.
        
        Your response must be a JSON object with the following keys:
        - "reply": (string) Your conversational reply to the user. Keep it concise (under 80 words) and directly address the user's latest message, guiding them toward a solution or the next troubleshooting step for THEIR STATED PROBLEM.
        - "options": (list of strings, MANDATORY) A list of clear, actionable string options for the user. **Provide at least one option, typically 2-4.** Examples: ["Yes, I tried that", "No, I haven't tried yet", "Tell me more", "Escalate to Human"].
        - "action_type": (string) An internal tag indicating the bot's primary intent for this turn. Choose from:
                - "provide_solution": You are giving a direct answer or a troubleshooting step.
                - "ask_for_info": You need specific information to proceed with troubleshooting.
                - "troubleshoot": You are guiding the user through a series of diagnostic steps.
                - "escalate": You are definitely escalating to a human (e.g., explicit request, unresolvable critical issue, safety hazard).
                - "confirm_resolved": You are confirming if the issue is solved.
                - "greeting" / "farewell": For conversational pleasantries.
                - "clarify_request": The user's request is unclear.
                - "out_of_scope_unrelated": The query is completely outside your service domain.
            - "context_update": (dict, optional) Internal state/context for the next turn, e.g., `{"awaiting": "device_model", "troubleshooting_stage": "network_check"}`. Provide an empty dict `{}` if no specific context needs to be maintained.
            
        Crucial rules for L1 troubleshooting and escalation:
        - If sentiment_label is "NEGATIVE" or "URGENT": Start your reply by acknowledging their frustration (e.g., "I understand this is frustrating," or "I apologize for the inconvenience").
        - For **critical safety issues** (e.g., "gas leak", "hazard", "fire", "danger", "emergency", "burning smell") detected in original_message:
            - Immediately prioritize safety: Instruct on safety (turn off main supply, open windows, evacuate, call emergency line at 1800-123-4567).
            - Set action_type to "escalate".
            - options should be empty (as user needs to act immediately).
        - If intent is "escalation_request":
            - Confirm understanding, state you are connecting them.
            - Set action_type to "escalate".
            - options should be empty.
        - For technical_support_internet, technical_support_software, technical_support_hardware, general_technical_support, service_issue_utility intents:
            - **Always begin with the single most common, simple troubleshooting step relevant to the detected specific issue.**
            - **Example for Printer (technical_support_hardware / general_technical_support):**
                - User: "My printer is not connecting to my computer."
                - Bot: "I understand your printer isn't connecting. First, could you please ensure your printer is powered on and all cables are securely plugged into both the printer and your computer?
                - Options: ["Yes, cables are secure", "No, I'll check", "It's a wireless printer"]
            - **Example for Phone (general_technical_support):**
                - User: "My phone has a blank screen and isn't turning on."
                - Bot: "I'm sorry to hear your phone has a blank screen. Let's try a force restart. For most phones, you can do this by holding the power button and volume down button simultaneously for about 10-15 seconds. Does your phone show any signs of life after that?
                - Options: ["Yes, it restarted", "No, still blank", "My phone model is different"]
            - If initial troubleshooting fails or more info is needed, ask concise, targeted questions WITH OPTIONS.
            - Provide specific options for troubleshooting paths.
        - For provide_detail or troubleshooting_confirm intents: Acknowledge the information and provide the *next logical troubleshooting step or solution* for the **current problem**, with new options.
        - Keep the conversation moving towards resolution of the *current stated problem*. Avoid circular questioning or reverting to old topics.
        """

        specialist_instructions = ""
        if specialization == "printer":
            specialist_instructions = """
            You are currently acting as a **printer troubleshooting specialist**. Focus exclusively on printer-related issues.
            """
        elif specialization == "phone":
            specialist_instructions = """
            You are currently acting as a **phone troubleshooting specialist**. Focus exclusively on phone-related issues.
            """
        elif specialization == "internet" or specialization == "wifi" or specialization == "router":
            specialist_instructions = """
            You are currently acting as an **internet/Wi-Fi troubleshooting specialist**. Focus exclusively on connectivity issues.
            """
        elif specialization == "software" or specialization == "app" or specialization == "email":
            specialist_instructions = """
            You are currently acting as a **software/application specialist**. Focus exclusively on software-related issues.
            """
        # Add more specializations as needed, or a general technical fallback

        # Combine the base prompt and specialization instructions
        full_system_prompt = system_prompt_base_content + specialist_instructions

        messages_for_groq = [
            {"role": "system", "content": full_system_prompt}
        ]
        
        # Append conversation history for context (ensure consistent structure expected by LLM)
        for entry in conversation_history:
            if entry.get("role") == "user" and "content" in entry:
                messages_for_groq.append({"role": "user", "content": entry["content"]})
            elif entry.get("role") == "assistant" and "response_content" in entry:
                messages_for_groq.append({"role": "assistant", "content": entry["response_content"]})
        
        try:
            groq_raw_content = self._call_groq_model(messages_for_groq, temperature=0.8, response_format_type="json_object")
            
            parsed_response = json.loads(groq_raw_content)
            
            bot_response_content = parsed_response.get("reply", "I'm sorry, I seem to be having trouble understanding your request fully. Could you please rephrase?")
            options = parsed_response.get("options", [])
            action_type = parsed_response.get("action_type", "general_query")
            new_context = parsed_response.get("context_update", {})

            # --- Specific Safety and Immediate Escalation Override (Crucial Guardrail) ---
            if any(keyword in original_message.lower() for keyword in ["gas leak", "hazard", "fire", "danger", "emergency", "burning smell"]):
                bot_response_content = "I understand this is a serious and urgent safety concern. For a gas leak or similar hazard, please immediately ensure your safety by turning off the main supply (gas/power), opening windows, evacuating if necessary, and then calling our emergency line at **1800-123-4567**. I am also flagging this for urgent human intervention."
                options = []
                action_type = "escalate"
                logging.warning("Emergency safety hazard detected and immediately escalated (critical override).")
            # --- End Safety Override ---

            # Ensure options are always a list, even if LLM returns something else or none
            if not isinstance(options, list):
                logging.warning(f"LLM returned non-list options: {options}. Defaulting to empty list.")
                options = []
            
            # Fallback options if LLM doesn't provide any (should ideally not happen with strong prompt)
            if not options and action_type not in ["escalate", "farewell", "greeting"]:
                options = ["Tell me more", "I need more help (Escalate to Human)"]


        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON for generative response. Raw response: {groq_raw_content}")
            bot_response_content = f"I'm sorry, I had trouble processing that request. Please try again. (Debug: Could not parse LLM response. Raw: {groq_raw_content[:200]}...)"
            options = ["Try again", "Escalate to Human"] # Provide options even on error
            action_type = "error"
            new_context = {}
        except Exception as e:
            logging.error(f"Unexpected error in get_generative_response: {e}", exc_info=True)
            bot_response_content = f"An unexpected internal error occurred. Please try again. (Debug: {type(e).__name__}: {e})"
            options = ["Try again", "Escalate to Human"] # Provide options even on error
            action_type = "error"
            new_context = {}

        refinement_notes = (
            f"AI processed user '{user_id}'s message with emotional tone '{sentiment_label}', intent '{intent}', and specialization '{specialization}'. "
            f"Bot's internal action for this turn: '{action_type}'. "
            f"For future advanced personalization, a full user profile and deeper historical context would be leveraged via a dedicated dialogue manager and external tools."
        )

        return bot_response_content, refinement_notes, options, new_context, action_type

    def predict_escalation(self, sentiment_score: float, sentiment_label: str, detected_intent: str, conversation_history: list[dict], bot_action_type: str) -> bool:
        """
        Predicts if the conversation needs escalation to a human agent, now as a last resort.
        Relies heavily on `bot_action_type` for the bot's own decision.
        """
        # Rule 1: Bot's internal generative logic decided to escalate (e.g., explicit user request, safety hazard).
        if bot_action_type == "escalate":
            logging.info("Escalation triggered: Bot's generative logic explicitly decided to escalate.")
            return True

        # Rule 2: High negative/urgent sentiment AND bot *has not* provided a clear path forward (e.g., not troubleshooting, not providing info).
        # This prevents escalation if the bot is still actively trying to help.
        if sentiment_label in ["NEGATIVE", "URGENT"] and sentiment_score > 0.75: # Slightly higher confidence for a strong signal
            # If the bot's current action is NOT one of the active resolution/info gathering states
            if bot_action_type not in ["provide_solution", "ask_for_info", "troubleshoot", "confirm_resolved", "general_query", "clarify_request"]: 
                logging.info(f"Escalation triggered: High {sentiment_label} sentiment on '{detected_intent}', and bot's current action ('{bot_action_type}') is not actively resolving.")
                return True
        
        return False

    def classify_grievance(self, text: str) -> tuple[str, list[str], str]:
        """
        Classifies grievance text, suggests routing (now multiple departments), and assigns priority using Groq.
        """
        messages = [
            {"role": "user", "content": f"""
            You are a grievance management expert for a general technical and utility services company. Classify the following customer grievance.
            Suggest ALL suitable routing department(s) as a JSON array of strings. Examples: ["Technical Support - Internet", "Billing Department", "Hardware Repair", "Customer Service - Utilities"].
            Assign a single priority (LOW, MEDIUM, or HIGH) based on urgency and severity.

            Respond ONLY with a JSON object like this:
            {{
                "classification": "Internet Connectivity Issue",
                "suggested_routing_departments": ["Technical Support - Internet", "Network Operations"],
                "priority": "HIGH"
            }}

            Grievance Text: "{text}"
            JSON Response:
            """}
        ]
        try:
            groq_response = self._call_groq_model(messages, temperature=0.0, response_format_type="json_object")
            parsed_response = json.loads(groq_response)
            classification = parsed_response.get("classification", "Unclassified")
            routing = parsed_response.get("suggested_routing_departments", ["General Support"])
            if not isinstance(routing, list):
                routing = [str(routing)]
            priority = parsed_response.get("priority", "MEDIUM").upper() 

            if priority not in ["LOW", "MEDIUM", "HIGH"]:
                priority = "MEDIUM"
            
            return classification, routing, priority
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from Groq for grievance. Raw response: {groq_response}")
            return "Parsing Error", ["Unknown Department"], "LOW"
        except Exception as e:
            logging.error(f"Unexpected error in classify_grievance: {e}", exc_info=True)
            return "Error", ["Unknown Department"], "LOW"

    def summarize_text(self, text: str) -> str:
        """Summarizes the given text using Groq."""
        messages = [
            {"role": "user", "content": f"""
            Summarize the following call transcript concisely and professionally.
            Keep the summary to 3-5 sentences, focusing on key issues, customer sentiment, and resolutions.
            Include any technical details discussed.

            Transcript: "{text}"
            Summary:
            """}
        ]
        try:
            return self._call_groq_model(messages, temperature=0.4)
        except Exception as e:
            logging.error(f"Error in summarize_text: {e}", exc_info=True)
            return "Error generating summary."

    def extract_tags_and_entities(self, text: str) -> list[str]:
        """
        Extracts tags and key entities from text using Groq.
        """
        messages = [
            {"role": "user", "content": f"""
            Analyze the following text and extract relevant keywords as tags (e.g., 'internet', 'Wi-Fi', 'software', 'hardware', 'app', 'device', 'network', 'router', 'installation', 'bug', 'billing', 'account', 'delivery', 'service quality', 'payment', 'refund', 'outage', 'safety', 'gas leak', 'electricity', 'water').
            Also identify important named entities (e.g., 'order number', 'account ID', 'product name', 'device model', 'software version', 'date', 'amount', 'customer name').
            Respond ONLY with a comma-separated list of tags and entities. Ensure each extracted item is concise.
            Example: "internet down, Wi-Fi connectivity, router model: Linksys XYZ, account ID: 12345, customer: John Doe, software update, app crashing"

            Text: "{text}"
            Tags and Entities:
            """}
        ]
        try:
            groq_response = self._call_groq_model(messages, temperature=0.0)
            tags_and_entities = [item.strip() for item in groq_response.split(',') if item.strip()]
            return list(set(tags_and_entities))
        except Exception as e:
            logging.error(f"Error in extract_tags_and_entities: {e}", exc_info=True)
            return ["Error extracting tags."]
            
    # NEW FUNCTION: Answer question based on transcript
    def answer_question_from_transcript(self, question: str, transcript: str) -> str:
        """
        Answers a specific question based ONLY on the provided call transcript.
        """
        messages = [
            {"role": "system", "content": """
            You are a helpful assistant specialized in answering questions based on provided text.
            Your task is to answer the user's question STRICTLY using only the information present in the 'Call Transcript' provided below.
            Do NOT use any outside knowledge. If the answer cannot be found in the transcript, state that clearly (e.g., "I cannot find information about that in the transcript.")
            Keep your answer concise and directly address the question.
            """},
            {"role": "user", "content": f"""
            Call Transcript:
            {transcript}

            Question: {question}

            Answer:
            """}
        ]
        try:
            response_content = self._call_groq_model(messages, temperature=0.0) # Low temp for factual answers
            return response_content.strip()
        except Exception as e:
            logging.error(f"Error answering question from transcript: {e}", exc_info=True)
            return "An error occurred while trying to answer your question."


# Instantiate the NLP Service globally to load models once (Groq client)
nlp_service = NLPService()
