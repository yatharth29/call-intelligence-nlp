# ai-customer-care/frontend/streamlit_app.py

import streamlit as st
import requests
import json
import os
from datetime import datetime
import time # For simulating loading times and better UX

# --- Configuration ---
# IMPORTANT: Update this URL if you deploy your backend to a public server (e.g., Replit, Render, or another Hugging Face Space).
# For local development, 'http://localhost:8000' is fine if your backend is also running locally.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Unified AI Customer Care System",
    page_icon="�",
    layout="wide", # Use wide layout for better space utilization
    initial_sidebar_state="expanded"
)

# --- Global Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial assistant message with full structure for consistent display
    st.session_state.messages.append({
        "role": "assistant",
        "response_content": "Hello! I am your AI assistant. How can I help you today?",
        "sentiment_label": "NEUTRAL",
        "sentiment_score": 1.0,
        "detected_intent": "greeting",
        "escalate_to_human": False,
        "refinement_notes": "Initial greeting message from AI assistant.",
        "processed_message": "Hello!",
        "is_voice_input": False,
        "options": [], # Options are now expected
        "context": {}  # Context is now expected
    })

# Ensure simulated_voice_input_value is always empty by default at start of script run
if 'simulated_voice_input_value' not in st.session_state:
    st.session_state.simulated_voice_input_value = "" 

# Store conversation context from backend (e.g., awaiting specific info)
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {}

# --- NEW SESSION STATE FOR CALL QA ---
# Stores the transcript used for the current Q&A session
if 'current_qa_transcript' not in st.session_state:
    st.session_state.current_qa_transcript = "" 
# Stores Q&A pairs for the call analysis tab
if 'call_qa_history' not in st.session_state:
    st.session_state.call_qa_history = [] 

# --- Function to process user input (either from chat_input or option button) ---
def process_user_input(message: str, is_voice_input: bool):
    # Add user message to chat history immediately
    # We explicitly add 'content' for user messages for backend, 'display_content' for frontend display
    display_content = f"**User ({'Voice Input (Simulated)' if is_voice_input else 'Text Input'}):** {message}"
    st.session_state.messages.append({"role": "user", "content": message, "display_content": display_content}) 
    
    try:
        # Prepare history for backend, ensuring only 'role' and 'content' are sent
        history_for_backend = []
        for msg_entry in st.session_state.messages:
            if msg_entry["role"] == "user":
                # User messages sent to backend should have 'content' as their key
                history_for_backend.append({"role": "user", "content": msg_entry.get("content", "")})
            elif msg_entry["role"] == "assistant":
                # Assistant messages sent to backend should use their 'response_content' as 'content' for the LLM's context
                history_for_backend.append({"role": "assistant", "content": msg_entry.get("response_content", "")})


        payload = {
            "message": message, # This is the current message
            "user_id": "demo_user_123",
            "is_voice_input": is_voice_input,
            "simulated_voice_text": message, # Send the actual text for processing (used by backend if is_voice_input is True)
            "conversation_history": history_for_backend # Pass formatted history
        }
        response = requests.post(
            f"{BACKEND_URL}/l1_automation/chat",
            json=payload,
            timeout=120 # Increased timeout for LLM calls
        )
        response.raise_for_status()
        chat_data = response.json()

        # Extract all data for storing in session state
        bot_response_content = chat_data.get('response', "I couldn't generate a response.")
        sentiment_label = chat_data.get('sentiment', {}).get('label', 'N/A')
        sentiment_score = chat_data.get('sentiment', {}).get('score', 0.0)
        detected_intent = chat_data.get('detected_intent', 'N/A')
        escalate = chat_data.get('escalate_to_human', False)
        refinement_notes = chat_data.get('generative_refinement_notes', 'No specific refinement notes.')
        processed_message_from_backend = chat_data.get('processed_message', message)
        options_from_llm = chat_data.get('options', [])
        new_context_from_backend = chat_data.get('context', {})
        action_type_from_backend = chat_data.get('action_type', 'general_query')


        # Store bot's full response (including all AI analysis) to session state
        st.session_state.messages.append({
            "role": "assistant",
            "response_content": bot_response_content,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "detected_intent": detected_intent,
            "escalate_to_human": escalate,
            "refinement_notes": refinement_notes,
            "processed_message": processed_message_from_backend,
            "is_voice_input": is_voice_input,
            "options": options_from_llm, # Store options
            "context": new_context_from_backend, # Store new context
            "action_type": action_type_from_backend # Store action type
        })
        st.session_state.conversation_context = new_context_from_backend # Update global context

    except requests.exceptions.ConnectionError:
        error_msg = f"Cannot connect to backend at {BACKEND_URL}. Please ensure the FastAPI backend is running."
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "response_content": "Oops! I'm having trouble connecting to my brain right now. Please ensure the backend is running and try again later."})
    except requests.exceptions.Timeout:
        error_msg = "The request timed out. The backend might be slow to respond or models are still loading. Please try again."
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "response_content": "It's taking a bit longer than expected. Please try again or rephrase your question."})
    except requests.exceptions.RequestException as e:
        error_msg = f"Error processing your request: {e}. Please check the backend logs for details."
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "response_content": f"An error occurred while processing your request: {e}. Please try again."})
    except json.JSONDecodeError:
        error_msg = "Error decoding response from backend. Received malformed data. This might happen if Groq does not return valid JSON."
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "response_content": "I received an unreadable response from the server. Please try again."})
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "response_content": "An unexpected error occurred. Please try again later."})
            
    st.rerun() # Always rerun after processing to update display and clear input field


# --- Main Title and Tabs ---
st.title("🤖 Unified AI Customer Care System")
st.markdown("---")

tab_l1, tab_grievance, tab_call_nlp = st.tabs([
    "🗣️ AI-Powered L1 Automation",
    "📝 Smart Grievance Management",
    "📞 Call Intelligence via NLP"
])

# --- Sidebar for User Persona, Goals, Feedback Loop & Features ---
with st.sidebar:
    st.header("✨ Project Overview")
    st.markdown(
        """
        Our Unified AI Customer Care System revolutionizes support by intelligently automating L1 queries,
        streamlining grievance management, and enhancing call insights.
        """
    )
    st.markdown("---")

    st.header("💡 Key Features & Impact")
    
    # 1. User Persona & Pain Points
    with st.expander("👤 User Persona & Pain Points"):
        st.markdown(
            """
            **Meet Sarah, a Customer Support Manager:**
            Sarah oversees a large team of agents. Her daily challenges include:
            * **Long Hold Times:** Customers wait too long for basic queries, leading to frustration.
            * **Delayed Grievance Handling:** Complaints get lost or take ages to route, impacting resolution times.
            * **Unstructured Call Data:** Training, quality analysis, and performance reviews are inefficient without structured call summaries and tags.
            """
        )
    
    # 2. Measurable Goals & Impact
    with st.expander("🎯 Measurable Goals & Impact"):
        st.markdown(
            """
            * **~25-30% reduction** in support costs.
            * **~30% decrease** in grievance resolution time.
            * **90%+ accuracy** in call summarization and tagging.
            * **Scalability:** Adaptable across banking, e-commerce, telecom, healthcare.
            """
        )

    # 3. AI-Powered L1 Automation (Chat & Voice)
    with st.expander("⚙️ AI-Powered L1 Automation (Chat & Voice)"):
        st.markdown(
            """
            An intelligent bot capable of autonomously resolving basic customer queries, designed for multimodal input.
            
            **Unique Features:**
            * **Emotional Tone Detection:** Real-time sentiment analysis of customer messages (e.g., POSITIVE, NEGATIVE, NEUTRAL) enables adaptive responses.
            * **Adaptive Responses:** The bot adjusts its conversational tone and suggestions based on detected customer sentiment, fostering empathy and improving user experience.
            * **Auto-Escalation Prediction:** Automatically flags complex or highly negative interactions for human agent intervention, providing full conversation context to the human.
            * **Generative Response Refinement (Conceptual for Personalization):** The bot's ability to provide more personalized responses by leveraging user profiles and interaction history (e.g., via a LangChain-orchestrated LLM).
            * **Conceptual Voice Input:** Designed to integrate speech-to-text (e.g., using OpenAI's Whisper model) for voice-based interactions, broadening accessibility.
            """
        )
    
    # 4. Smart Grievance Management
    with st.expander("⚙️ Smart Grievance Management"):
        st.markdown(
            """
            A real-time complaint classification and intelligent routing system.
            
            **Impact:**
            * **Before:** Manual reading and forwarding leads to significant delays, misrouting, and increased customer frustration and potential churn.
            * **After (AI-Powered):** Instant, AI-driven classification and automated routing ensure grievances reach the correct department with appropriate priority immediately, drastically improving resolution time and customer satisfaction. This frees up human agents to focus on complex, high-priority cases.
            """
        )

    # 5. Call Intelligence via NLP
    with st.expander("⚙️ Call Intelligence via NLP"):
        st.markdown(
            """
            Leverages natural language processing to summarize and tag support call transcripts for efficient training and quality analysis. This module directly integrates and enhances your team member's existing work, utilizing the power of Groq/Mistral for analysis.
            
            **Impact:**
            * **Before:** Manual call review is time-consuming and inconsistent, hindering effective agent training and quality analysis.
            * **After (AI-Powered):** Automated summarization, tagging (extracting keywords and entities), and overall sentiment analysis provide actionable insights for efficient training, faster audits, and improved quality assurance. After summarizing and tagging, you can ask questions about the transcript for more detailed clarification.
            """
        )

    # 6. Feedback Loop & Continual Improvement
    with st.expander("🛠️ Feedback Loop & Continual Improvement"):
        st.markdown(
            """
            Our system is designed for continuous learning and refinement, ensuring its AI models remain effective and up-to-date:
            * **User Feedback:** Implicit (escalation rates) & Explicit (surveys).
            * **Agent Feedback (Human-in-the-Loop):** Corrections on classifications and responses for model retraining.
            * **Performance Monitoring:** Tracking key metrics to identify areas for optimization.
            * **New Data Ingestion:** Regular updates with new interactions to keep AI models current.
            """
        )

    st.markdown("---")
    st.info("💡 **Tip:** Ensure your FastAPI backend is running at `" + BACKEND_URL + "` for this app to function correctly.")


# --- Tab 1: AI-Powered L1 Automation (Chatbot) ---
with tab_l1:
    st.header("AI-Powered L1 Automation (Chat/Voice)")
    st.markdown("Our intelligent bot handles basic customer queries autonomously. It understands **emotional tone**, adapts responses, and predicts when human intervention is needed. Supports both text and conceptual voice input.")

    # Display chat messages from history
    # Use a container to keep chat history fixed while input scrolls
    chat_container = st.container()
    with chat_container:
        for msg_idx, message_entry in enumerate(st.session_state.messages):
            with st.chat_message(message_entry["role"]):
                # For display, use 'display_content' for user messages, otherwise 'response_content'
                if message_entry["role"] == "user":
                    st.markdown(message_entry.get("display_content", message_entry.get("content", "")))
                else: # Assistant message
                    st.markdown(message_entry.get("response_content", "Error: No response content found."))

                    # Display AI insights in an expander always at the bottom of the bot's turn
                    with st.expander("🤖 AI Insights (Click to expand)"):
                        if message_entry.get("is_voice_input"):
                            st.markdown(f"**Transcribed Input (Whisper Conceptual):** `{message_entry.get('processed_message', 'N/A')}`")
                        
                        st.markdown(f"**Detected Emotional Tone (Sentiment):** <span style='background-color:#ADD8E6; padding: 5px; border-radius: 5px; font-weight: bold;'>{message_entry.get('sentiment_label', 'N/A')}</span> (Confidence: {message_entry.get('sentiment_score', 0.0):.2f})", unsafe_allow_html=True)
                        st.markdown(f"**Detected Intent:** <span style='background-color:#90EE90; padding: 5px; border-radius: 5px;'>`{message_entry.get('detected_intent', 'N/A')}`</span>", unsafe_allow_html=True)
                        if message_entry.get("escalate_to_human"):
                            st.warning("🚨 **Auto-Escalation Predicted!** This query is flagged for human agent review. A human agent will take over shortly.")
                        st.info(f"**Generative Response Refinement Notes (Conceptual for Personalization):** *{message_entry.get('refinement_notes', 'No specific refinement notes.')}*")
                    
                    # Display options if the bot provided any
                    if message_entry.get("options"):
                        st.markdown("---")
                        st.markdown("**Choose an option or type your reply:**")
                        cols = st.columns(len(message_entry["options"]))
                        for i, option in enumerate(message_entry["options"]):
                            if cols[i].button(option, key=f"option_btn_{msg_idx}_{i}"):
                                process_user_input(option, is_voice_input=False)


    # --- Input Widgets (placed after chat display, within a form to prevent immediate rerun on each character type) ---
    with st.form("chat_input_form", clear_on_submit=True):
        input_method = st.radio("Choose input method:", ("Text Input", "Simulate Voice Input"), key="input_method_radio_form")
        
        col_input, col_send = st.columns([0.8, 0.2])

        if input_method == "Text Input":
            user_input_text_area = col_input.text_input(
                "Type your message here...",
                key="text_input_area",
                placeholder="Type your message...",
                label_visibility="collapsed"
            )
            sent_message = col_send.form_submit_button("Send Text")
            if sent_message and user_input_text_area:
                with st.spinner("Analyzing and generating response..."):
                    process_user_input(user_input_text_area, is_voice_input=False)
        else: # Simulate Voice Input
            col_input.info("💡 **Conceptual Voice Input:** Please type the *transcribed text* that Whisper *would* produce. This text will be processed as your voice input.")
            simulated_voice_text_display = col_input.text_area(
                "Type simulated transcribed voice input here:",
                key="chat_voice_input_area_final",
                height=80,
                value=st.session_state.simulated_voice_input_value, # Now uses the session state value
                label_visibility="collapsed"
            )
            sent_voice_message = col_send.form_submit_button("Send Voice (Simulated)")
            if sent_voice_message: # Process even if empty, but warn
                if simulated_voice_text_display.strip(): # Check if text is not just whitespace
                    st.session_state.simulated_voice_input_value = "" # Clear after sending
                    with st.spinner("Analyzing and generating response..."):
                        process_user_input(simulated_voice_text_display, is_voice_input=True)
                else:
                    st.warning("Please enter some transcribed text for simulated voice input.")
        


# --- Tab 2: Smart Grievance Management ---
with tab_grievance:
    st.header("Smart Grievance Management")
    st.markdown("Our system provides real-time complaint classification and intelligent routing, significantly speeding up redressal and improving efficiency. It can now suggest **multiple relevant departments** and provide more nuanced priority.")

    grievance_input = st.text_area("Enter customer grievance details:", height=180, key="grievance_text_area",
                                   value="My recent internet bill has an incorrect charge of $50 for an unlimited data plan I never subscribed to. This is unacceptable and needs to be resolved urgently! Account: 987654. I also had a problem with the delivery of a recent product.")

    if st.button("Classify & Route Grievance"):
        if grievance_input:
            with st.spinner("Classifying and routing grievance..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/grievance_management/grievance",
                        json={"grievance_text": grievance_input, "customer_id": "cust_001"},
                        timeout=30
                    )
                    response.raise_for_status()
                    grievance_data = response.json()

                    st.success("Grievance Analysis Complete!")
                    st.markdown(f"**📝 Classification:** `{grievance_data['classification']}`")
                    
                    # Display multiple routing departments
                    st.markdown(f"**➡️ Suggested Routing:** `{', '.join(grievance_data.get('suggested_routing', ['N/A']))}`")
                    st.markdown(f"**⚡ Priority:** `{grievance_data['priority']}`")

                    st.markdown("---")
                    

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to backend at {BACKEND_URL}. Please ensure the FastAPI backend is running.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. The backend might be slow to respond.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error processing grievance: {e}. Please check the backend logs for details.")
                except json.JSONDecodeError:
                    st.error("Error decoding response from backend. This might happen if Groq does not return valid JSON for classification.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter some grievance text to classify.")

# --- Tab 3: Call Intelligence via NLP ---
with tab_call_nlp:
    st.header("Call Intelligence via NLP")
    st.markdown("Leverage natural language processing to summarize and tag support call transcripts for efficient training and quality analysis. This module directly integrates and enhances your team member's existing work.")

    sample_transcript = """
    Agent: Hello, thank you for calling Tech Support. How may I help you today?
    Customer: Hi. I'm really frustrated. My internet has been completely out for the last 2 hours. I can't work from home! My order number is 12345.
    Agent: I understand your frustration. Let me check that for you. Can you confirm your account number please?
    Customer: It's 987654. This happens almost every month. I pay for premium service!
    Agent: I see here there's a regional outage in your area. Our technicians are already working on it. The estimated fix time is within the next 4 hours.
    Customer: Four hours? That's unacceptable! I need to finish this report now.
    Agent: I apologize for the inconvenience. We're doing our best to restore service as quickly as possible. Would you like me to create a ticket for you to receive SMS updates on the restoration progress?
    Customer: Yes, please do that. And I expect some compensation for this constant issue.
    Agent: I've created ticket #7890. You'll receive updates. Regarding compensation, once service is restored, you can visit our website or chat with us to discuss credit options.
    Customer: Fine. Thank you.
    """
    transcript_input = st.text_area("Paste Call Transcript here:", value=sample_transcript, height=300, key="transcript_text_area")

    if st.button("Summarize & Tag Call"):
        if transcript_input:
            # Store the current transcript for Q&A before processing
            st.session_state.current_qa_transcript = transcript_input
            # Clear previous Q&A history if a new transcript is being analyzed
            st.session_state.call_qa_history = [] 

            with st.spinner("Analyzing transcript (summarizing, tagging, sentiment)..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/call-intelligence/call_nlp", # Changed URL here
                        json={"transcript_text": transcript_input, "call_id": f"call_{datetime.now().strftime('%Y%m%d%H%M%S')}"},
                        timeout=120 # Extended timeout for summarization, which can be resource-intensive
                    )
                    response.raise_for_status()
                    call_nlp_data = response.json()

                    st.success("Call Analysis Complete!")
                    st.subheader("📝 Summary:")
                    st.write(call_nlp_data.get('summary', 'No summary generated.'))

                    st.subheader("🏷️ Tags & Key Entities:")
                    st.write(", ".join(call_nlp_data.get('tags', ['No tags extracted.'])))

                    st.subheader("😊 Overall Call Sentiment:")
                    sentiment_label = call_nlp_data.get('sentiment_overall', {}).get('label', 'N/A')
                    sentiment_score = call_nlp_data.get('sentiment_overall', {}).get('score', 0.0)
                    st.info(f"**Sentiment:** **{sentiment_label}** (Confidence: {sentiment_score:.2f})")

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to backend at {BACKEND_URL}. Please ensure the FastAPI backend is running.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. The backend might be slow to respond or models are still loading.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error processing call transcript: {e}. Please check the backend logs for details.")
                except json.JSONDecodeError:
                    st.error("Error decoding response from backend. This might happen if Groq does not return valid JSON for sentiment/tags.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please paste a call transcript to analyze.")

    # --- NEW Q&A SECTION FOR CALL INTELLIGENCE TAB ---
    st.markdown("---")
    st.subheader("Ask Questions about the Transcript")
    st.markdown("Ask questions about the provided call transcript for more detailed clarification.")

    # Display Q&A history
    qa_container = st.container(height=300, border=True)
    with qa_container:
        if not st.session_state.call_qa_history:
            st.markdown("Once you summarize a transcript, you can ask questions here!")
        else:
            for qa_pair in st.session_state.call_qa_history:
                st.markdown(f"**Q:** {qa_pair['question']}")
                st.markdown(f"**A:** {qa_pair['answer']}")
                st.markdown("---") # Separator for Q&A pairs

    # Input for Q&A
    qa_question = st.text_input("Your question:", key="qa_question_input", 
                                disabled=(not st.session_state.current_qa_transcript),
                                placeholder="E.g., What was the customer's account number?")
    
    if st.button("Ask Question", disabled=(not st.session_state.current_qa_transcript)):
        if qa_question and st.session_state.current_qa_transcript:
            with st.spinner("Getting answer from transcript..."):
                try:
                    qa_payload = {
                        "question": qa_question,
                        "transcript_text": st.session_state.current_qa_transcript
                    }
                    qa_response = requests.post(
                        f"{BACKEND_URL}/call-intelligence/ask_question", # New Q&A endpoint
                        json=qa_payload,
                        timeout=60 # Shorter timeout for quick Q&A
                    )
                    qa_response.raise_for_status()
                    qa_data = qa_response.json()
                    answer = qa_data.get("answer", "Could not retrieve an answer from the transcript.")
                    
                    st.session_state.call_qa_history.append({"question": qa_question, "answer": answer})
                    # Clear the input field after asking
                    st.session_state.qa_question_input = "" 

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to backend at {BACKEND_URL}. Please ensure the FastAPI backend is running.")
                except requests.exceptions.Timeout:
                    st.error("The Q&A request timed out. Please try again.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error asking question: {e}. Check backend logs.")
                except json.JSONDecodeError:
                    st.error("Error decoding Q&A response from backend.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during Q&A: {e}")
            st.rerun() # Rerun to update Q&A history display
        else:
            st.warning("Please summarize a transcript first and then type your question.")


st.markdown("---")
st.caption("Developed for CyFuture AI Hackathon 2025")
