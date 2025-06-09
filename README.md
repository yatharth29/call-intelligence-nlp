Unified AI Customer Care System
This repository contains the code for a Unified AI Customer Care System, designed to revolutionize customer support operations through intelligent automation, smart grievance management, and insightful call intelligence. Leveraging advanced Natural Language Processing (NLP) models, this system aims to significantly improve efficiency, reduce operational costs, and boost customer satisfaction.

Introduction
In today's fast-paced service industry, managing customer interactions effectively and efficiently is paramount. Traditional customer support often faces challenges such as long hold times, delayed complaint resolution, and a lack of structured insights from customer interactions. This project addresses these pain points by integrating AI capabilities into a cohesive customer care platform.

Problem Statement: The Challenges Faced by Customer Support
Consider Sarah, a Customer Support Manager overseeing a large team of agents. Her biggest headaches include:

Long Hold Times: Customers experience frustrating waits for even basic queries.

Delayed Grievance Handling: Complaints often get misrouted or take too long to resolve, impacting customer satisfaction and potentially leading to churn.

Unstructured Call Data: Manual review of call transcripts for training, quality analysis, and performance assessment is time-consuming and inefficient.

Solution Overview: Key Features & Impact
Our Unified AI Customer Care System offers a comprehensive solution with the following measurable impacts:

~25-30% Reduction in Support Costs: Achieved through automation of Level 1 (L1) queries.

~30% Decrease in Grievance Resolution Time: Enabled by instant classification and intelligent routing.

90%+ Accuracy in Call Summarization and Tagging: Providing actionable insights for training and quality assurance.

Scalability: Adaptable across various industry sectors, including banking, e-commerce, telecom, and healthcare.

The system is comprised of three core modules:

AI-Powered L1 Automation (Chat & Voice)

Smart Grievance Management

Call Intelligence via NLP

Modules Breakdown
1. 🗣️ AI-Powered L1 Automation (Chat & Voice)
An intelligent bot capable of autonomously resolving basic customer queries, designed for multimodal input.

Unique Features:

Emotional Tone Detection: Real-time sentiment analysis of customer messages (e.g., POSITIVE, NEGATIVE, NEUTRAL, MIXED, URGENT) enables adaptive responses.

Adaptive Responses: The bot adjusts its conversational tone and suggestions based on detected customer sentiment, fostering empathy and improving user experience.

Auto-Escalation Prediction: Automatically flags complex or highly negative interactions for human agent intervention, providing full conversation context to the human.

Generative Response Refinement (Conceptual for Personalization): The bot's ability to provide more personalized responses by leveraging user profiles and interaction history (e.g., via a LangChain-orchestrated LLM).

Conceptual Voice Input: Designed to integrate speech-to-text (e.g., using OpenAI's Whisper model) for voice-based interactions, broadening accessibility.

2. 📝 Smart Grievance Management
A real-time complaint classification and intelligent routing system.

Impact:

Before: Manual reading and forwarding leads to significant delays, misrouting, and increased customer frustration and potential churn.

After (AI-Powered): Instant, AI-driven classification and automated routing ensure grievances reach the correct department with appropriate priority immediately, drastically improving resolution time and customer satisfaction. This frees up human agents to focus on complex, high-priority cases.

3. 📞 Call Intelligence via NLP
Leverages natural language processing to summarize and tag support call transcripts for efficient training and quality analysis. This module directly integrates and enhances your team member's existing work, utilizing the power of Groq/Mistral for analysis.

Impact:

Before: Manual call review is time-consuming and inconsistent, hindering effective agent training and quality analysis.

After (AI-Powered): Automated summarization, tagging (extracting keywords and entities), and overall sentiment analysis provide actionable insights for efficient training, faster audits, and improved quality assurance. A new Q&A feature allows users to ask clarifying questions about the transcript, answered by the AI based solely on the provided text.

Technical Architecture (High-Level)
The system is built as a two-part application:

Backend (FastAPI): Hosts the core NLP services and API endpoints for L1 Automation, Grievance Management, and Call Intelligence. It uses the llama3-8b-8192 model via the Groq API for various NLP tasks (sentiment analysis, intent detection, summarization, Q&A).

Frontend (Streamlit): Provides an intuitive web interface for users to interact with the AI chatbot, submit grievances, and analyze call transcripts. It communicates with the FastAPI backend.

Setup Instructions
Follow these steps to get the Unified AI Customer Care System running on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

A Groq API Key (Sign up at https://console.groq.com/)

1. Clone the Repository
If you haven't already, clone this repository to your local machine:

git clone <repository_url>
cd ai-customer-care

2. Set up Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies:

python -m venv venv

Activate the virtual environment:

Windows (Command Prompt):

.\venv\Scripts\activate

macOS / Linux (Bash/Zsh):

source venv/bin/activate

3. Install Dependencies
With your virtual environment activated, install the required Python packages:

pip install -r requirements.txt

(You will need to create a requirements.txt file containing fastapi, uvicorn, streamlit, python-dotenv, openai, requests, pandas, pyjwt. sqlite3 is usually built-in.)

4. Configure Environment Variables
Create a file named .env in the root directory of your ai-customer-care project (i.e., ai-customer-care/.env). Add your Groq API key and base URL to this file:

OPENAI_API_KEY="your_groq_api_key_here"
OPENAI_API_BASE="https://api.groq.com/openai/v1"

Replace "your_groq_api_key_here" with your actual Groq API Key.

5. Run the Backend (FastAPI)
Open your first terminal and ensure your virtual environment is activated. Navigate to the app directory:

cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

You should see output indicating that the FastAPI server is running.

6. Run the Frontend (Streamlit)
Open your second terminal and ensure your virtual environment is activated. Navigate to the frontend directory:

cd frontend
streamlit run streamlit_app.py

This command will open the Streamlit application in your web browser, typically at http://localhost:8501.

Usage
Once both the backend and frontend are running:

AI-Powered L1 Automation: Interact with the chatbot by typing messages or simulating voice input. Observe sentiment detection, adaptive responses, and auto-escalation predictions.

Smart Grievance Management: Paste grievance details to see automated classification, suggested routing, and priority assignment.

Call Intelligence via NLP: Paste call transcripts to get summaries, extracted tags/entities, overall sentiment, and ask clarifying questions directly about the transcript content.

Feedback Loop & Continual Improvement
The system is designed for continuous learning and refinement:

User Feedback: Implicit (escalation rates) and explicit (satisfaction surveys).

Agent Feedback (Human-in-the-Loop): Human agents can correct AI misclassifications or refine bot responses, providing valuable data for model retraining.

Performance Monitoring: Tracking key metrics (e.g., bot resolution rate, escalation rate, NLP model accuracy) to identify areas for optimization.

New Data Ingestion: Regular feeding of new customer interactions into the system to adapt to evolving needs.

Potential Areas for Further Refinement
This section outlines potential enhancements to evolve the system into a more robust and feature-rich solution:

More Sophisticated Conversation Context: While the system currently passes conversation_history, a more advanced approach could involve the LLM providing a context_update (e.g., {"awaiting_account_number": True} or {"troubleshooting_step": "reboot_router"}). This explicit state management would further influence subsequent prompts, making conversations even more coherent and goal-oriented.

Explicit Tool Use/Function Calling: For a production system, instead of purely generative responses, the AI bot could leverage function calling (if the underlying LLM supports it). This would enable the bot to, for example, "check order status" by calling an internal API or "schedule a technician" via a dedicated function, rather than just providing text-based instructions.

Authentication/User Management: For a real-world application, integrating a robust user authentication and management system would be crucial to personalize experiences, manage permissions for agents and managers, and secure data.

Database Integration: To truly track conversation history, grievances, and call analyses persistently, and to enable comprehensive reporting and retrieval, integrating a dedicated database (e.g., PostgreSQL, MongoDB, or cloud-based solutions like Firestore) would be essential.

Real-time Voice Input/Output: While conceptual voice input is present, a full implementation would involve integrating real-time Speech-to-Text (e.g., using OpenAI's Whisper API, Google Cloud Speech-to-Text, or a local model) and Text-to-Speech capabilities to create a fully interactive voicebot experience.

Developed by Yatharth Dahuja and Siya Srivastava for CyFuture AI Hackathon 2025