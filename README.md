# 📞 Call Intelligence via NLP

A smart customer call analyzer that uses Natural Language Processing (NLP) to extract actionable insights from customer support conversations.

Built with:
- Python
- Streamlit
- Groq API (Mistral model)
- .env for API key management

---

## 🚀 Features

- Analyze call transcripts instantly
- Extract summary, sentiment, issue type, urgency
- Powered by LLM (Mistral 24B via Groq API)
- Clean Streamlit interface

---

## 🛠️ Setup Instructions (Windows)

1. Clone the Repo

   git clone https://github.com/your-username/call-intelligence-nlp.git
   cd call-intelligence-nlp

2. Create Virtual Environment

   python -m venv venv
   venv\Scripts\activate

3. Install Dependencies

   pip install -r requirements.txt

4. Create .env file

   Create a file named `.env` in the root folder and add:

   OPENAI_API_KEY=gsk_your_actual_groq_key_here
   OPENAI_API_BASE=https://api.groq.com/openai/v1

5. Run the App

   streamlit run app.py

   Open your browser and go to http://localhost:8501

---

## 🧪 Sample Transcript to Try

Customer: My internet is down since this morning.
Agent: I'm sorry. Can I get your account ID?
Customer: Sure, it's 12345.
Agent: I've raised a ticket. You'll get updates soon.

Paste it in the box and click "Analyze"

---

## 📂 Project Structure

call-intelligence-nlp/
├── app.py               # Main Streamlit app
├── .env                 # API keys (NOT to be shared)
├── requirements.txt     # Required Python packages
└── README.md            # This file

---

## 🔐 Security Tip

NEVER upload your `.env` or API keys to GitHub or any repo.  
Make sure `.env` is in `.gitignore`.

---

## 📈 Future Enhancements

- Audio transcription support
- Save results to CSV
- Graphs and metrics dashboard
- Deploy to Streamlit Cloud or Render

---

## 🙋‍♂️ Built By

Created by [Your Name] and team during [Hackathon Name]  
Assisted by ChatGPT + Groq for prototyping

---

## 📄 License

MIT License. Free to use, share, and remix.

