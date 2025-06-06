import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

st.title("📞 Call Intelligence Analyzer")

transcript = st.text_area("Paste the call transcript here:")

if st.button("Analyze"):
    if not transcript.strip():
        st.warning("Please paste a call transcript before analyzing.")
    else:
        with st.spinner("Analyzing call with Groq..."):
            prompt = f"""
You are a customer support quality analyst. Read the following transcript and extract:
- Short summary
- Sentiment (Positive / Neutral / Negative)
- Issue Type (Billing, Technical, Emotional, Other)
- Urgency (Low / Medium / High)

Transcript:
{transcript}
"""
            try:
                response = client.chat.completions.create(
                    model="mistral-saba-24b",
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.choices[0].message.content
                st.success("✅ Analysis Complete:")
                st.text(result)
            except Exception as e:
                st.error(f"❌ Error: {e}")
