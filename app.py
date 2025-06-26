import streamlit as st
import numpy as np
import joblib
import os
import requests
from dotenv import load_dotenv

# Securely load Mistral API key
try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except (AttributeError, KeyError):
    load_dotenv()
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Function to call Mistral LLM API
def get_career_recommendations_llm(skill):
    url = "https://api.mistral.ai/v1/chat/completions"  # Example endpoint, update if needed
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Given the following skill: {skill}, suggest 3 suitable career options with a short explanation for each."
    data = {
        "model": "mistral-tiny",  # Update to your preferred model
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error contacting Mistral API: {e}"

@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("model/model.pkl")
    le_skill = joblib.load("model/skill_encoder.pkl")
    le_career = joblib.load("model/career_encoder.pkl")
    return model, le_skill, le_career

model, le_skill, le_career = load_model_and_encoders()

# Get list of skills known to the encoder for validation
skill_list = list(le_skill.classes_)

st.title("Career Recommendation System")
st.write("Enter your skill to get recommended career options.")

# User input
user_skill = st.text_input("Enter your skill (e.g. Python, Data Analysis, Web Development):").strip().lower()

col1, col2 = st.columns(2)

with col1:
    if st.button("Get ML Recommendation"):
        if user_skill == '':
            st.warning("Please enter a skill.")
        else:
            # Check if the skill is in the encoder classes
            skill_matches = [skill for skill in skill_list if skill.lower() == user_skill]

            if len(skill_matches) == 0:
                st.error(f"Skill '{user_skill}' not found in our database. Please enter a valid skill.")
            else:
                skill_to_predict = skill_matches[0]
                # Encode the skill
                skill_encoded = le_skill.transform([skill_to_predict])[0]
                # Predict probabilities for all careers
                probs = model.predict_proba([[skill_encoded]])[0]
                # Get top 3 career predictions
                top3_idx = np.argsort(probs)[-3:][::-1]
                st.success(f"Top career recommendations for skill '{skill_to_predict}':")
                for idx in top3_idx:
                    career_name = le_career.inverse_transform([idx])[0]
                    confidence = probs[idx] * 100
                    st.write(f"- {career_name} (Confidence: {confidence:.2f}%)")

with col2:
    if st.button("Get LLM Recommendation (Mistral)"):
        if user_skill == '':
            st.warning("Please enter a skill.")
        else:
            with st.spinner("Contacting Mistral LLM API..."):
                llm_result = get_career_recommendations_llm(user_skill)
            st.markdown("**Mistral LLM Career Suggestions:**")
            st.write(llm_result)
