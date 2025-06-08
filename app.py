import streamlit as st
import numpy as np
import joblib

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

if st.button("Get Recommendation"):
    if user_skill == '':
        st.warning("Please enter a skill.")
    else:
        # Check if the skill is in the encoder classes
        # Note: le_skill.classes_ might be capitalized or formatted, so we normalize both
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
