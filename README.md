# 🎓 Career Guru – AI Career Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://career-guru-fpdaqrezuvdhi4qsgrjoww.streamlit.app/)

**Try it online:**  
[https://career-guru-fpdaqrezuvdhi4qsgrjoww.streamlit.app/](https://career-guru-fpdaqrezuvdhi4qsgrjoww.streamlit.app/)

---

## 📖 Overview
Career Guru is a machine learning– and LLM-powered career recommendation system that suggests suitable career paths based on a user's skills. It simplifies the decision-making process for students and job seekers by leveraging both predictive modeling and generative AI.

---

## 🚀 Features
- Predicts career options based on user-entered skills using a trained ML model
- **NEW:** Suggests career options using the Mistral LLM API for more creative, generative recommendations
- Uses Label Encoding and a classification model for prediction
- Interactive frontend built using Streamlit
- Simple and intuitive UI for user interaction

---

## 🧠 Tech Stack
- Python, Pandas, NumPy
- scikit-learn
- Streamlit
- joblib
- **Mistral LLM API**
- requests, python-dotenv

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/babneek/career-guru.git
cd career-guru
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your Mistral API key
Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## ☁️ Deploying on Streamlit Cloud
1. Go to your app's settings on Streamlit Cloud.
2. Add your Mistral API key in the **Secrets** section:
   ```
   MISTRAL_API_KEY = your_mistral_api_key_here
   ```
3. Deploy your app as usual.

---

## 🧩 How to Use
1. Enter your skill in the web app.
2. Click **Get ML Recommendation** to see model-based career options.
3. Click **Get LLM Recommendation (Mistral)** to get creative suggestions from the Mistral LLM.
4. Explore the suggestions and plan your career path!

---

## 📁 Folder Structure
```
career-guru/
├── app.py
├── model/
│   ├── model.pkl
│   ├── skill_encoder.pkl
│   └── career_encoder.pkl
├── data/
│   └── career_data.xlsx
├── requirements.txt
├── README.md
```

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## 📄 License
This project is licensed under the MIT License.

---

## 🙏 Acknowledgements
- [Streamlit](https://streamlit.io/) for the web app framework
- [Python](https://www.python.org/) for the programming language
- [Mistral AI](https://mistral.ai/) for the LLM API
