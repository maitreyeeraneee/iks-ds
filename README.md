# Dopamine Reset AI + Indian Knowledge Systems (IKS)

## Production-Ready Streamlit App: Addiction Control with Triguna, Gita/Yoga, RL

### 🚀 Quick Start
1. **Virtual Env** (recommended):
   ```
   python -m venv venv
   venv\\Scripts\\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```
2. **Install**:
   ```
   python -m pip install -r requirements.txt
   ```
3. **Run**:
   ```
   streamlit run app.py
   ```
   Open http://localhost:8501

### 🌟 New IKS Features
- **Triguna Mapping**: ML predicts Sattva/Rajas/Tamas balance
- **Intervention Engine**: Personalized Gita verses + Yoga (Pranayama, Asanas) by addiction/guna
- **RL Optimization**: Tracks success, learns best interventions (Q-learning)
- **Gamification**: Streaks, discipline score
- **IKS Chatbot**: Motivational guidance + quotes
- **Scalable**: OpenAI-ready, wearable stubs

### 📊 Core ML
- Risk Score (RF Regressor) + SHAP
- Multi-addiction support
- Dashboards + alerts

### 🛠️ Architecture
```
iks2/
├── app.py (Streamlit tabs/UI)
├── model.py (RF Risk + Triguna Classifier)
├── utils.py (Charts/RL/IKS engine)
├── config.py
├── data/iks_interventions.json
├── data/raw/sample_data_augmented.csv
└── requirements.txt
```

Production-ready Python code with comments, error handling, session persistence.

