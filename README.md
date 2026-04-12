# Dopamine Reset + IKS
Production ML App for Addiction Recovery using Indian Knowledge Systems

[![Streamlit App](https://img.shields.io/badge/Live_Demo-orange?style=for-the-badge&logo=streamlit&logoColor=white)](https://iks-ds.streamlit.app/)

## Overview
Advanced Streamlit ML app combining Ayurveda Triguna analysis, Bhagavad Gita interventions, and reinforcement learning for addiction recovery.

**Key Features:**
- Risk prediction dashboards
- Triguna (Sattva/Rajas/Tamas) balance
- Personalized IKS interventions
- Real-time meditation timer
- RL-optimized recommendations

## Tech Stack
- Frontend: Streamlit, Plotly, Custom CSS (glassmorphism)
- ML: scikit-learn (Random Forest), joblib
- Data: Pandas, JSON interventions
- Deployment: Streamlit Cloud ready

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure
```
├── app.py           # Main Streamlit app
├── model.py         # ML model trainer
├── utils.py         # Charts, RL, IKS logic
├── config.py        # Configuration
├── data/            # IKS interventions JSON
├── requirements.txt # Dependencies
├── runtime.txt      # Python 3.10
└── README.md
```

## ML Models
- Random Forest Regressor/Classifier for risk + triguna
- Q-Learning for intervention optimization
- Rule-based fallback for <5 data points

Production-ready with caching, error handling, and session persistence.

---

Made with Streamlit by Maitreyee

