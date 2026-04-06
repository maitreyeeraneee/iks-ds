# Dopamine Reset AI + IKS Enhancement TODO

Status: Plan Approved ✅

## Step-by-Step Implementation Plan

### 1. [✅] Data & Config Preparation
   - Augment sample_data.csv → data/raw/sample_data_augmented.csv with triguna columns/labels (sattva/rajas/tamas targets)
   - Create data/iks_interventions.json (structured IKS recs)
   - Update requirements.txt (add gymnasium==0.29.1 for RL sim)
   - Update config.py (add IKS_JSON_PATH, RL_Q_TABLE, EPSILON)

### 2. [✅] Enhance utils.py (Intervention Engine + Gamification + RL + Chat)
   - Add imports (json, config, gymnasium)
   - New: load_iks_json, rl_get_best_action/update_q, get_state_key, create_triguna_pie, init_iks_session_state
   - Enhanced: triguna_mapping, intervention_engine (RL+JSON), calculate_discipline_score/update_streak_rl, iks_chatbot
   - Session state for rl_q_table, history, streak, chat

### 3. [✅] Update model.py (Triguna ML Integration)
   - DataProcessor: Load augmented.csv, add dominant_guna, triguna numerical
   - ModelTrainer: Risk regressor + triguna_classifier (RF + LabelEncoder)
   - Predict: risk_score/state + triguna dict/probs/dominant, SHAP

### 4. [✅] Update app.py (Full UI Integration)
   - Sidebar: Status inputs + Gamification metrics (streak/score/history)
   - Tabs: Risk Dashboard (gauge/insights/charts), Triguna pie/desc, Interventions (RL recs + success/fail buttons → update RL/streak), Chatbot (chat_input + history)
   - Full integration: model predict → triguna/interventions/gamification/RL loop
   - Scalability stubs (OpenAI/wearable buttons)

### 5. [✅] Testing & Polish
   - Model auto-retrains on first run (augmented data + Triguna)
   - End-to-end tested: Predict → Triguna → IKS recs → Success button → RL Q-update + streak/score
   - README.md enhanced with full features/setup
   - All features production-ready: Multi-addiction, RL loop, chat, scalability stubs

**TASK COMPLETE ✅**

