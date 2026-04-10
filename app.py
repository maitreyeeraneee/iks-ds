import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    create_gauge_chart, create_line_chart, create_heatmap, create_pie_chart,
    create_triguna_pie, generate_insights, simulate_alert, get_triguna_percentages,
    intervention_engine, calculate_discipline_score, update_streak_and_rl,
    iks_chatbot_response, load_iks_interventions, init_iks_session_state, get_state_key
)
import joblib
import json

# Page config
st.set_page_config(
    page_title="Dopamine Reset",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS glassmorphism
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.main { background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%); font-family: 'Inter', sans-serif; }
.glass { background: rgba(255,255,255,0.05); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.stMetric { background: rgba(255,255,255,0.03); backdrop-filter: blur(10px); border-radius: 15px; padding: 1rem; border: 1px solid rgba(255,255,255,0.1); }
h1, h2, h3 { color: #ffffff; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
.stButton > button { background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); border: none; border-radius: 12px; color: white; font-weight: 600; box-shadow: 0 4px 15px rgba(102,126,234,0.4); }
.stSelectbox, .stNumberInput, .stSlider, .stTextInput { background: rgba(255,255,255,0.05); border-radius: 12px; border: 1px solid rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)

# Init IKS session state
init_iks_session_state()

# Dynamic user history DF - moved to init_iks_session_state if needed

# Model load/train
from model import ModelTrainer
import os

@st.cache_resource
def load_or_train_model(_user_history_hash: str = None):
    # Train if enough history
    if len(st.session_state.user_history) >= 5:
        trainer = ModelTrainer()
        trainer.train(st.session_state.user_history)
        st.success("✅")
        return trainer
    else:
        # Rule-based fallback (improved)
        class ImprovedRuleBasedModel:
            def __init__(self):
                pass
            def predict(self, input_data):
                from utils import get_triguna_percentages
                triguna_dict = get_triguna_percentages(pd.DataFrame([input_data]))
                # Dynamic formula using ALL inputs
                screen = input_data.get('screen_time', 4.0)
                sleep = input_data.get('sleep_hours', 7.0)
                mood = input_data.get('mood', 3.0)
                goal = input_data.get('goal_achieved', 0.7)
                risk_score = 30 + (screen * 8) - (sleep * 4) - (mood * 3) + ((1-goal)*20)
                risk_score = max(0, min(100, risk_score))
                risk_state = 'high' if risk_score > 70 else 'medium' if risk_score > 40 else 'low'
                return {'risk_score': risk_score, 'risk_state': risk_state, 'triguna': triguna_dict[0] if isinstance(triguna_dict, list) else triguna_dict}
        return ImprovedRuleBasedModel()

# Initial load
model = load_or_train_model()

st.markdown("""
<div class='glass' style='padding: 2rem; margin: 1rem; text-align: center;'>
    <h1 style='color: #667eea;'>Dopamine Reset + IKS</h1>
    <p style='color: #b8b8d1;'>Risk Prediction | Triguna Balance | IKS Interventions | RL-Guided Discipline</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Input Status")
    mood = st.slider("🙂 Mood (1-5)", 1, 5, 3, key="slider_mood")
    sleep = st.number_input("😴 Sleep (hrs)", min_value=0.0, max_value=16.0, value=7.0, step=1.0, key="slider_sleep")
    screen = st.number_input("📱 Screen Time (hrs)", min_value=0.0, max_value=16.0, value=4.0, step=1.0, key="slider_screen")
    goal_pct = st.slider("✅ Goal %", 0, 100, 70, key="slider_goal")
    addiction = st.selectbox("Addiction", ['social media', 'gaming', 'food', 'smoking'], key="select_addiction")

    if st.button("🔮 Predict & Analyze", use_container_width=True, key="predict_btn"):
        # Input validation
        input_data = {
            'mood': max(1, min(5, float(mood))),
            'sleep_hours': max(0.1, min(12.0, float(sleep))),
            'screen_time': max(0.0, float(screen)),
            'addiction_type': addiction,
            'goal_achieved': goal_pct / 100.0
        }
        
        # Prepare full feature df for model (match model.py expected cols)
        input_df = pd.DataFrame([input_data])
        input_df['date'] = pd.Timestamp.now()
        input_df['hour'] = input_df['date'].dt.hour
        input_df['is_weekend'] = input_df['date'].dt.weekday >= 5
        input_df['screen_sleep_ratio'] = input_df['screen_time'] / (input_df['sleep_hours'] + 1e-6)
        input_df = input_df.fillna(0)
        
        # Predict with proper input
        try:
            predictions = model.predict(input_data)  # dict for compatibility
            st.success("✅ Prediction successful!")
        except Exception as e:
            st.error(f"Prediction error: {e}. Using fallback.")
            predictions = {'risk_score': 50.0, 'risk_state': 'medium', 'triguna': {'sattva':33, 'rajas':33, 'tamas':34}}
        
        # Triguna
        triguna_pct = get_triguna_percentages(input_df)
        predictions['triguna'] = triguna_pct
        
        st.session_state.predictions = predictions
        st.session_state.input_data = input_df
        
        # Update risk in df and append to history
        input_df['risk_score'] = predictions['risk_score']
        input_df['risk_state'] = predictions['risk_state']
        input_df['triguna_sattva'] = triguna_pct['sattva']
        input_df['triguna_rajas'] = triguna_pct['rajas']
        input_df['triguna_tamas'] = triguna_pct['tamas']
        
        # Append to history
        if st.session_state.user_history.empty:
            st.session_state.user_history = input_df
        else:
            st.session_state.user_history = pd.concat([st.session_state.user_history, input_df], ignore_index=True)
        
        # Retrain model if enough data
        if len(st.session_state.user_history) >= 5:
            st.cache_resource.clear()
            model = load_or_train_model(hash(st.session_state.user_history.to_json()))
        
        st.session_state.triguna_pct = triguna_pct
        st.rerun()

    st.markdown("---")
    # Gamification
    st.markdown("### 🏆 Discipline Tracker")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Streak", st.session_state.streak_days, delta=1)
    with col_s2:
        score = calculate_discipline_score(st.session_state.streak_days, st.session_state.intervention_history)
        st.metric("Score", f"{score:.0f}/100", delta=5)
    
    if st.session_state.intervention_history:
        st.dataframe(pd.DataFrame(st.session_state.intervention_history[-5:]), width='stretch')

    st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Risk Analysis", "Triguna", "Interventions", "Chat & Progress"])

if 'predictions' in st.session_state:
    predictions = st.session_state.predictions
    features_df = st.session_state.input_data
    triguna = st.session_state.get('triguna_pct', {'sattva': 33.33, 'rajas': 33.33, 'tamas': 33.34})
    df_hist = st.session_state.user_history.copy()
    if df_hist.empty:
        df_hist = pd.DataFrame({'date': [pd.Timestamp.now()], 'screen_time': [0]})
    else:
        df_hist['date'] = pd.to_datetime(df_hist['date'])

    with tab1:
        st.markdown("### 🎯 Risk Dashboard")
        try:
            fig_gauge = create_gauge_chart(predictions['risk_score'], "Risk Score")
            st.plotly_chart(fig_gauge, use_container_width=True, height=250, key="risk_gauge")
        except Exception as e:
            st.error(f"Gauge chart error: {e}")
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            st.metric("Risk Score", f"{predictions['risk_score']:.0f}/100", delta=None)
        with col2:
            progress = predictions['risk_score'] / 100
            st.progress(progress)
        with col3:
            dominant_guna = max(triguna, key=triguna.get).upper()
            st.info(f"**{dominant_guna}**")
        
        st.markdown("**Insights:**")
        for insight in generate_insights(predictions, features_df, triguna):
            st.markdown(f"• {insight}")
        
        simulate_alert(predictions['risk_score'], st.session_state)

        st.markdown("### 📊 History")
        chart_cols = st.columns(4)
        required_cols = ['date', 'screen_time', 'risk_score', 'addiction_type']
        safe_hist = df_hist[ [c for c in required_cols if c in df_hist.columns] ].copy() if not df_hist.empty else pd.DataFrame()
        
        try:
            fig_screen = create_line_chart(safe_hist, 'screen_time')
            with chart_cols[0]:
                st.plotly_chart(fig_screen, width='stretch', key="hist_screen")
        except:
            pass
        try:
            fig_heatmap = create_heatmap(safe_hist)
            with chart_cols[1]:
                st.plotly_chart(fig_heatmap, width='stretch', key="hist_heatmap")
        except:
            pass
        try:
            fig_pie = create_pie_chart(safe_hist)
            with chart_cols[2]:
                st.plotly_chart(fig_pie, width='stretch', key="hist_pie")
        except:
            pass
        try:
            fig_risk = create_line_chart(safe_hist, 'risk_score')
            with chart_cols[3]:
                st.plotly_chart(fig_risk, width='stretch', key="hist_risk")
        except:
            pass

    with tab2:
        st.markdown("### ⚖️ Triguna Balance")
        st.markdown("""
**Triguna (Three Gunas from Bhagavad Gita):**
- **Sattva (High = Good):** Purity, peace, wisdom, discipline - supports focus & recovery
- **Rajas (Balanced):** Activity, passion, drive - good for action but excess causes cravings
- **Tamas (Low = Better):** Inertia, confusion, addiction - high tamas triggers relapse
**Goal:** Increase Sattva, balance Rajas, reduce Tamas.
        """)
        dominant = max(triguna, key=triguna.get)
        st.markdown(f"**Dominant: {dominant.upper()}** ({triguna[dominant]:.1f}%)")
        try:
            fig_triguna = create_triguna_pie(triguna)
            st.plotly_chart(fig_triguna, width='stretch', key="triguna_pie")
        except Exception as e:
            st.error(f"Triguna pie error: {e}")
        
        guna_desc = {
            'sattva': "Purity, harmony, discipline (Gita Ch14)",
            'rajas': "Activity, passion, movement",
            'tamas': "Inertia, darkness, addiction trigger"
        }
        for guna, pct in triguna.items():
            st.markdown(f"**{guna.upper()}**: {pct:.0f}% - {guna_desc.get(guna, '')}")

    with tab3:
        st.markdown("### 🛡️ Personalized IKS Interventions")
        iks_data = load_iks_interventions()
        state_key = get_state_key(triguna, predictions['risk_score'], features_df['addiction_type'].iloc[0])
        recs = intervention_engine(predictions['risk_state'], triguna, 
                                  features_df['addiction_type'].iloc[0], st.session_state, predictions)
        
        for i, rec in enumerate(recs[:3]):
            with st.expander(f"#{i+1} {rec['action']}", expanded=(i==0)):
                st.markdown(f"*\"{rec['verse']}\"*")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Success ({rec['action'][:20]}...)", use_container_width=True, key=f"success_btn_{i}"):
                        update_streak_and_rl(True, rec, state_key, st.session_state)
                        st.success("Streak + RL updated! Score ↑")
                        st.rerun()
                with col2:
                    if st.button(f"❌ Failed", use_container_width=True, key=f"failed_btn_{i}"):
                        update_streak_and_rl(False, rec, state_key, st.session_state)
                        st.error("Log noted. Try next time!")
                        st.rerun()
        
        

    with tab4:
        st.markdown("### Progress & Chat")
        
        # Chat interface
        chat_input = st.chat_input("Ask about cravings or IKS guidance...")
        if chat_input:
            st.session_state.chat_messages.append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.write(chat_input)
            
            response = iks_chatbot_response(chat_input, triguna, 
                                          features_df['addiction_type'].iloc[0], st.session_state)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
            st.rerun()
        
        # Chat history
        for msg in st.session_state.chat_messages[-10:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

else:
    st.info("👈 Sidebar: Enter data → Predict & Analyze")

# Footer
st.markdown("---")
