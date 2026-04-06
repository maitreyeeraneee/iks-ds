import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from model import ModelTrainer
from utils import (
    create_gauge_chart, create_line_chart, create_heatmap, create_pie_chart,
    create_triguna_pie, generate_insights, simulate_alert, triguna_mapping,
    intervention_engine, calculate_discipline_score, update_streak_and_rl,
    iks_chatbot_response, load_iks_interventions, init_iks_session_state, get_state_key
)
import joblib
import json

# Page config
st.set_page_config(
    page_title="Dopamine Reset + IKS",
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
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_trained.joblib')
        return model
    except:
        st.warning("No trained model. Using rule-based predictions.")
        class RuleBasedModel:
            def __init__(self):
                pass
            def predict(self, input_data):
                import numpy as np
                from utils import triguna_mapping
                triguna_dict = triguna_mapping(pd.DataFrame([input_data]))[0]
                risk_score = 50 + (input_data['screen_time'] * 5) - (input_data['sleep_hours'] * 3) + np.random.normal(0,5)
                risk_score = np.clip(risk_score, 0, 100)
                risk_state = 'high' if risk_score > 70 else 'medium' if risk_score > 40 else 'low'
                return {'risk_score': risk_score, 'risk_state': risk_state, 'triguna': triguna_dict}
        model = RuleBasedModel()
        return model

model = load_model()

st.markdown("""
<div class='glass' style='padding: 2rem; margin: 1rem; text-align: center;'>
    <h1 style='color: #667eea;'>Dopamine Reset + IKS</h1>
    <p style='color: #b8b8d1;'>Risk Prediction | Triguna Balance | IKS Interventions | RL-Guided Discipline</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Input Status")
    with st.container():
        st.markdown('<div class="glass" style="padding: 1rem;">', unsafe_allow_html=True)
        mood = st.slider("🙂 Mood (1-5)", 1, 5, 3, key="slider_mood")
        sleep = st.slider("😴 Sleep (hrs)", 0.0, 12.0, 7.0, key="slider_sleep")
        screen = st.slider("📱 Screen Time (hrs)", 0.0, 12.0, 4.0, key="slider_screen")
        addiction = st.selectbox("Addiction", ['social media', 'gaming', 'food', 'smoking'], key="select_addiction")
        goal_pct = st.slider("✅ Goal %", 0, 100, 70, key="slider_goal")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 Predict & Analyze", use_container_width=True, key="predict_btn"):
        input_data = {
            'mood': mood, 'sleep_hours': sleep, 'screen_time': screen,
            'addiction_type': addiction, 'goal_achieved': goal_pct / 100
        }
        try:
            predictions = model.predict(input_data)
        except Exception as e:
            st.error(f"Prediction error: {e}. Using fallback.")
            predictions = {'risk_score': 50.0, 'risk_state': 'medium', 'triguna': {'sattva':33, 'rajas':33, 'tamas':34}}
        
        st.session_state.predictions = predictions
        input_df = pd.DataFrame([input_data])
        try:
            triguna_result = triguna_mapping(input_df)
            if isinstance(triguna_result, dict):
                triguna_dict = triguna_result
            elif hasattr(triguna_result, 'iloc'):
                triguna_dict = triguna_result.iloc[0].to_dict() if len(triguna_result) > 0 else {'sattva':33, 'rajas':33, 'tamas':34}
            else:
                triguna_dict = {'sattva':33, 'rajas':33, 'tamas':34}
            input_df['triguna_dict'] = [triguna_dict]
        except Exception as e:
            st.warning(f"Triguna calculation failed: {e}. Using defaults.")
            input_df['triguna_dict'] = [{'sattva':33, 'rajas':33, 'tamas':34}]
        input_df['date'] = pd.Timestamp.now()
        input_df['risk_score'] = predictions['risk_score']
        input_df['risk_state'] = predictions['risk_state']
        
        # Safe column addition
        required_cols = ['date', 'mood', 'sleep_hours', 'screen_time', 'addiction_type', 'goal_achieved', 'risk_score', 'risk_state']
        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0  # or NaN
        
        input_df = input_df.assign(
            triguna_sattva = lambda df: [d.get('sattva', 33) for d in df.triguna_dict],
            triguna_rajas = lambda df: [d.get('rajas', 33) for d in df.triguna_dict],
            triguna_tamas = lambda df: [d.get('tamas', 34) for d in df.triguna_dict]
        ).drop('triguna_dict', axis=1)
        
        # Safe append
        if st.session_state.user_history.empty:
            st.session_state.user_history = input_df[required_cols + ['triguna_sattva', 'triguna_rajas', 'triguna_tamas']]
        else:
            st.session_state.user_history = pd.concat([st.session_state.user_history, input_df], ignore_index=True)
        st.session_state.input_data = input_df
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
        st.dataframe(pd.DataFrame(st.session_state.intervention_history[-5:]), use_container_width=True)

    st.markdown("---")
    # Chatbot
    st.markdown("### 💬 IKS Chatbot")
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Risk Analysis", "Triguna", "Interventions", "Chat & Progress"])

if 'predictions' in st.session_state:
    predictions = st.session_state.predictions
    features_df = st.session_state.input_data
    triguna = { 'sattva': features_df['triguna_sattva'].iloc[0], 'rajas': features_df['triguna_rajas'].iloc[0], 'tamas': features_df['triguna_tamas'].iloc[0] }
    df_hist = st.session_state.user_history.copy()
    if df_hist.empty:
        df_hist = pd.DataFrame({'date': [pd.Timestamp.now()], 'screen_time': [0]})
    else:
        df_hist['date'] = pd.to_datetime(df_hist['date'])

    with tab1:
        st.markdown("### 🎯 Risk Dashboard")
        col1, col2 = st.columns([3,1])
        with col1:
            try:
                fig_gauge = create_gauge_chart(predictions['risk_score'], "Risk Score")
                st.plotly_chart(fig_gauge, use_container_width=True, key="risk_gauge")
            except Exception as e:
                st.error(f"Gauge chart error: {e}")
        with col2:
            st.markdown(f'<div class="glass" style="padding:1.5rem;text-align:center;"><h3>{predictions["risk_state"].upper()}</h3><h2 style="color:#667eea;">{predictions["risk_score"]:.0f}</h2></div>', unsafe_allow_html=True)
        
        st.markdown("**Insights:**")
        for insight in generate_insights(predictions, features_df):
            st.markdown(f"• {insight}")
        
        simulate_alert(predictions['risk_score'], st.session_state)

        st.markdown("### 📊 History")
        cols = st.columns(4)
        with cols[0]: 
            required_cols = ['date', 'screen_time', 'risk_score', 'addiction_type']
            safe_hist = df_hist[ [c for c in required_cols if c in df_hist.columns] ].copy() if not df_hist.empty else pd.DataFrame()
            
            try:
                fig_screen = create_line_chart(safe_hist, 'screen_time')
                st.plotly_chart(fig_screen, use_container_width=True, key="hist_screen")
            except:
                st.empty()
            try:
                fig_heatmap = create_heatmap(safe_hist)
                st.plotly_chart(fig_heatmap, use_container_width=True, key="hist_heatmap")
            except:
                st.empty()
            try:
                fig_pie = create_pie_chart(safe_hist)
                st.plotly_chart(fig_pie, use_container_width=True, key="hist_pie")
            except:
                st.empty()
            try:
                fig_risk = create_line_chart(safe_hist, 'risk_score')
                st.plotly_chart(fig_risk, use_container_width=True, key="hist_risk")
            except:
                st.empty()

    with tab2:
        st.markdown("### ⚖️ Triguna Balance")
        dominant = predictions.get('triguna_dominant', max(triguna, key=triguna.get))
        st.markdown(f"**Dominant Guna: {dominant.upper()}** | {triguna[dominant]:.0f}%")
        try:
            fig_triguna = create_triguna_pie(triguna)
            st.plotly_chart(fig_triguna, use_container_width=True, key="triguna_pie")
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
                    if st.button(f"✅ Success ({rec['action'][:20]}...)", use_container_width=True):
                        update_streak_and_rl(True, rec, state_key, st.session_state)
                        st.success("Streak + RL updated! Score ↑")
                        st.rerun()
                with col2:
                    if st.button(f"❌ Failed", use_container_width=True):
                        update_streak_and_rl(False, rec, state_key, st.session_state)
                        st.error("Log noted. Try next time!")
                        st.rerun()
        
        # Scalability stubs
        st.markdown("---")
        col_api, col_wear = st.columns(2)
        with col_api:
            st.button("🔌 Connect OpenAI Chat", disabled=True)
        with col_wear:
            st.button("⌚ Sync Wearable HR", disabled=True)

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
st.markdown("<p style='text-align:center;color:#808080;'>Production-Ready | IKS + AI + RL | Scalable for APIs/Wearables 🧠🙏</p>", unsafe_allow_html=True)
