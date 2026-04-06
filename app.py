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
    page_title="Dopamine Reset AI + IKS",
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

# Model load/train
@st.cache_resource
def load_model():
    try:
        return joblib.load('model_trained.joblib')
    except:
        with st.spinner("🔄 Training enhanced model with Triguna..."):
            trainer = ModelTrainer()
            trainer.train()
            joblib.dump(trainer, 'model_trained.joblib')
        return joblib.load('model_trained.joblib')

model = load_model()

st.markdown("""
<div class='glass' style='padding: 2rem; margin: 1rem; text-align: center;'>
    <h1 style='color: #667eea;'>🧠 Dopamine Reset AI + Indian Knowledge Systems</h1>
    <p style='color: #b8b8d1;'>AI Risk Prediction | Triguna Balance | IKS Interventions | RL-Guided Discipline</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Input Status")
    with st.container():
        st.markdown('<div class="glass" style="padding: 1rem;">', unsafe_allow_html=True)
        mood = st.slider("🙂 Mood (1-5)", 1, 5, 3)
        sleep = st.slider("😴 Sleep (hrs)", 0.0, 12.0, 7.0)
        screen = st.slider("📱 Screen Time (hrs)", 0.0, 12.0, 4.0)
        addiction = st.selectbox("Addiction", ['social media', 'gaming', 'food', 'smoking'])
        goal_pct = st.slider("✅ Goal %", 0, 100, 70)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 Predict & Analyze", use_container_width=True):
        input_data = {
            'mood': mood, 'sleep_hours': sleep, 'screen_time': screen,
            'addiction_type': addiction, 'goal_achieved': goal_pct / 100
        }
        predictions = model.predict(input_data)
        st.session_state.predictions = predictions
        st.session_state.input_data = pd.DataFrame([input_data])
        st.session_state.input_data['triguna'] = triguna_mapping(st.session_state.input_data)
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
    triguna = features_df['triguna'].iloc[0] if 'triguna' in features_df.columns else triguna_mapping(features_df)
    df_hist = model.processor.load_data().tail(100)

    with tab1:
        st.markdown("### 🎯 Risk Dashboard")
        col1, col2 = st.columns([3,1])
        with col1:
            st.plotly_chart(create_gauge_chart(predictions['risk_score'], "Risk Score"), use_container_width=True)
        with col2:
            st.markdown(f'<div class="glass" style="padding:1.5rem;text-align:center;"><h3>{predictions["risk_state"].upper()}</h3><h2 style="color:#667eea;">{predictions["risk_score"]:.0f}</h2></div>', unsafe_allow_html=True)
        
        st.markdown("**Insights:**")
        for insight in generate_insights(predictions, features_df):
            st.markdown(f"• {insight}")
        
        simulate_alert(predictions['risk_score'], st.session_state)

        st.markdown("### 📊 History")
        cols = st.columns(4)
        with cols[0]: st.plotly_chart(create_line_chart(df_hist, 'screen_time'))
        with cols[1]: st.plotly_chart(create_heatmap(df_hist))
        with cols[2]: st.plotly_chart(create_pie_chart(df_hist))
        with cols[3]: st.plotly_chart(create_line_chart(df_hist, 'craving_level'))

    with tab2:
        st.markdown("### ⚖️ Triguna Balance")
        dominant = predictions.get('triguna_dominant', max(triguna, key=triguna.get))
        st.markdown(f"**Dominant Guna: {dominant.upper()}** | {triguna[dominant]:.0f}%")
        st.plotly_chart(create_triguna_pie(triguna), use_container_width=True)
        
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
                                  features_df['addiction_type'].iloc[0], st.session_state)
        
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
