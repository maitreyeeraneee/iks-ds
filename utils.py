
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from config import config
from typing import Dict, Any, List
import gymnasium as gym
from gymnasium import spaces

def create_gauge_chart(score: float, title: str = "Addiction Risk Score") -> go.Figure:
    """Create interactive gauge chart for risk score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "darkred"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=400, font=dict(size=12))
    return fig

def create_line_chart(df: pd.DataFrame, col: str) -> go.Figure:
    """Line chart for usage over time - safe."""
    if df.empty or col not in df.columns or 'date' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data for chart. Add predictions to see trends.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template='plotly_dark', height=350, title=f'{col.capitalize()} Trend')
        return fig
    df_plot = df.tail(30).copy()
    df_plot['date'] = pd.to_datetime(df_plot['date'])
    fig = px.line(df_plot, x='date', y=col, title=f'{col.capitalize()} Over Time', markers=True)
    fig.update_layout(template='plotly_dark', height=350)
    fig.update_traces(line_color='#636EFA')
    return fig

def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap for craving patterns by hour/day - safe."""
    if df.empty or 'date' not in df.columns or 'risk_score' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="More data needed for heatmap (3+ days).", xref="paper", yref="paper")
        fig.update_layout(template='plotly_dark', height=350, title='Craving Heatmap')
        return fig
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day_name()
    pivot = df.pivot_table(index='day', columns='hour', values='risk_score', aggfunc='mean')
    fig = px.imshow(pivot.fillna(0), title='Risk Heatmap (Hour vs Day)', color_continuous_scale='RdYlGn_r')
    fig.update_layout(template='plotly_dark', height=350)
    return fig

def create_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Pie chart for addiction type distribution - safe."""
    if df.empty or 'addiction_type' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Log more entries to see distribution.", xref="paper", yref="paper")
        fig.update_layout(template='plotly_dark', height=350, title='Addiction Types')
        return fig
    type_counts = df['addiction_type'].value_counts()
    fig = px.pie(values=type_counts.values, names=type_counts.index, title='Addiction Type Distribution')
    fig.update_layout(template='plotly_dark', height=350)
    return fig

def create_triguna_pie(triguna: Dict[str, float]) -> go.Figure:
    """Pie chart for Triguna distribution."""
    fig = px.pie(values=list(triguna.values()), names=list(triguna.keys()), 
                 title='Your Triguna Balance', color_discrete_map={'sattva': 'lightgreen', 'rajas': 'orange', 'tamas': 'darkred'})
    fig.update_layout(template='plotly_dark', height=350)
    return fig

def generate_insights(predictions: Dict[str, Any], features: pd.DataFrame) -> list:
    """Generate textual insights from predictions."""
    score = predictions['risk_score']
    state = predictions['risk_state']
    triguna = predictions.get('triguna', {})
    insights = [f"Risk: {state.upper()} ({score:.1f}/100)"]
    
    if triguna:
        dominant = max(triguna, key=triguna.get)
        insights.append(f"dominant Guna: {dominant.upper()} ({triguna[dominant]:.0f}%)")
    
    screen = features['screen_time'].iloc[0]
    sleep = features['sleep_hours'].iloc[0]
    if screen > 5 and sleep < 6:
        insights.append("⚠️ High screen + low sleep = relapse risk ↑")
    
    return insights

def simulate_alert(risk_score: float, session_state: Any):
    """Real-time high-risk alert."""
    if risk_score > 70 and not session_state.get('alert_active', False):
        session_state.alert_active = True
        with st.container():
            st.error("🚨 HIGH RISK! Practice IKS intervention now.")
            if st.button("Reset Alert", use_container_width=True, key="reset_alert_btn"):
                session_state.alert_active = False
                st.rerun()
    elif risk_score <= 70:
        session_state.alert_active = False

def load_iks_interventions() -> Dict:
    """Load IKS interventions from JSON."""
    try:
        with open(config.IKS_JSON_PATH, 'r') as f:
            return json.load(f)
    except:
        st.warning("IKS JSON not found, using fallback.")
        return {}

def get_state_key(triguna: Dict, risk_score: float, addiction_type: str) -> str:
    """RL state: dominant_guna_risk_addiction."""
    dominant = max(triguna, key=triguna.get)
    risk_level = 'high' if risk_score > 70 else 'med' if risk_score > 40 else 'low'
    return f"{dominant}_{risk_level}_{addiction_type}"

def rl_get_best_action(state_key: str, interventions: List[Dict], q_table: Dict, epsilon: float = 0.1) -> Dict:
    """Epsilon-greedy action selection."""
    if state_key not in q_table:
        q_table[state_key] = {i: 0.0 for i in range(len(interventions))}
    
    if np.random.random() < epsilon:
        return np.random.choice(interventions)
    
    q_values = q_table[state_key]
    best_idx = max(q_values, key=q_values.get)
    return interventions[best_idx]

def rl_update_q(state_key: str, action_idx: int, reward: float, next_interventions: List[Dict], 
                q_table: Dict, lr: float = config.LEARNING_RATE, gamma: float = config.DISCOUNT_FACTOR):
    """Q-learning update."""
    if state_key not in q_table:
        q_table[state_key] = {i: 0.0 for i in range(len(next_interventions))}
    
    current_q = q_table[state_key][action_idx]
    next_max_q = max(q_table[state_key].values()) if next_interventions else 0
    new_q = current_q + lr * (reward + gamma * next_max_q - current_q)
    q_table[state_key][action_idx] = new_q

def triguna_mapping(features: pd.DataFrame) -> list[dict]:
    """Enhanced Triguna mapping - safe for DataFrame input, returns list of dicts for row-wise."""
    if features.empty:
        return [{'sattva': 33.3, 'rajas': 33.3, 'tamas': 33.4}]
    
    result = []
    for idx in range(len(features)):
        row = features.iloc[idx]
        try:
            mood = row.get('mood', 3.0)
            sleep = row.get('sleep_hours', 7.0)
            screen = row.get('screen_time', 4.0)
            goal = row.get('goal_achieved', 0.7)
            
            sattva_score = (mood / 5 * sleep / 8 * goal) * 100
            tamas_score = ((5 - mood) / 5 * screen / 8 * (1 - goal)) * 100
            rajas_score = 50.0
            total = sattva_score + tamas_score + rajas_score
            if total > 0:
                sattva_score = (sattva_score / total) * 100
                rajas_score = (rajas_score / total) * 100
                tamas_score = (tamas_score / total) * 100
            
            result.append({'sattva': sattva_score, 'rajas': rajas_score, 'tamas': tamas_score})
        except Exception:
            result.append({'sattva': 33.3, 'rajas': 33.3, 'tamas': 33.4})
    
    return result

def intervention_engine(risk_state: str, triguna: Dict[str, float], addiction_type: str, 
                       session_state: Any, predictions=None, iks_data: Dict = None) -> List[Dict]:
    """Dynamic IKS interventions with RL."""
    if iks_data is None:
        iks_data = load_iks_interventions()
    
    dominant_guna = max(triguna, key=triguna.get)
    risk_score = predictions['risk_score'] if predictions else 50
    state_key = get_state_key(triguna, risk_score, addiction_type)
    
    interventions = iks_data.get(addiction_type, {}).get(dominant_guna, [])
    
    # RL best action
    best_int = rl_get_best_action(state_key, interventions, session_state.get('rl_q_table', config.RL_Q_TABLE))
    
    recs = [best_int] + interventions[:2]  # Best + top 2
    return recs

def calculate_discipline_score(streak_days: int, history: List) -> float:
    """Gamification score."""
    if not history:
        return streak_days * 5
    success_rate = sum(1 for h in history if h['success']) / len(history)
    return min(100, streak_days * 4 + success_rate * 60)

def update_streak_and_rl(success: bool, intervention: Dict, state_key: str, session_state: Any):
    """Update streak and RL on user feedback."""
    # Streak
    if success:
        session_state.streak_days = session_state.get('streak_days', 0) + 1
        reward = 1.0
    else:
        session_state.streak_days = 0
        reward = -0.5
    
    # History
    session_state.intervention_history = session_state.get('intervention_history', []) + [{
        'success': success, 'intervention': intervention['action'], 'time': pd.Timestamp.now()
    }]
    
    # RL update (action_idx approximate as len)
    action_idx = 0  # Simplify; hash or id later
    rl_update_q(state_key, action_idx, reward, [], session_state.get('rl_q_table', config.RL_Q_TABLE))
    
    session_state.rl_q_table = config.RL_Q_TABLE  # Persist

def iks_chatbot_response(query: str, triguna: Dict, addiction_type: str, session_state: Any) -> str:
    """Advanced IKS chatbot (sim + OpenAI stub)."""
    dominant = max(triguna, key=triguna.get).upper()
    
    import random
    prompts = [
        f"Craving {addiction_type}? Dominant Guna: {dominant}. Gita 2.56: Steady wisdom amid disturbance. Try Nadi Shodhana pranayama.",
        f"Gita Ch6 self-control for {addiction_type}. Practice: Ujjayi breath. You master the mind (6.36).",
        f"Detach (Gita 2.70). For {dominant} guna, {random.choice(['walk', 'chant Om', 'Savasana'])}. Namaste 🙏",
        f"Yoga Sutra: Tame fluctuations. {addiction_type} craving? Focus breath 1min."
    ]
    
    response = np.random.choice(prompts)
    
    # OpenAI stub
    if config.OPENAI_API_KEY != "your-openai-key-here":
        # Future: openai.ChatCompletion...
        pass
    
    return response

def init_iks_session_state():
    """Initialize session state for IKS features."""
    defaults = {
        'streak_days': 0,
        'intervention_history': [],
        'rl_q_table': config.RL_Q_TABLE.copy(),
        'chat_messages': [],
        'discipline_score': 0,
        'user_history': pd.DataFrame(columns=['date', 'mood', 'sleep_hours', 'screen_time', 'addiction_type', 'goal_achieved', 'risk_score', 'risk_state', 'triguna_sattva', 'triguna_rajas', 'triguna_tamas'])
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

if __name__ == '__main__':
    init_iks_session_state()
