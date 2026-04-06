# Dopamine Reset AI - Configuration for Scalability
# API keys and settings for future integrations (wearables, OpenAI, etc.)

class Config:
    OPENAI_API_KEY = "your-openai-key-here"  # For real chatbot; sim mode now
    
    # Wearable data stubs (future Fitbit/Apple Health)
    WEARABLE_ENABLED = False
    HEART_RATE_THRESHOLD = 90  # bpm for stress detection
    
    # RL sim params
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    
    # IKS Recs Database (expandable)
    INTERVENTIONS = {
        'social media': {
            'high_tamas': ["Bhagavad Gita 6.16: Moderation in use; Practice Pranayama (4-7-8)."],
            'high_rajas': ["Gita 2.62: Dwell on objects → attachment; Short walk + mantra chant."]
        },
        'gaming': {
            'high_tamas': ["Yoga Asana: Savasana 5min; Gita 2.70: Detached observer."],
            'high_rajas': ["Focus task: 1min breath focus; Discipline practice."]
        },
    # Add food, smoking...
    }
    
    IKS_JSON_PATH = "data/iks_interventions.json"
    
    # RL Q-Table init (state: guna_risk, action: intervention_id)
    RL_Q_TABLE = {}
    RL_EPSILON = 0.1  # Exploration

# Global config
config = Config()
