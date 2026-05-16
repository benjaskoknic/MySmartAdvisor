"""
MySmartAdvisor - Behavioral Finance Robo-Advisor
A fintech MVP that profiles users psychologically and recommends investment portfolios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"

st.set_page_config(
    page_title="MySmartAdvisor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Fintech Premium Dark/Light
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ══════════════════════════════════════════════
   DESIGN TOKENS — LIGHT (default)
   ══════════════════════════════════════════════ */
:root {
    --blue-primary:      #1A56DB;
    --blue-dark:         #0F3A8A;
    --red-accent:        #E53E3E;
    --success:           #10B981;
    --warning:           #F59E0B;

    --bg-canvas:         #FFFFFF;
    --bg-card:           #F3F4F6;
    --bg-card-blue:      linear-gradient(135deg, #EBF2FF, #DBEAFE);
    --border-card-blue:  #BFDBFE;
    --bg-sidebar:        #F0F8FF;
    --bg-sidebar-card:   #C1E2FF;
    --bg-question:       #F3F4F6;
    --border-question:   #DBEAFE;
    --border-divider:    #E5E7EB;
    --border-input:      #E5E7EB;
    --bg-radio:          #F3F4F6;
    --bg-radio-hover:    #F0F8FF;
    --bg-progress:       #E5E7EB;
    --bg-disclaimer:     #FEF2F2;
    --border-disclaimer: #E53E3E;
    --text-disclaimer:   #7F1D1D;
    --text-body:         #4B5563;
    --text-heading:      #1F2937;
    --text-heading-dark: #111827;
    --text-muted:        #6B7280;
    --text-sidebar-body: #374151;
    --shadow-card:       0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
    --shadow-metric:     0 2px 8px rgba(0,0,0,0.05);
    --shadow-question:   0 2px 12px rgba(26,86,219,0.07);
    --chart-grid:        #E5E7EB;
    --sidebar-title-color: #0F3A8A;
}

/* ══════════════════════════════════════════════
   DESIGN TOKENS — DARK OVERRIDE
   Streamlit sets data-theme="dark" on <html> when
   config.toml has base="dark" OR user picks dark.
   ══════════════════════════════════════════════ */
[data-theme="dark"] {
    --blue-primary:      #3B82F6;
    --blue-dark:         #1D4ED8;
    --red-accent:        #F87171;
    --success:           #34D399;
    --warning:           #FBBF24;

    --bg-canvas:         #0D1B2A;
    --bg-card:           #162032;
    --bg-card-blue:      linear-gradient(135deg, #152240, #1A2D50);
    --border-card-blue:  #1E3A6E;
    --bg-sidebar:        #0F1E30;
    --bg-sidebar-card:   #1A2D50;
    --bg-question:       #162032;
    --border-question:   #1E3A6E;
    --border-divider:    #1E3A6E;
    --border-input:      #2A4070;
    --bg-radio:          #162032;
    --bg-radio-hover:    #1A2D50;
    --bg-progress:       #1E3A6E;
    --bg-disclaimer:     #2D1515;
    --border-disclaimer: #C53030;
    --text-disclaimer:   #FEB2B2;
    --text-body:         #A0AEC0;
    --text-heading:      #E2E8F0;
    --text-heading-dark: #F7FAFC;
    --text-muted:        #718096;
    --text-sidebar-body: #CBD5E0;
    --shadow-card:       0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.3);
    --shadow-metric:     0 2px 8px rgba(0,0,0,0.3);
    --shadow-question:   0 2px 12px rgba(0,0,0,0.3);
    --chart-grid:        #1E3A6E;
    --sidebar-title-color: #93C5FD;
}

/* ══════════════════════════════════════════════
   STREAMLIT CHROME
   ══════════════════════════════════════════════ */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-canvas) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
}
[data-testid="stHeader"]     { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ══════════════════════════════════════════════
   GLOBAL
   ══════════════════════════════════════════════ */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4              { font-family: 'Syne', sans-serif; }
footer                      { visibility: hidden; }

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ══════════════════════════════════════════════
   HERO BANNER  (always dark-blue gradient — no change)
   ══════════════════════════════════════════════ */
.hero-banner {
    background: linear-gradient(135deg, #0F3A8A 0%, #1A56DB 60%, #2563EB 100%);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 250px; height: 250px;
    background: rgba(229,62,62,0.12);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(229,62,62,0.9);
    color: white;
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ══════════════════════════════════════════════
   CARDS
   ══════════════════════════════════════════════ */
.card {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 1.8rem;
    border: 1px solid var(--border-divider);
    box-shadow: var(--shadow-card);
    margin-bottom: 1.2rem;
}
.card-blue {
    background: var(--bg-card-blue);
    border: 1px solid var(--border-card-blue);
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-heading);
    margin-bottom: 0.5rem;
}

/* ══════════════════════════════════════════════
   METRIC CARDS
   ══════════════════════════════════════════════ */
.metric-row   { display: flex; gap: 1rem; margin-bottom: 1.2rem; }
.metric-card  {
    flex: 1;
    background: var(--bg-card);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1px solid var(--border-divider);
    box-shadow: var(--shadow-metric);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-heading-dark);
}
.metric-delta { font-size: 0.8rem; color: var(--success); font-weight: 500; }

/* ══════════════════════════════════════════════
   RISK SCORE
   ══════════════════════════════════════════════ */
.risk-score-display  { text-align: center; padding: 2rem; }
.risk-score-number   { font-family: 'Syne', sans-serif; font-size: 4.5rem; font-weight: 800; color: var(--blue-primary); line-height: 1; }
.risk-score-label    { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px; margin-top: 0.3rem; }

/* ══════════════════════════════════════════════
   PROFILE BADGE
   ══════════════════════════════════════════════ */
.profile-badge {
    display: inline-block;
    padding: 0.6rem 1.4rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    margin: 1rem 0;
}

/* ══════════════════════════════════════════════
   QUESTION CARD
   ══════════════════════════════════════════════ */
.question-card {
    background: var(--bg-question);
    border-radius: 16px;
    padding: 2rem;
    border: 1.5px solid var(--border-question);
    margin-bottom: 1rem;
    box-shadow: var(--shadow-question);
}
.question-number {
    font-size: 0.7rem;
    color: var(--blue-primary);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}
.question-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-heading);
    line-height: 1.4;
}

/* ══════════════════════════════════════════════
   SIDEBAR CARDS
   ══════════════════════════════════════════════ */
.sidebar-section {
    background: var(--bg-sidebar-card);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    border-left: 3px solid var(--blue-primary);
}
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--sidebar-title-color);
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.sidebar-body { font-size: 0.82rem; color: var(--text-sidebar-body); line-height: 1.5; }

.sidebar-disclaimer {
    background: var(--bg-disclaimer);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    border-left: 3px solid var(--border-disclaimer);
    font-size: 0.75rem;
    color: var(--text-disclaimer);
    line-height: 1.5;
    margin-top: 1rem;
}

/* ══════════════════════════════════════════════
   DIVIDER & PROGRESS
   ══════════════════════════════════════════════ */
.section-divider { border: none; border-top: 1px solid var(--border-divider); margin: 2rem 0; }
.progress-container { background: var(--bg-progress); border-radius: 10px; height: 6px; margin: 0.8rem 0; }
.progress-fill { height: 6px; border-radius: 10px; background: linear-gradient(90deg, var(--blue-primary), #60A5FA); transition: width 0.5s ease; }

/* ══════════════════════════════════════════════
   ANIMATIONS
   ══════════════════════════════════════════════ */
@keyframes fadeSlideUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
.animate-in { animation: fadeSlideUp 0.5s ease forwards; }

/* ══════════════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
   ══════════════════════════════════════════════ */

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1A56DB, #2563EB);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2.5rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(26,86,219,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0F3A8A, #1A56DB);
    box-shadow: 0 6px 20px rgba(26,86,219,0.4);
    transform: translateY(-1px);
}

/* ── Radio buttons ── */
.stRadio > div { gap: 0.6rem; }
.stRadio label {
    background: var(--bg-radio);
    border: 1.5px solid var(--border-input);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    cursor: pointer;
    transition: all 0.15s ease;
    font-size: 0.92rem;
    color: var(--text-body);
}
.stRadio label:hover {
    border-color: var(--blue-primary);
    background: var(--bg-radio-hover);
}

/* ── Slider & Selectbox ── */
.stSlider > div > div     { color: var(--blue-primary); }
.stSelectbox > div > div  { border-radius: 10px; border-color: var(--border-input); }

/* ── Number input & selectbox ── */
input[type="number"],
.stSelectbox > div > div {
    border-color: var(--border-input) !important;
    background-color: var(--bg-card) !important;
    color: var(--text-heading) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border-color: var(--border-input) !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    background: var(--bg-card);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & ML
# ─────────────────────────────────────────────

@st.cache_data
def generate_synthetic_dataset(n=600):
    """Generate a synthetic behavioral finance dataset with psychological coherence."""
    np.random.seed(42)
    data = []

    for _ in range(n):
        # Each feature captures a behavioral/psychological dimension
        loss_aversion     = np.random.uniform(0, 10)   # higher = more loss-averse
        prob_sensitivity  = np.random.uniform(0, 10)   # higher = prefers certainty
        volatility_react  = np.random.uniform(0, 10)   # higher = more reactive
        time_preference   = np.random.uniform(0, 10)   # higher = more patient
        uncertainty_avoid = np.random.uniform(0, 10)   # higher = avoids ambiguity
        risk_reward_bias  = np.random.uniform(0, 10)   # higher = seeks high reward

        # Compute a raw risk score using behavioral rules
        raw = (
            (10 - loss_aversion) * 0.25 +
            (10 - prob_sensitivity) * 0.15 +
            (10 - volatility_react) * 0.20 +
            time_preference * 0.20 +
            (10 - uncertainty_avoid) * 0.10 +
            risk_reward_bias * 0.10
        )
        raw = np.clip(raw + np.random.normal(0, 0.5), 1, 10)

        # Map to 5-category profile
        if raw <= 2:   profile = 0  # Muy Conservador
        elif raw <= 4: profile = 1  # Conservador
        elif raw <= 6: profile = 2  # Moderado
        elif raw <= 8: profile = 3  # Agresivo
        else:          profile = 4  # Muy Agresivo

        data.append([
            loss_aversion, prob_sensitivity, volatility_react,
            time_preference, uncertainty_avoid, risk_reward_bias,
            raw, profile
        ])

    cols = [
        'loss_aversion', 'prob_sensitivity', 'volatility_react',
        'time_preference', 'uncertainty_avoid', 'risk_reward_bias',
        'raw_score', 'perfil_riesgo'
    ]
    return pd.DataFrame(data, columns=cols)


@st.cache_resource
def train_model():
    """Train the Decision Tree Classifier."""
    df = generate_synthetic_dataset()
    features = ['loss_aversion', 'prob_sensitivity', 'volatility_react',
                 'time_preference', 'uncertainty_avoid', 'risk_reward_bias']
    X = df[features]
    y = df['perfil_riesgo']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, features, df


@st.cache_data(ttl=3600)
def download_market_data(tickers, period="5y"):
    """Download historical price data via yfinance."""
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw[['Close']]
            prices.columns = tickers
        prices.dropna(how='all', inplace=True)
        # Forward-fill minor gaps
        prices.ffill(inplace=True)
        return prices
    except Exception as e:
        st.warning(f"No se pudieron cargar datos de mercado: {e}")
        return None


# ─────────────────────────────────────────────
# PORTFOLIO DEFINITIONS
# ─────────────────────────────────────────────

PORTFOLIOS = {
    0: {
        "name": "Muy Conservador",
        "emoji": "🛡️",
        "color": "#1A56DB",
        "assets": {"BND": 0.60, "GLD": 0.20, "SPY": 0.20},
        "psychology": "Tienes una alta aversión a pérdidas y priorizas la preservación del capital. La prospect theory sugiere que el dolor de perder supera ampliamente la satisfacción de ganar en tu caso.",
        "behavioral": "Muestras fuertes sesgos de certidumbre y aversión a la ambigüedad. Tus decisiones bajo incertidumbre tienden a ser muy defensivas.",
        "horizon": "< 2 años",
        "expected_return": "4–6% anual",
    },
    1: {
        "name": "Conservador",
        "emoji": "⚓",
        "color": "#2563EB",
        "assets": {"BND": 0.50, "SPY": 0.35, "GLD": 0.15},
        "psychology": "Valoras la seguridad pero tienes cierta apertura al crecimiento. Tu tolerancia al riesgo es limitada pero racional, con un fuerte anclaje emocional en la estabilidad.",
        "behavioral": "Exhibes sesgo de status quo moderado. Prefieres rendimientos predecibles sobre oportunidades de alto potencial con incertidumbre.",
        "horizon": "2–4 años",
        "expected_return": "6–8% anual",
    },
    2: {
        "name": "Moderado",
        "emoji": "⚖️",
        "color": "#0EA5E9",
        "assets": {"SPY": 0.50, "QQQ": 0.30, "BND": 0.20},
        "psychology": "Balanceas bien riesgo y retorno. Tu función de valor es relativamente simétrica entre pérdidas y ganancias, con buena resiliencia emocional ante volatilidad.",
        "behavioral": "Muestras preferencias temporales saludables. Tu capacidad para diferir gratificación te permite capturar premios de riesgo de largo plazo.",
        "horizon": "4–7 años",
        "expected_return": "8–11% anual",
    },
    3: {
        "name": "Agresivo",
        "emoji": "🚀",
        "color": "#F59E0B",
        "assets": {"QQQ": 0.60, "SPY": 0.25, "IWM": 0.15},
        "psychology": "Eres cómodo con la volatilidad y buscas rendimientos superiores. Tu curva de utilidad es cóncava pero con baja sensibilidad a pérdidas intermedias.",
        "behavioral": "Exhibes alta tolerancia a la incertidumbre y buena capacidad de razonamiento probabilístico. Tus decisiones son más analíticas que emocionales.",
        "horizon": "7–10 años",
        "expected_return": "11–14% anual",
    },
    4: {
        "name": "Muy Agresivo",
        "emoji": "⚡",
        "color": "#E53E3E",
        "assets": {"QQQ": 0.70, "IWM": 0.20, "SPY": 0.10},
        "psychology": "Maximizas retorno esperado sobre seguridad. Tu apetito por riesgo es excepcional, con baja aversión a pérdidas y alta sensación de control sobre el mercado.",
        "behavioral": "Muestras confianza extrema y alta tolerancia a la ambigüedad. Cuidado con el sesgo de exceso de confianza (overconfidence bias) que puede nublar la gestión de riesgo.",
        "horizon": "> 10 años",
        "expected_return": "14–18% anual",
    },
}

PROFILE_LABELS = ["Muy Conservador", "Conservador", "Moderado", "Agresivo", "Muy Agresivo"]


# ─────────────────────────────────────────────
# QUESTIONNAIRE
# ─────────────────────────────────────────────

QUESTIONS = [
    {
        "id": "q1",
        "text": "Tu inversión cae un 25% en 3 semanas por turbulencia de mercado. ¿Cuál es tu reacción instintiva?",
        "behavioral": "loss_aversion",
        "options": [
            ("Vendo todo inmediatamente — no puedo soportar más pérdidas", 0),
            ("Espero con nerviosismo pero no actúo", 3),
            ("Mantengo mi estrategia original sin alteraciones", 6),
            ("Analizo si es oportunidad y considero comprar más", 9),
        ]
    },
    {
        "id": "q2",
        "text": "¿Cuál de estas opciones prefieres?",
        "behavioral": "prob_sensitivity",
        "options": [
            ("Recibir $5.000 garantizados hoy", 1),
            ("70% de probabilidad de ganar \$8.000 (sino, \$0)", 4),
            ("50% de probabilidad de ganar \$12.000 (sino, \$0)", 7),
            ("20% de probabilidad de ganar \$30.000 (sino, \$0)", 9),
        ]
    },
    {
        "id": "q3",
        "text": "Un amigo te ofrece invertir en su startup tecnológica. No hay garantías, pero el potencial es 10x en 3 años. ¿Qué porcentaje de tus ahorros arriesgarías?",
        "behavioral": "risk_reward_bias",
        "options": [
            ("0% — no arriesgo ahorros en algo incierto", 0),
            ("Hasta 5% — solo dinero que puedo perder", 3),
            ("Hasta 15% — creo en el potencial y gestiono riesgo", 7),
            ("Hasta 30% o más — las grandes oportunidades requieren grandes apuestas", 10),
        ]
    },
    {
        "id": "q4",
        "text": "¿Qué prefieres?",
        "behavioral": "time_preference",
        "options": [
            ("Recibir $80.000 hoy mismo", 1),
            ("Recibir $100.000 dentro de 6 meses", 4),
            ("Recibir $150.000 dentro de 2 años", 7),
            ("Recibir $300.000 dentro de 5 años", 9),
        ]
    },
    {
        "id": "q5",
        "text": "Lanzas una moneda. Si sale cara ganas $10.000. Si sale sello pierdes $6.000. ¿Juegas?",
        "behavioral": "loss_aversion",
        "options": [
            ("Definitivamente no — el riesgo de pérdida me paraliza", 0),
            ("Probablemente no — prefiero certeza aunque pierda la oportunidad", 3),
            ("Probablemente sí — el valor esperado es positivo", 7),
            ("Definitivamente sí — es matemáticamente favorable", 9),
        ]
    },
    {
        "id": "q6",
        "text": "Tu portafolio sube 40% en 6 meses por un rally inesperado. ¿Qué haces?",
        "behavioral": "volatility_react",
        "options": [
            ("Vendo todo — quiero asegurar las ganancias antes de que el mercado revierta", 1),
            ("Vendo parcialmente para reducir exposición", 4),
            ("No hago nada — mi estrategia es de largo plazo", 7),
            ("Compro más — si subió así, el momentum continuará", 9),
        ]
    },
    {
        "id": "q7",
        "text": "¿En cuánto tiempo necesitas acceso a la mayoría de este capital?",
        "behavioral": "time_preference",
        "options": [
            ("En menos de 1 año", 1),
            ("Entre 1 y 3 años", 3),
            ("Entre 3 y 7 años", 7),
            ("En más de 7 años", 10),
        ]
    },
    {
        "id": "q8",
        "text": "Imagina dos escenarios para el próximo año. ¿Cuál prefieres?",
        "behavioral": "prob_sensitivity",
        "options": [
            ("Ganar exactamente 5% seguro", 1),
            ("80% chance de ganar 8%, 20% chance de ganar 0%", 4),
            ("60% chance de ganar 15%, 40% chance de perder 5%", 7),
            ("40% chance de ganar 35%, 60% chance de perder 10%", 10),
        ]
    },
    {
        "id": "q9",
        "text": "Estás en una racha de mala suerte: 3 inversiones consecutivas terminaron en pérdida. ¿Cómo afecta esto tu próxima decisión?",
        "behavioral": "volatility_react",
        "options": [
            ("Me retiro temporalmente del mercado — necesito claridad emocional", 1),
            ("Reduzco significativamente el monto a invertir", 3),
            ("Mantengo mi proceso sin dejarme llevar por el historial reciente", 7),
            ("Aumento la apuesta — estadísticamente, la suerte debe cambiar", 8),
        ]
    },
    {
        "id": "q10",
        "text": "¿Cómo describirías tu relación con la incertidumbre financiera?",
        "behavioral": "uncertainty_avoid",
        "options": [
            ("La evito a toda costa — prefiero menos retorno pero certeza total", 0),
            ("La tolero en pequeñas dosis si el retorno potencial lo justifica", 4),
            ("Me siento cómodo con ella — es parte inherente de invertir", 7),
            ("La busco activamente — donde hay incertidumbre hay oportunidad", 10),
        ]
    },
    {
        "id": "q11",
        "text": "Un analista te dice que una acción tiene 85% de probabilidad de subir 20% o 15% de caer 60%. ¿Inviertes?",
        "behavioral": "risk_reward_bias",
        "options": [
            ("No — el 15% de probabilidad de pérdida catastrófica es inaceptable", 0),
            ("Quizás una cantidad muy pequeña, como experimento", 4),
            ("Sí, pero con posición limitada — el valor esperado es bueno", 7),
            ("Sí, con posición significativa — las probabilidades están a mi favor", 10),
        ]
    },
    {
        "id": "q12",
        "text": "¿Cómo tomas decisiones financieras importantes generalmente?",
        "behavioral": "uncertainty_avoid",
        "options": [
            ("Con mucha investigación y solo cuando tengo alta confianza", 1),
            ("Después de consultar expertos y comparar opciones", 4),
            ("Combinando análisis propio con intuición bien fundamentada", 7),
            ("Con agilidad — las oportunidades no esperan análisis perfectos", 10),
        ]
    },
]


# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────

def compute_behavioral_features(answers: dict) -> dict:
    """
    Transform raw answers into behavioral dimensions.
    Each dimension is normalized 0–10.
    """
    behavioral_scores = {
        "loss_aversion": [],
        "prob_sensitivity": [],
        "volatility_react": [],
        "time_preference": [],
        "uncertainty_avoid": [],
        "risk_reward_bias": [],
    }

    for q in QUESTIONS:
        qid = q["id"]
        dim = q["behavioral"]
        if qid in answers:
            score = answers[qid]
            behavioral_scores[dim].append(score)

    # Average each dimension and invert where needed
    features = {}
    for dim, vals in behavioral_scores.items():
        if vals:
            avg = np.mean(vals)
            # Invert loss_aversion, prob_sensitivity, volatility_react, uncertainty_avoid
            # so higher value = higher risk tolerance
            if dim in ["loss_aversion", "prob_sensitivity", "volatility_react", "uncertainty_avoid"]:
                features[dim] = avg  # already encoded: higher raw score = less averse
            else:
                features[dim] = avg
        else:
            features[dim] = 5.0  # default neutral

    return features


def compute_risk_score(features: dict) -> float:
    """Compute weighted risk score from behavioral features."""
    weights = {
        "loss_aversion": 0.25,
        "prob_sensitivity": 0.15,
        "volatility_react": 0.20,
        "time_preference": 0.20,
        "uncertainty_avoid": 0.10,
        "risk_reward_bias": 0.10,
    }
    score = sum(features.get(k, 5) * w for k, w in weights.items())
    return np.clip(score, 1, 10)


# ─────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────

def simulate_portfolio(prices_df: pd.DataFrame, weights: dict,
                        initial_investment: float, monthly_contribution: float,
                        years: int) -> pd.DataFrame:
    """
    Simulate portfolio performance with contributions.
    Returns a DataFrame with portfolio value over time.
    """
    available = [t for t in weights if t in prices_df.columns]
    if not available:
        return None

    # Normalize weights
    total_w = sum(weights[t] for t in available)
    norm_w = {t: weights[t] / total_w for t in available}

    prices = prices_df[available].copy()
    prices = prices.iloc[-252 * years:] if len(prices) > 252 * years else prices

    # Daily returns
    returns = prices.pct_change().dropna()

    # Portfolio daily return
    port_returns = pd.Series(0.0, index=returns.index)
    for t, w in norm_w.items():
        port_returns += returns[t] * w

    # Build growth curve with monthly contributions
    portfolio_values = []
    current_value = initial_investment
    monthly_idx = 0

    for i, (date, ret) in enumerate(port_returns.items()):
        current_value *= (1 + ret)
        # Add monthly contribution every ~21 trading days
        monthly_idx += 1
        if monthly_idx >= 21:
            current_value += monthly_contribution
            monthly_idx = 0
        portfolio_values.append((date, current_value))

    df_sim = pd.DataFrame(portfolio_values, columns=["date", "value"])
    df_sim.set_index("date", inplace=True)
    return df_sim


def compute_metrics(sim_df: pd.DataFrame, initial: float) -> dict:
    """Calculate key portfolio performance metrics."""
    if sim_df is None or sim_df.empty:
        return {}

    final = sim_df["value"].iloc[-1]
    years = (sim_df.index[-1] - sim_df.index[0]).days / 365.25

    total_return = (final - initial) / initial
    annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    daily_returns = sim_df["value"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (annualized_return - 0.045) / max(volatility, 0.001)

    rolling_max = sim_df["value"].cummax()
    drawdown = (sim_df["value"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "final_value": final,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# THEME DETECTION & CHART HELPERS
# ─────────────────────────────────────────────

def _is_dark() -> bool:
    """
    Detect whether Streamlit is currently in dark mode.
    config.toml sets base="dark", and users can toggle via the ⋮ menu.
    Streamlit exposes the active theme through st.get_option (>= 1.27)
    or falls back to the config value.
    """
    try:
        theme = st.get_option("theme.base")
        return theme == "dark"
    except Exception:
        return True  # default: dark (matches config.toml)

def _chart_colors():
    """Return a dict of chart-specific colors for the active theme."""
    dark = _is_dark()
    return {
        "grid":       "#1E3A6E" if dark else "#E5E7EB",
        "tick":       "#718096" if dark else "#6B7280",
        "gauge_bg":   "#162032" if dark else "#F3F4F6",
        "gauge_s1":   "#1E3A6E" if dark else "#DBEAFE",
        "gauge_s2":   "#1A3560" if dark else "#BFDBFE",
        "gauge_s3":   "#1A4070" if dark else "#BAE6FD",
        "gauge_s4":   "#2D3A20" if dark else "#FDE68A",
        "gauge_s5":   "#3D2020" if dark else "#FECACA",
        "annotation": "#E2E8F0" if dark else "#1F2937",
        "muted_text": "#718096" if dark else "#6B7280",
    }

CHART_TEMPLATE = dict(
    font_family="DM Sans",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=100, b=20),
)

def chart_gauge(score: float, profile_name: str) -> go.Figure:
    c = _chart_colors()
    color_map = {
        "Muy Conservador": "#1A56DB",
        "Conservador":     "#2563EB",
        "Moderado":        "#0EA5E9",
        "Agresivo":        "#F59E0B",
        "Muy Agresivo":    "#E53E3E",
    }
    color = color_map.get(profile_name, "#1A56DB")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 10", "font": {"size": 36, "family": "Syne", "color": color}},
        gauge={
            "axis": {"range": [1, 10], "tickwidth": 1, "tickcolor": c["tick"],
                     "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": c["gauge_bg"],
            "steps": [
                {"range": [1, 2],  "color": c["gauge_s1"]},
                {"range": [2, 4],  "color": c["gauge_s2"]},
                {"range": [4, 6],  "color": c["gauge_s3"]},
                {"range": [6, 8],  "color": c["gauge_s4"]},
                {"range": [8, 10], "color": c["gauge_s5"]},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score},
        },
        title={"text": f"<b>Risk Score</b><br><span style='font-size:13px;color:{c['muted_text']}'>{profile_name}</span>",
               "font": {"size": 16, "family": "Syne"}},
    ))
    fig.update_layout(height=280, **CHART_TEMPLATE)
    return fig


def chart_pie(weights: dict, profile_name: str) -> go.Figure:
    c = _chart_colors()
    labels = list(weights.keys())
    values = [v * 100 for v in weights.values()]
    colors = ["#1A56DB", "#E53E3E", "#0EA5E9", "#F59E0B", "#10B981", "#8B5CF6"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(colors=colors[:len(labels)], line=dict(color="white", width=2)),
        textfont=dict(size=13, family="DM Sans"),
        textposition="outside",
        texttemplate="%{label}<br><b>%{value:.0f}%</b>",
    ))
    fig.update_layout(
        title=dict(text=f"<b>Composición del Portafolio</b><br><span style='color:{c['muted_text']};font-size:12px'>{profile_name}</span>",
                   font=dict(size=15, family="Syne"), x=0.5),
        height=340,
        showlegend=False,
        **CHART_TEMPLATE
    )
    fig.add_annotation(text=f"<b>{profile_name}</b>", x=0.5, y=0.5,
                       font=dict(size=11, family="Syne", color=c["annotation"]),
                       showarrow=False)
    return fig


def chart_portfolio_vs_spy(sim_portfolio: pd.DataFrame, sim_spy: pd.DataFrame,
                            profile_name: str) -> go.Figure:
    c = _chart_colors()
    fig = go.Figure()

    port_norm = sim_portfolio["value"] / sim_portfolio["value"].iloc[0] * 100
    spy_norm  = sim_spy["value"]  / sim_spy["value"].iloc[0]  * 100

    fig.add_trace(go.Scatter(
        x=port_norm.index, y=port_norm.values,
        name=f"🎯 {profile_name}",
        line=dict(color="#1A56DB", width=2.5),
        fill="tozeroy", fillcolor="rgba(26,86,219,0.06)"
    ))
    fig.add_trace(go.Scatter(
        x=spy_norm.index, y=spy_norm.values,
        name="📊 SPY (S&P 500)",
        line=dict(color="#E53E3E", width=2, dash="dot"),
    ))

    fig.update_layout(
        title=dict(text=f"<b>Portafolio vs. Benchmark (SPY)</b><br><span style='color:{c['muted_text']};font-size:12px'>Rendimiento normalizado (base 100)</span>",
                   font=dict(size=15, family="Syne"), x=0),
        xaxis=dict(showgrid=True, gridcolor=c["grid"], title=""),
        yaxis=dict(showgrid=True, gridcolor=c["grid"], title="Valor (base 100)"),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        height=380,
        **CHART_TEMPLATE
    )
    return fig


def chart_growth_simulation(sim_df: pd.DataFrame, initial: float,
                              monthly: float, profile_name: str) -> go.Figure:
    c = _chart_colors()
    fig = go.Figure()

    sim_monthly = sim_df["value"].resample("ME").last()

    fig.add_trace(go.Scatter(
        x=sim_monthly.index, y=sim_monthly.values,
        name="💼 Portafolio con rendimiento",
        line=dict(color="#1A56DB", width=2.5),
        fill="tonexty", fillcolor="rgba(26,86,219,0.08)"
    ))

    contribution_values = [initial + monthly * i for i in range(len(sim_monthly))]
    fig.add_trace(go.Scatter(
        x=sim_monthly.index, y=contribution_values,
        name="💰 Solo aportes (sin rendimiento)",
        line=dict(color="#9CA3AF", width=1.8, dash="dot"),
    ))

    fig.update_layout(
        title=dict(text=f"<b>Simulación de Crecimiento</b><br><span style='color:{c['muted_text']};font-size:12px'>Capital inicial ${initial:,.0f} + ${monthly:,.0f}/mes</span>",
                   font=dict(size=15, family="Syne"), x=0),
        xaxis=dict(showgrid=True, gridcolor=c["grid"]),
        yaxis=dict(showgrid=True, gridcolor=c["grid"],
                   tickprefix="$", tickformat=",.0f"),
        legend=dict(orientation="h", y=1.12, x=0),
        height=360,
        **CHART_TEMPLATE
    )
    return fig


def chart_feature_importance(model, features) -> go.Figure:
    c = _chart_colors()
    importance = model.feature_importances_
    labels = {
        "loss_aversion":    "Aversión a Pérdidas",
        "prob_sensitivity": "Sens. Probabilística",
        "volatility_react": "Reacción a Volatilidad",
        "time_preference":  "Preferencia Temporal",
        "uncertainty_avoid":"Evitación Incertidumbre",
        "risk_reward_bias": "Sesgo Riesgo/Retorno",
    }
    names = [labels.get(f, f) for f in features]
    sorted_idx = np.argsort(importance)
    bar_colors = ["#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#1A56DB"]

    fig = go.Figure(go.Bar(
        y=[names[i] for i in sorted_idx],
        x=[importance[i] for i in sorted_idx],
        orientation="h",
        marker=dict(color=[bar_colors[i % len(bar_colors)] for i in range(len(sorted_idx))],
                    line=dict(color="white", width=0.5)),
        text=[f"{importance[i]:.1%}" for i in sorted_idx],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"<b>Importancia de Variables</b><br><span style='color:{c['muted_text']};font-size:12px'>Contribución al modelo ML</span>",
                   font=dict(size=14, family="Syne"), x=0),
        xaxis=dict(tickformat=".0%", showgrid=True, gridcolor=c["grid"]),
        yaxis=dict(autorange="reversed"),
        height=280,
        **CHART_TEMPLATE
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
            <span style='font-family:Syne;font-size:1.5rem;font-weight:800;color:#1A56DB;'>My</span><span style='font-family:Syne;font-size:1.5rem;font-weight:800;color:var(--text-heading);'>Smart</span><span style='font-family:Syne;font-size:1.5rem;font-weight:800;color:#E53E3E;'>Advisor</span>
            <div style='font-size:0.7rem;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;margin-top:2px;'>Behavioral Finance AI</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='sidebar-section'>
            <div class='sidebar-title'>🧠 Behavioral Economics</div>
            <div class='sidebar-body'>
            Este sistema aplica <strong>Prospect Theory</strong> (Kahneman & Tversky) para modelar cómo las personas evalúan pérdidas y ganancias de forma asimétrica. La aversión a pérdidas, sesgos de disponibilidad y preferencias de tiempo son variables clave del análisis.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='sidebar-section'>
            <div class='sidebar-title'>🤖 Machine Learning</div>
            <div class='sidebar-body'>
            Usamos un <strong>Decision Tree Classifier</strong> entrenado con 600+ registros sintéticos de perfiles conductuales. Las respuestas del cuestionario se transforman en 6 dimensiones psicológicas que el modelo usa para clasificar el perfil de riesgo.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='sidebar-section'>
            <div class='sidebar-title'>📊 Tecnologías</div>
            <div class='sidebar-body'>
            🐍 Python · Streamlit<br>
            📈 yfinance · Plotly<br>
            🤖 scikit-learn<br>
            🔢 pandas · numpy
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='sidebar-disclaimer'>
            ⚠️ <strong>Disclaimer</strong><br>
            Esta plataforma tiene fines exclusivamente educativos y no constituye asesoría financiera real. Las simulaciones son históricas y no garantizan rendimientos futuros.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Progress
        if "step" in st.session_state:
            step = st.session_state["step"]
            total_steps = 3
            progress_pct = int((step / total_steps) * 100)
            st.markdown(f"""
            <div style='font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;'>Progreso</div>
            <div class='progress-container'>
                <div class='progress-fill' style='width:{progress_pct}%;'></div>
            </div>
            <div style='font-size:0.78rem;color:#1A56DB;font-weight:600;'>{progress_pct}% completado</div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # Initialize session state
    if "step" not in st.session_state:
        st.session_state["step"] = 0
    if "answers" not in st.session_state:
        st.session_state["answers"] = {}
  

    render_sidebar()
    model, accuracy, features, df = train_model()

    step = st.session_state["step"]

    # ─── STEP 0: Landing ───
    if step == 0:
        st.markdown("""
        <div class='hero-banner animate-in'>
            <div class='hero-badge'>✦ Behavioral Finance AI</div>
            <div class='hero-title'>MySmartAdvisor</div>
            <div class='hero-subtitle'>Descubre tu perfil psicológico-financiero y obtén una cartera de inversión personalizada basada en datos reales de mercado.</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='card card-blue'>
                <div style='font-size:2rem;margin-bottom:0.5rem;'>🧠</div>
                <div class='card-title'>Perfilamiento Conductual</div>
                <div style='font-size:0.85rem;color:var(--text-body);'>Analizamos tu psicología financiera usando principios de Prospect Theory y Behavioral Economics.</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='card card-blue'>
                <div style='font-size:2rem;margin-bottom:0.5rem;'>🤖</div>
                <div class='card-title'>IA & Machine Learning</div>
                <div style='font-size:0.85rem;color:var(--text-body);'>Un Decision Tree Classifier clasifica tu perfil entre 5 niveles de riesgo con alta precisión.</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='card card-blue'>
                <div style='font-size:2rem;margin-bottom:0.5rem;'>📊</div>
                <div class='card-title'>Datos Reales de Mercado</div>
                <div style='font-size:0.85rem;color:var(--text-body);'>Portafolios con ETFs reales, simulaciones históricas y benchmark contra el S&P 500.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([2, 1.5, 2])
        with col_b:
            if st.button("🚀  Iniciar mi análisis", use_container_width=True):
                st.session_state["step"] = 1
                st.rerun()

        st.markdown("""
        <div style='text-align:center;margin-top:2rem;'>
            <span style='font-size:0.8rem;color:var(--text-muted);'>⏱ &nbsp;12 preguntas · ~4 minutos · Resultados inmediatos</span>
        </div>
        """, unsafe_allow_html=True)

        # ML Info section
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;margin-bottom:1.5rem;'>
            <span style='font-family:Syne;font-size:1.2rem;font-weight:700;color:var(--text-heading-dark);'>¿Cómo funciona el sistema?</span>
        </div>
        """, unsafe_allow_html=True)

        process_steps = [
            ("01", "Cuestionario Conductual", "Responde 12 preguntas basadas en dilemas financieros reales"),
            ("02", "Análisis Psicológico", "El sistema mide 6 dimensiones conductuales clave"),
            ("03", "Clasificación ML", "Un árbol de decisión determina tu perfil de riesgo"),
            ("04", "Portafolio Personalizado", "Recibe una cartera con ETFs reales y datos históricos"),
        ]
        steps_html = "<div style='display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;'>"
        for num, title, desc in process_steps:
            steps_html += f"<div style='flex:1;min-width:180px;background:var(--bg-card);border-radius:14px;padding:1.2rem;border:1px solid var(--border-divider);text-align:center;'>"
            steps_html += f"<div style='font-family:Syne;font-size:2rem;font-weight:800;color:var(--blue-primary);opacity:0.25;'>{num}</div>"
            steps_html += f"<div style='font-family:Syne;font-size:0.9rem;font-weight:700;color:var(--text-heading);margin-bottom:0.3rem;'>{title}</div>"
            steps_html += f"<div style='font-size:0.78rem;color:var(--text-muted);'>{desc}</div>"
            steps_html += "</div>"
        steps_html += "</div>"
        st.markdown(steps_html, unsafe_allow_html=True)

        # Model accuracy info
        st.markdown("<br>", unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("🎯 Precisión del Modelo ML", f"{accuracy:.1%}")
        with col_m2:
            st.metric("📋 Registros de Entrenamiento", "600+")
        with col_m3:
            st.metric("🔢 Dimensiones Conductuales", "6")

    # ─── STEP 1: Questionnaire ───
    elif step == 1:
        st.markdown("""
        <div style='margin-bottom:2rem;'>
            <div style='font-family:Syne;font-size:1.8rem;font-weight:800;color:var(--text-heading-dark);'>Análisis Conductual</div>
            <div style='color:var(--text-muted);font-size:0.95rem;margin-top:0.3rem;'>Responde con honestidad. No hay respuestas correctas o incorrectas — cada elección revela algo sobre tu psicología financiera.</div>
        </div>
        """, unsafe_allow_html=True)

        answers = {}
        all_answered = True

        for i, q in enumerate(QUESTIONS):
            options_text = [opt[0] for opt in q["options"]]

            st.markdown(f"""
            <div class='question-card'>
                <div class='question-number'>Pregunta {i+1} de {len(QUESTIONS)}</div>
                <div class='question-text'>{q['text']}</div>
            </div>
            """, unsafe_allow_html=True)

            choice = st.radio(
                label=f"q_{i}",
                options=options_text,
                index=None,
                key=f"radio_{q['id']}",
                label_visibility="collapsed"
            )

            if choice is not None:
                val = next((score for text, score in q["options"] if text == choice), None)
                if val is not None:
                    answers[q["id"]] = val
            else:
                all_answered = False

            st.markdown("<br>", unsafe_allow_html=True)

        # Progress indicator
        answered_count = sum(1 for q in QUESTIONS if f"radio_{q['id']}" in st.session_state
                             and st.session_state[f"radio_{q['id']}"] is not None)
        st.markdown(f"""
        <div style='margin-bottom:1.5rem;'>
            <div style='font-size:0.8rem;color:#718096;margin-bottom:6px;'>{answered_count} / {len(QUESTIONS)} preguntas respondidas</div>
            <div class='progress-container'>
                <div class='progress-fill' style='width:{int(answered_count/len(QUESTIONS)*100)}%;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_back, col_next = st.columns([1, 3])
        with col_back:
            if st.button("← Volver", use_container_width=True):
                st.session_state["step"] = 0
                st.rerun()
        with col_next:
            if not all_answered:
                st.info("💡 Responde todas las preguntas para continuar.")
            else:
                if st.button("Analizar mi perfil  →", use_container_width=True):
                    st.session_state["answers"] = answers
                    st.session_state["step"] = 2
                    st.rerun()

    # ─── STEP 2: Results & Dashboard ───
    elif step == 2:
        answers = st.session_state.get("answers", {})

        if not answers:
            st.warning("No se encontraron respuestas. Por favor vuelve al cuestionario.")
            if st.button("← Reiniciar"):
                st.session_state["step"] = 0
                st.rerun()
            return

        # Compute behavioral features
        features_dict = compute_behavioral_features(answers)
        risk_score = compute_risk_score(features_dict)

        # ML prediction
        feature_order = ['loss_aversion', 'prob_sensitivity', 'volatility_react',
                         'time_preference', 'uncertainty_avoid', 'risk_reward_bias']
        X_user = np.array([[features_dict[f] for f in feature_order]])
        profile_idx = model.predict(X_user)[0]
        profile_data = PORTFOLIOS[profile_idx]

        # ── HERO RESULT BANNER ──
        profile_colors = {
            "Muy Conservador": "#1A56DB",
            "Conservador": "#2563EB",
            "Moderado": "#0EA5E9",
            "Agresivo": "#F59E0B",
            "Muy Agresivo": "#E53E3E",
        }
        pcolor = profile_colors.get(profile_data["name"], "#1A56DB")

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{pcolor}15,{pcolor}08);
                    border:1.5px solid {pcolor}30;border-radius:20px;padding:2rem 2.5rem;
                    margin-bottom:2rem;'>
            <div style='font-size:0.7rem;color:{pcolor};font-weight:700;letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:0.5rem;'>✦ Tu perfil detectado</div>
            <div style='font-family:Syne;font-size:2.5rem;font-weight:800;color: var(--text-heading);'>
                {profile_data['emoji']} {profile_data['name']}
            </div>
            <div style='font-size:0.95rem;color:#718096;margin-top:0.3rem;'>
                Risk Score: <strong style='color:{pcolor};font-family:Syne;font-size:1.1rem;'>{risk_score:.1f} / 10</strong>
                &nbsp;·&nbsp; Horizonte recomendado: <strong>{profile_data['horizon']}</strong>
                &nbsp;·&nbsp; Retorno esperado: <strong>{profile_data['expected_return']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── GAUGE + PIE ──
        col_gauge, col_pie = st.columns([1, 1])
        with col_gauge:
            st.plotly_chart(chart_gauge(risk_score, profile_data["name"]),
                            use_container_width=True, config={"displayModeBar": False})
        with col_pie:
            st.plotly_chart(chart_pie(profile_data["assets"], profile_data["name"]),
                            use_container_width=True, config={"displayModeBar": False})

        # ── BEHAVIORAL ANALYSIS ──
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:Syne;font-size:1.3rem;font-weight:700;color:var(--text-heading);margin-bottom:1rem;'>
            🧠 Análisis Psicológico-Conductual
        </div>
        """, unsafe_allow_html=True)

        col_psy, col_beh = st.columns(2)
        with col_psy:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>🔬 Perfil Psicológico</div>
                <div style='font-size:0.87rem;color:#A0AEC0;line-height:1.6;margin-top:0.5rem;'>
                    {profile_data['psychology']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_beh:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>📐 Sesgos Conductuales</div>
                <div style='font-size:0.87rem;color:#A0AEC0;line-height:1.6;margin-top:0.5rem;'>
                    {profile_data['behavioral']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── BEHAVIORAL DIMENSIONS RADAR ──
        dim_labels = {
            "loss_aversion": "Tolerancia<br>a Pérdidas",
            "prob_sensitivity": "Razonamiento<br>Probabilístico",
            "volatility_react": "Estabilidad<br>Emocional",
            "time_preference": "Paciencia<br>Financiera",
            "uncertainty_avoid": "Tolerancia<br>Incertidumbre",
            "risk_reward_bias": "Búsqueda<br>de Retorno",
        }

        radar_vals = [features_dict[f] for f in feature_order]
        radar_labels = [dim_labels[f] for f in feature_order]

        _c = _chart_colors()
        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill='toself',
            fillcolor=f"rgba(26,86,219,0.15)",
            line=dict(color="#1A56DB", width=2),
            name="Tu perfil"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=9),
                                gridcolor=_c["grid"], linecolor=_c["grid"]),
                angularaxis=dict(tickfont=dict(size=11, family="DM Sans"),
                                 gridcolor=_c["grid"], linecolor=_c["grid"]),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            title=dict(text="<b>Dimensiones Conductuales</b>",
                       font=dict(size=14, family="Syne"), x=0.5),
            height=360,
            **CHART_TEMPLATE
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        # ── PORTFOLIO COMPOSITION TABLE ──
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:Syne;font-size:1.3rem;font-weight:700;color:var(--text-heading);margin-bottom:1rem;'>
            💼 Portafolio Recomendado
        </div>
        """, unsafe_allow_html=True)

        etf_info = {
            "SPY": "SPDR S&P 500 ETF — Acciones de gran capitalización EE.UU.",
            "QQQ": "Invesco Nasdaq-100 ETF — Tecnología y crecimiento",
            "BND": "Vanguard Total Bond Market ETF — Renta fija diversificada",
            "GLD": "SPDR Gold Shares — Oro como cobertura",
            "IWM": "iShares Russell 2000 ETF — Small caps EE.UU.",
            "VTI": "Vanguard Total Stock Market ETF — Mercado total EE.UU.",
        }

        portfolio_rows = []
        for ticker, weight in profile_data["assets"].items():
            portfolio_rows.append({
                "Ticker": f"{ticker}",
                "Descripción": etf_info.get(ticker, ticker),
                "Asignación": f"{weight*100:.0f}%",
            })
        df_port = pd.DataFrame(portfolio_rows)
        st.dataframe(df_port, use_container_width=True, hide_index=True)

        # ── SIMULATION PARAMETERS ──
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:Syne;font-size:1.3rem;font-weight:700;color:var(--text-heading);margin-bottom:1rem;'>
            📈 Simulación Financiera
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ Configurar parámetros de simulación", expanded=True):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                initial_investment = st.number_input(
                    "Inversión inicial (USD)", min_value=1000, max_value=1_000_000,
                    value=10000, step=1000, format="%d"
                )
            with col_s2:
                monthly_contrib = st.number_input(
                    "Aporte mensual (USD)", min_value=0, max_value=50000,
                    value=500, step=100, format="%d"
                )
            with col_s3:
                sim_years = st.selectbox("Horizonte temporal", [1, 2, 3, 5, 7, 10], index=3)

        # ── LOAD MARKET DATA ──
        all_tickers = list(profile_data["assets"].keys()) + ["SPY"]
        all_tickers = list(set(all_tickers))

        with st.spinner("⬇️ Descargando datos de mercado..."):
            prices = download_market_data(all_tickers, period="5y")

        if prices is not None and not prices.empty:
            # Simulate portfolio
            sim_port = simulate_portfolio(
                prices, profile_data["assets"],
                initial_investment, monthly_contrib, sim_years
            )
            # Simulate SPY benchmark
            spy_prices = prices[["SPY"]] if "SPY" in prices.columns else None
            sim_spy = None
            if spy_prices is not None:
                sim_spy = simulate_portfolio(
                    spy_prices, {"SPY": 1.0},
                    initial_investment, monthly_contrib, sim_years
                )

            if sim_port is not None:
                metrics = compute_metrics(sim_port, initial_investment)
                spy_metrics = compute_metrics(sim_spy, initial_investment) if sim_spy is not None else {}

                # ── METRICS ROW ──
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric(
                        "💰 Valor Final Estimado",
                        f"${metrics.get('final_value', 0):,.0f}",
                        delta=f"+${metrics.get('final_value',0)-initial_investment:,.0f}"
                    )
                with col_m2:
                    ar = metrics.get('annualized_return', 0)
                    spy_ar = spy_metrics.get('annualized_return', 0)
                    st.metric(
                        "📈 Retorno Anualizado",
                        f"{ar:.1%}",
                        delta=f"vs SPY {spy_ar:.1%}"
                    )
                with col_m3:
                    st.metric(
                        "📊 Volatilidad Anual",
                        f"{metrics.get('volatility', 0):.1%}"
                    )
                with col_m4:
                    st.metric(
                        "⚡ Sharpe Ratio",
                        f"{metrics.get('sharpe', 0):.2f}"
                    )

                col_m5, col_m6 = st.columns(2)
                with col_m5:
                    st.metric(
                        "📉 Drawdown Máximo",
                        f"{metrics.get('max_drawdown', 0):.1%}"
                    )
                with col_m6:
                    st.metric(
                        "📊 Retorno Total",
                        f"{metrics.get('total_return', 0):.1%}"
                    )

                # ── CHARTS ──
                if sim_spy is not None:
                    st.plotly_chart(
                        chart_portfolio_vs_spy(sim_port, sim_spy, profile_data["name"]),
                        use_container_width=True, config={"displayModeBar": False}
                    )

                st.plotly_chart(
                    chart_growth_simulation(sim_port, initial_investment,
                                            monthly_contrib, profile_data["name"]),
                    use_container_width=True, config={"displayModeBar": False}
                )

        else:
            st.warning("No se pudieron cargar datos históricos. Verifica tu conexión a internet.")

        # ── ML MODEL INFO ──
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:Syne;font-size:1.3rem;font-weight:700;color:var(--text-heading);margin-bottom:1rem;'>
            🤖 Modelo de Machine Learning
        </div>
        """, unsafe_allow_html=True)

        col_ml1, col_ml2, col_ml3 = st.columns(3)
        with col_ml1:
            st.metric("🎯 Accuracy del Modelo", f"{accuracy:.1%}")
        with col_ml2:
            st.metric("🌳 Profundidad del Árbol", f"{model.get_depth()}")
        with col_ml3:
            st.metric("📋 Hojas del Árbol", f"{model.get_n_leaves()}")

        st.plotly_chart(
            chart_feature_importance(model, feature_order),
            use_container_width=True, config={"displayModeBar": False}
        )

        # ── RESET BUTTON ──
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        col_r1, col_r2, col_r3 = st.columns([1.5, 1, 1.5])
        with col_r2:
            if st.button("🔄  Reiniciar análisis", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.markdown("""
        <div style='text-align:center;margin-top:1rem;font-size:0.75rem;color:#718096;'>
            MySmartAdvisor · Behavioral Finance AI · Fines educativos únicamente
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()