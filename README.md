# 💎 MySmartAdvisor — Behavioral Finance Robo-Advisor

> **Un MVP fintech que perfi la psicológicamente tu riesgo financiero y recomienda portafolios de inversión basados en datos reales de mercado.**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 🎯 Descripción

**MySmartAdvisor** es un roboadvisor de nueva generación que combina **Behavioral Economics**, **Machine Learning** y **datos reales de mercado** para ofrecerte una recomendación de inversión personalizada.

A diferencia de los cuestionarios financieros tradicionales, este sistema analiza tu **psicología financiera real** a través de dilemas conductuales, escenarios probabilísticos y decisiones bajo incertidumbre. El resultado es un perfil de riesgo más preciso y honesto que cualquier pregunta directa de tolerancia al riesgo.

---

## ✨ Características principales

- **🧠 Perfilamiento conductual profundo** — 12 preguntas basadas en Prospect Theory y Behavioral Finance
- **🤖 Machine Learning** — Decision Tree Classifier entrenado con 600+ registros sintéticos
- **📊 Risk Score 1–10** — Gauge visual interactivo con clasificación en 5 niveles
- **💼 Portafolios automáticos** — ETFs reales (SPY, QQQ, BND, GLD, IWM)
- **📈 Datos reales de mercado** — Descarga histórica via yfinance (últimos 5 años)
- **📉 Simulación financiera** — Con inversión inicial, aportes mensuales y horizonte configurable
- **🏆 Benchmark vs SPY** — Comparación automática contra el S&P 500
- **🎨 UI premium fintech** — Diseño moderno tipo startup SaaS

---

## 🚀 Deployment en Streamlit Community Cloud

### Paso 1: Fork el repositorio
```bash
git clone https://github.com/tu-usuario/mysmartadvisor.git
cd mysmartadvisor
```

### Paso 2: Instalar dependencias localmente (opcional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Paso 3: Deploy en Streamlit Cloud
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu repositorio GitHub
3. Selecciona `app.py` como archivo principal
4. ¡Haz clic en Deploy!

> **Nota:** No se requieren variables de entorno ni configuración adicional. La aplicación funciona completamente out-of-the-box.

---

## 📁 Estructura del proyecto

```
MySmartAdvisor/
│
├── app.py              # Aplicación principal (monolítica para simplicidad)
├── requirements.txt    # Dependencias Python
└── README.md           # Este archivo
```

---

## 🧠 Behavioral Economics — Marco Conceptual

### Prospect Theory (Kahneman & Tversky, 1979)
La teoría prospectos describe cómo las personas evalúan pérdidas y ganancias de manera **asimétrica**. El dolor de perder $100 es psicológicamente más intenso que el placer de ganar $100. Este sistema mide esta asimetría en cada usuario.

### Dimensiones conductuales analizadas

| Dimensión | Descripción | Sesgo asociado |
|-----------|-------------|----------------|
| **Aversión a Pérdidas** | Qué tan dolorosas son las pérdidas vs. las ganancias | Loss aversion |
| **Sensibilidad Probabilística** | Cómo se procesan las probabilidades | Overweighting small probs |
| **Reacción a Volatilidad** | Respuesta emocional ante movimientos bruscos | Myopic loss aversion |
| **Preferencia Temporal** | Capacidad para diferir gratificación | Present bias |
| **Evitación de Incertidumbre** | Aversión a lo desconocido vs. riesgo calculable | Ambiguity aversion |
| **Sesgo Riesgo/Retorno** | Propensión a buscar altos retornos | Overconfidence |

---

## 🤖 Machine Learning

### Modelo: Decision Tree Classifier
- **Dataset**: 600 registros sintéticos generados con coherencia psicológica-financiera
- **Features**: 6 dimensiones conductuales (valores 0–10)
- **Target**: 5 perfiles de riesgo (Muy Conservador → Muy Agresivo)
- **Split**: 80% training / 20% testing
- **Accuracy**: ~85–90% (varía por semilla)
- **Hiperparámetros**: `max_depth=6`, `min_samples_leaf=10`

### ¿Por qué Decision Tree?
- **Interpretabilidad**: Las reglas de decisión son transparentes y explicables
- **Velocidad**: Clasificación instantánea sin latencia
- **No requiere normalización**: Adecuado para variables ordinales
- **Evita overfitting**: Con `max_depth` y `min_samples_leaf` controlados

---

## 📊 Perfiles de Riesgo

| Score | Perfil | Assets |
|-------|--------|--------|
| 1–2 | 🛡️ Muy Conservador | 60% BND · 20% GLD · 20% SPY |
| 3–4 | ⚓ Conservador | 50% BND · 35% SPY · 15% GLD |
| 5–6 | ⚖️ Moderado | 50% SPY · 30% QQQ · 20% BND |
| 7–8 | 🚀 Agresivo | 60% QQQ · 25% SPY · 15% IWM |
| 9–10 | ⚡ Muy Agresivo | 70% QQQ · 20% IWM · 10% SPY |

---

## 🛠️ Tecnologías

| Tecnología | Uso |
|------------|-----|
| **Python 3.10+** | Lenguaje base |
| **Streamlit** | Framework UI web |
| **pandas & numpy** | Procesamiento de datos |
| **scikit-learn** | Machine Learning |
| **yfinance** | Datos históricos de mercado |
| **Plotly** | Visualizaciones interactivas |
| **Google Fonts** | Tipografía (Syne + DM Sans) |

---

## 📸 Screenshots sugeridos

Al documentar el proyecto, incluir capturas de:

1. **Landing page** — Hero banner con las 3 propuestas de valor
2. **Cuestionario** — Una pregunta conductual con opciones de radio
3. **Dashboard de resultados** — Gauge + pie chart del portafolio
4. **Análisis conductual** — Radar de dimensiones psicológicas
5. **Benchmark chart** — Portafolio vs SPY en gráfico temporal
6. **Simulación** — Curva de crecimiento con aportes mensuales

---

## ⚠️ Disclaimer

> Esta plataforma tiene fines **exclusivamente educativos** y **no constituye asesoría financiera real**. Las simulaciones son históricas y no garantizan rendimientos futuros. Siempre consulta a un asesor financiero certificado antes de tomar decisiones de inversión.

---

## 👨‍💻 Autor

Desarrollado como prototipo académico de **Behavioral Finance + AI** para demostrar la aplicación de economía conductual en fintech moderno.

---

*MySmartAdvisor · Behavioral Finance AI · 2024*
