from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PREPROCESSOR_PATH = BASE_DIR / "preprocessors" / "preprocessor.pkl"
MODELS_DIR = BASE_DIR / "models"

# ---------- Učitavanje preprocessora ----------
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ---------- Učitavanje modela ----------
model_f2 = joblib.load(MODELS_DIR / "best_f2_model.pkl")
model_recall = joblib.load(MODELS_DIR / "best_recall_model.pkl")

st.set_page_config(page_title="Bank Marketing Prediction", page_icon="🏦")
st.title("🏦 Bank Marketing Prediction")
st.write("Predikcija pretplate klijenta na oročeni depozit (y = yes/no).")

# ---------- Strategija ----------
strategy = st.selectbox(
    "Strategija predikcije",
    options=["max_f2", "max_recall"],
    help=(
        "**max_f2**: Balansira Recall i Precision (favorizuje Recall 2x). "
        "Banka želi da uhvati što više potencijalnih klijenata uz prihvatljiv broj lažnih pozitiva.\n\n"
        "**max_recall**: Maksimizira Recall bez obzira na Precision. "
        "Banka želi da ne propusti nijednog klijenta koji bi se pretplatio."
    ),
)

# ---------- Kategorijski inputi ----------
st.subheader("Lični podaci")

job = st.selectbox(
    "Zanimanje",
    options=[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ],
)

marital = st.selectbox(
    "Bračni status",
    options=["divorced", "married", "single", "unknown"],
)

education = st.selectbox(
    "Nivo obrazovanja",
    options=[
        "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
        "professional.course", "university.degree", "unknown",
    ],
)

st.subheader("Finansijski podaci")

default = st.selectbox(
    "Kreditno zaduženje (default)",
    options=["no", "unknown", "yes"],
)

housing = st.selectbox(
    "Stambeni kredit",
    options=["no", "unknown", "yes"],
)

loan = st.selectbox(
    "Lični kredit",
    options=["no", "unknown", "yes"],
)

st.subheader("Kontakt kampanja")

contact = st.selectbox(
    "Tip kontakta",
    options=["cellular", "telephone"],
)

month = st.selectbox(
    "Mesec poslednjeg kontakta",
    options=[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ],
)

poutcome = st.selectbox(
    "Ishod prethodne kampanje",
    options=["failure", "nonexistent", "success"],
)

# ---------- Numerički inputi ----------
st.subheader("Numerički podaci")

age = st.number_input(
    "Starost",
    min_value=18,
    max_value=100,
    value=40,
    step=1,
)

campaign = st.number_input(
    "Broj kontakata u ovoj kampanji",
    min_value=1,
    max_value=50,
    value=1,
    step=1,
)

pdays = st.number_input(
    "Broj dana od prethodnog kontakta (-1 = nije prethodno kontaktiran)",
    min_value=-1,
    max_value=999,
    value=-1,
    step=1,
)

previous = st.number_input(
    "Broj prethodnih kontakata pre ove kampanje",
    min_value=0,
    max_value=50,
    value=0,
    step=1,
)

cons_price_idx = st.number_input(
    "Indeks potrošačkih cena (cons.price.idx)",
    min_value=90.0,
    max_value=100.0,
    value=93.994,
    step=0.001,
    format="%.3f",
)

cons_conf_idx = st.number_input(
    "Indeks poverenja potrošača (cons.conf.idx)",
    min_value=-60.0,
    max_value=-20.0,
    value=-40.5,
    step=0.1,
    format="%.1f",
)

nr_employed = st.number_input(
    "Broj zaposlenih - kvartalno (nr.employed)",
    min_value=4500.0,
    max_value=5500.0,
    value=5099.1,
    step=0.1,
    format="%.1f",
)

# ---------- Predikcija ----------
if st.button("Predict"):
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "month": month,
                "poutcome": poutcome,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "cons.price.idx": cons_price_idx,
                "cons.conf.idx": cons_conf_idx,
                "nr.employed": nr_employed,
            }
        ]
    )

    input_encoded = preprocessor.transform(input_data)

    # Odabir modela po strategiji
    if strategy == "max_f2":
        model = model_f2
        strategy_desc = (
            "F2 strategija — balansira Recall i Precision, favorizuje Recall (2x)."
        )
    else:
        model = model_recall
        strategy_desc = (
            "Recall strategija — maksimizira Recall bez obzira na Precision."
        )

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0, 1]

    st.subheader("📊 Rezultati")
    st.write(f"**Strategija:** {strategy} — {strategy_desc}")
    st.write(f"**Model:** {type(model.named_steps[list(model.named_steps.keys())[-1]]).__name__ if hasattr(model, 'named_steps') else type(model).__name__}")
    st.write(f"**Predikcija:** {'✅ YES (pretplatiće se)' if prediction == 1 else '❌ NO (neće se pretplatiti)'}")
    st.write(f"**Verovatnoća (YES):** {probability:.4f} ({probability * 100:.2f}%)")
    st.write(f"**Verovatnoća (NO):** {1 - probability:.4f} ({(1 - probability) * 100:.2f}%)")

    # Progress bar za vizualizaciju
    st.progress(float(probability))
    st.caption(f"Verovatnoća pretplate: {probability * 100:.1f}%")