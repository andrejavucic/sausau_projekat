from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

# ---------- FastAPI app ----------
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API za predviđanje pretplate klijenta na oročeni depozit (y = yes/no). "
                "Koristi 2 najbolja modela sa dve strategije: max_f2 i max_recall.",
    version="2.0.0",
)

# ---------- Putanje do fajlova ----------
BASE_DIR = Path(__file__).resolve().parent
PREPROCESSOR_PATH = BASE_DIR / "preprocessors" / "preprocessor.pkl"
MODELS_DIR = BASE_DIR / "models"

FEATURE_NAMES_PATH = BASE_DIR / "data" / "processed" / "feature_names.npy"

# ---------- Učitavanje preprocessora ----------
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"[OK] Učitan preprocessor iz: {PREPROCESSOR_PATH}")
except FileNotFoundError:
    preprocessor = None
    print("[UPOZORENJE] Preprocessor nije pronađen! Pokreni prvo preprocessing.py")

# ---------- Strategije ----------
Strategy = Literal["max_f2", "max_recall"]

STRATEGY_DESCRIPTIONS = {
    "max_f2": (
        "F2 strategija — balansira Recall i Precision, ali favorizuje Recall (2x). "
        "Banka želi da uhvati što više potencijalnih klijenata uz prihvatljiv broj lažnih pozitiva."
    ),
    "max_recall": (
        "Recall strategija — maksimizira Recall bez obzira na Precision. "
        "Banka želi da ne propusti nijednog klijenta koji bi se pretplatio (minimizira FN)."
    ),
}

# ---------- Učitavanje DVA najbolja modela ----------
MODEL_FILES = {
    "best_f2_model": "Najbolji F2 Model",
    "best_recall_model": "Najbolji Recall Model",
}

models = {}

for fname, display_name in MODEL_FILES.items():
    path = MODELS_DIR / f"{fname}.pkl"
    if path.exists():
        model = joblib.load(path)
        # Ako je pipeline, izvuci samo naziv poslednjeg koraka
        if hasattr(model, "named_steps"):
            estimator_name = type(
                list(model.named_steps.values())[-1]
            ).__name__
        else:
            estimator_name = type(model).__name__
        models[fname] = {
            "model": model,
            "display_name": display_name,
            "estimator": estimator_name,
        }
        print(f"[OK] Učitan model: {display_name} ({estimator_name})")
    else:
        models[fname] = None
        print(f"[UPOZORENJE] Model nije pronađen: {path}")

if not any(models.values()):
    raise RuntimeError(
        "Nijedan model nije pronađen u models/ direktorijumu! "
        "Pokreni prvo train.py da istreniraš modele."
    )

# ---------- Učitavanje metrika ----------
METRICS_PATH = BASE_DIR / "analysis" / "metrics_table.txt"
metrics_text = ""
if METRICS_PATH.exists():
    metrics_text = METRICS_PATH.read_text(encoding="utf-8")
    print(f"[OK] Učitane metrike iz: {METRICS_PATH}")

# ---------- Pydantic model za ulaz ----------
class ClientInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Starost klijenta (18-100)", examples=[42])
    job: str = Field(..., description="Zanimanje", examples=["admin."])
    marital: str = Field(..., description="Bračni status", examples=["married"])
    education: str = Field(..., description="Nivo obrazovanja", examples=["university.degree"])
    default: str = Field(..., description="Da li ima kreditno zaduženje", examples=["no"])
    housing: str = Field(..., description="Da li ima stambeni kredit", examples=["yes"])
    loan: str = Field(..., description="Da li ima lični kredit", examples=["no"])
    contact: str = Field(..., description="Tip kontakta (cellular/telephone)", examples=["cellular"])
    month: str = Field(..., description="Mesec poslednjeg kontakta", examples=["may"])
    poutcome: str = Field(
        ..., description="Ishod prethodne kampanje", examples=["nonexistent"]
    )
    campaign: int = Field(..., ge=1, description="Broj kontakata u ovoj kampanji", examples=[1])
    pdays: int = Field(
        ...,
        description="Broj dana od prethodnog kontakta (-1 = nije prethodno kontaktiran)",
        examples=[-1],
    )
    previous: int = Field(
        ..., ge=0, description="Broj prethodnih kontakata pre ove kampanje", examples=[0]
    )
    cons_price_idx: float = Field(
        ..., alias="cons.price.idx", description="Indeks potrošačkih cena", examples=[93.994]
    )
    cons_conf_idx: float = Field(
        ..., alias="cons.conf.idx", description="Indeks poverenja potrošača", examples=[-40.5]
    )
    nr_employed: float = Field(
        ..., alias="nr.employed", description="Broj zaposlenih (kvartalno)", examples=[5099.1]
    )

    class Config:
        populate_by_name = True


class SingleClientInput(ClientInput):
    strategy: Strategy = Field(
        ...,
        description="Strategija predikcije: 'max_f2' (balans) ili 'max_recall' (ne propustiti pozitivne)",
    )


class BatchClientInput(BaseModel):
    clients: List[ClientInput] = Field(..., description="Lista klijenata za batch predikciju")
    strategy: Strategy = Field(
        ...,
        description="Strategija predikcije: 'max_f2' (balans) ili 'max_recall' (ne propustiti pozitivne)",
    )


# ---------- Pomoćna funkcija za predikciju ----------
def predict_client(client: ClientInput, strategy: Strategy) -> dict:
    """Propušta jednog klijenta kroz preprocessor i oba modela, vraća rezultat za izabranu strategiju."""
    if preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Preprocessor nije dostupan. Pokreni preprocessing.py prvo.",
        )

    # Mapiranje strategije na konkretan model
    strategy_model_map = {
        "max_f2": "best_f2_model",
        "max_recall": "best_recall_model",
    }
    chosen_key = strategy_model_map[strategy]
    alternative_key = "best_recall_model" if strategy == "max_f2" else "best_f2_model"

    if models.get(chosen_key) is None or models.get(alternative_key) is None:
        raise HTTPException(
            status_code=500,
            detail="Modeli nisu dostupni. Pokreni train.py prvo.",
        )

    # Kreiraj DataFrame sa istim redosledom kolona kao u treningu
    input_dict = {
        "age": client.age,
        "job": client.job,
        "marital": client.marital,
        "education": client.education,
        "default": client.default,
        "housing": client.housing,
        "loan": client.loan,
        "contact": client.contact,
        "month": client.month,
        "poutcome": client.poutcome,
        "campaign": client.campaign,
        "pdays": client.pdays,
        "previous": client.previous,
        "cons.price.idx": client.cons_price_idx,
        "cons.conf.idx": client.cons_conf_idx,
        "nr.employed": client.nr_employed,
    }

    input_df = pd.DataFrame([input_dict])

    # Propuštanje kroz preprocessor (OneHotEncoder + StandardScaler)
    try:
        input_preprocessed = preprocessor.transform(input_df)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Greška pri enkodiranju podataka: {str(e)}",
        )

    def _predict_one(model_key: str) -> dict:
        """Vraća predikciju za jedan model."""
        model_info = models[model_key]
        model = model_info["model"]
        y_pred = model.predict(input_preprocessed)[0]
        y_prob = model.predict_proba(input_preprocessed)[0, 1]
        return {
            "model": model_info["display_name"],
            "prediction": "yes" if y_pred == 1 else "no",
            "prediction_code": int(y_pred),
            "probability_yes": round(float(y_prob), 4),
            "probability_no": round(1.0 - float(y_prob), 4),
        }

    chosen_pred = _predict_one(chosen_key)
    #alternative_pred = _predict_one(alternative_key)

    return {
        "strategy": strategy,
        "strategy_description": STRATEGY_DESCRIPTIONS[strategy],
        "chosen_prediction": chosen_pred,
        #"alternative_prediction": alternative_pred,
    }


# ==================== ENDPOINTI ====================


@app.get("/")
def home():
    """Početna stranica — provera da API radi."""
    return {
        "message": "Bank Marketing Prediction API je pokrenut!",
        "docs": "/docs",
        "strategies": {
            "max_f2": STRATEGY_DESCRIPTIONS["max_f2"],
            "max_recall": STRATEGY_DESCRIPTIONS["max_recall"],
        },
        "endpoints": {
            "GET /": "Ova stranica",
            "GET /health": "Provera stanja API-ja",
            "GET /models": "Lista dostupnih modela, strategija i metrika",
            "POST /predict": "Predikcija za jednog klijenta (obavezno polje: strategy)",
            "POST /predict/batch": "Batch predikcija za više klijenata (obavezno polje: strategy)",
        },
    }


@app.get("/health")
def health():
    """Provera stanja API-ja — da li su preprocessor i modeli učitani."""
    loaded_models = [name for name, info in models.items() if info is not None]
    return {
        "status": "ok" if preprocessor is not None and loaded_models else "degraded",
        "preprocessor_loaded": preprocessor is not None,
        "models_loaded": loaded_models,
        "total_models_available": len(loaded_models),
    }


@app.get("/models")
def list_models():
    """Lista dva najbolja modela + strategije + metrike."""
    available_models = {}
    for fname, info in models.items():
        if info is not None:
            available_models[fname] = {
                "display_name": info["display_name"],
                "estimator_type": info["estimator"],
            }

    return {
        "available_models": available_models,
        "strategies": {
            "max_f2": {
                "description": STRATEGY_DESCRIPTIONS["max_f2"],
                "uses_model": "best_f2_model",
            },
            "max_recall": {
                "description": STRATEGY_DESCRIPTIONS["max_recall"],
                "uses_model": "best_recall_model",
            },
        },
        "metrics_table": metrics_text if metrics_text else "Metrike nisu dostupne (pokreni train.py)",
    }


@app.post("/predict")
def predict(client: SingleClientInput):
    """
    Predikcija za jednog klijenta sa izabranom strategijom.

    Šalješ atribute klijenta (JSON) + `strategy` (\"max_f2\" ili \"max_recall\").
    API vraća predikciju izabranog modela + alternativnog modela za poređenje.
    """
    client_data = ClientInput.model_validate(
        client.model_dump(exclude={"strategy"}, by_alias=True),
    )
    result = predict_client(client_data, client.strategy)
    return {
        "input": client.model_dump(by_alias=True, exclude={"strategy"}),
        "strategy": result["strategy"],
        "strategy_description": result["strategy_description"],
        "chosen_prediction": result["chosen_prediction"],
        #"alternative_prediction": result["alternative_prediction"],
    }


@app.post("/predict/batch")
def predict_batch(batch: BatchClientInput):
    """
    Batch predikcija za više klijenata sa izabranom strategijom.

    Šalješ listu klijenata + `strategy` (\"max_f2\" ili \"max_recall\").
    API vraća predikcije za svakog klijenta posebno.
    """
    if len(batch.clients) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maksimalan broj klijenata po batch zahtevu je 1000.",
        )

    results = []
    for i, client in enumerate(batch.clients):
        try:
            result = predict_client(client, batch.strategy)
            results.append(
                {
                    "client_index": i,
                    "input": client.model_dump(by_alias=True),
                    "strategy": result["strategy"],
                    "chosen_prediction": result["chosen_prediction"],
                    #"alternative_prediction": result["alternative_prediction"],
                }
            )
        except HTTPException as e:
            results.append(
                {
                    "client_index": i,
                    "error": e.detail,
                }
            )

    return {
        "total_clients": len(batch.clients),
        "strategy": batch.strategy,
        "strategy_description": STRATEGY_DESCRIPTIONS[batch.strategy],
        "successful": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
        "results": results,
    }