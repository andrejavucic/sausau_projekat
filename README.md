# Sausau Projekat — Bank Marketing Classification

Projekat za predviđanje pretplate klijenata na oročeni depozit (`y` = yes/no) korišćenjem *Bank Marketing* dataseta (UCI Machine Learning Repository). Cilj je izgraditi modele mašinskog učenja koji maksimizuju **recall** i **F2** skor — jer je banci važnije da ne propusti potencijalnog klijenta nego da pošalje par bespotrebnih poziva.

## Struktura projekta

```
sausau_projekat/
├── data/
│   ├── raw/                         # Sirovi podaci (bank-additional-full.csv)
│   └── processed/                   # Očišćeni, enkodirani i podeljeni podaci (.npy, .csv)
├── src/
│   ├── EDA.py                       # Eksplorativna analiza podataka
│   ├── preprocessing.py             # Čišćenje, enkodiranje, SMOTE, čuvanje podataka
│   ├── train.py                     # Treniranje i evaluacija 5 modela (GridSearchCV + K-Fold)
│   ├── analysis.py                  # Dubinska analiza modela (pragovi, overfitting, baseline)
│   └── feature.py                   # Feature importance i poređenje Top 10/20/svi atributi
├── models/                          # Sačuvani trenirani modeli (.pkl)
├── preprocessors/                   # Sačuvani preprocessor (OneHotEncoder + Scaler)
├── analysis/                        # CSV tabele sa metrikama i dijagnostikom
│   └── figures/                     # Grafikoni: ROC, confusion matrice, overfitting, itd.
├── EDA_figures/                     # Grafikoni iz eksplorativne analize
├── main.py                          # Ulazna tačka (placeholder)
├── ui.py                            # Streamlit korisnički interfejs za predikcije
├── pyproject.toml                   # Konfiguracija projekta i zavisnosti
└── README.md
```

## Dataset

**Bank Marketing (Additional Full)** — 41.188 uzoraka, 20 ulaznih atributa + ciljna promenljiva `y`.

Atributi uključuju:
- **Numeričke**: `age`, `campaign`, `pdays`, `previous`, `cons.price.idx`, `cons.conf.idx`, `nr.employed` (i izbačeni `emp.var.rate`, `euribor3m`)
- **Kategorijske**: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`

Kolone `duration` i `day_of_week` su uklonjene (duration je poznat tek nakon poziva, day_of_week nema dovoljan uticaj).

Dataset je **nebalansiran**: ~89% klijenata nije prihvatilo ponudu (`no`), ~11% jeste (`yes`).

## Pipeline

### 1. Eksplorativna analiza (`src/EDA.py`)

- Histogrami svih numeričkih atributa
- Distribucija ciljne promenljive `y`
- Pretplata po zanimanju, obrazovanju, mesecu, danu u nedelji, ishodu prethodne kampanje
- Boxplot starosti u odnosu na pretplatu
- Stopa pretplate po starosnim grupama
- Top 5 zanimanja po stopi pretplate
- **Korelaciona matrica** (otkrivena visoka korelacija: `euribor3m` ↔ `emp.var.rate` = 0.97, `euribor3m` ↔ `nr.employed` = 0.95 → 2 od 3 izbačene)

### 2. Preprocesiranje (`src/preprocessing.py`)

| Korak | Detalji |
|-------|---------|
| Čišćenje | Uklanjanje duplikata, zamena `pdays=999` → `-1` |
| Uklanjanje kolona | `duration`, `day_of_week`, `emp.var.rate`, `euribor3m` |
| Outlieri (zakomentarisano) | IQR za `age` i `campaign`, 99. percentil za `previous` — 3.147 redova (~7.6%) |
| Enkodiranje | **OneHotEncoder** (`drop='first'`) za kategorijske; **StandardScaler** za numeričke |
| Train/Test split | 80/20, `stratify=y`, `random_state=42` |
| SMOTE | Balansiranje samo na train skupu (nakon enkodiranja) |
| Čuvanje | `preprocessor.pkl`, svi `.npy` i `.csv` fajlovi u `data/processed/` |

### 3. Treniranje modela (`src/train.py`)

Pet modela sa optimizovanim hiperparametrima (prvobitno pronađenim GridSearch-om):

| Model | Ključni Hiperparametri |
|-------|------------------------|
| **Logistic Regression** | `C=0.01`, `penalty='l1'`, `solver='saga'`, `max_iter=2000` |
| **KNN** | `n_neighbors=25`, `weights='uniform'`, `p=1` (Manhattan) |
| **Decision Tree** | `criterion='entropy'`, `max_depth=5`, `min_samples_leaf=1` |
| **Random Forest** | `n_estimators=50`, `max_depth=5`, `min_samples_split=10`, `max_features='sqrt'` |
| **Gradient Boosting** | `n_estimators=100`, `learning_rate=0.03`, `max_depth=2`, `subsample=0.8` |

- **Stratified K-Fold** (k=5) unakrsna validacija
- SMOTE primenjen unutar pipeline-a za svaki fold (bez curenja podataka)
- Metrika optimizacije: **F2 skor** (β=2, recall je duplo važniji od precision)
- Evaluacija na test skupu sa svim metrikama: Accuracy, Precision, Recall, F1, F2, ROC-AUC

### 4. Rezultati

```
Model                          F2     Recall  Precision         F1    ROC-AUC
--------------------------------------------------------------------------------
Logistic Regression        0.5532     0.6562     0.3398     0.4478     0.7954
KNN                        0.5266     0.6530     0.2968     0.4081     0.7725
Decision Tree              0.5316     0.5754     0.4073     0.4770     0.7748
Random Forest              0.5552     0.6261     0.3822     0.4747     0.7951
Gradient Boosting          0.5389     0.5948     0.3915     0.4722     0.7971

🏆 Najbolji F2:   Random Forest (0.5552)
🏆 Najbolji Recall: Logistic Regression (0.6562)
```

### 5. Dubinska analiza (`src/analysis.py`)

- **Analiza pragova** (0.30, 0.35, 0.40, 0.50) — heatmape za Recall i F2 po pragovima
- **Dijagnostika overfittinga** — poređenje Train vs Test F1/F2 (svi modeli pokazuju dobru generalizaciju)
- **Baseline poređenje** — `DummyClassifier(stratified)` kao minimalna granica
- **ROC krive** za svih 5 modela
- **Confusion matrice** (prag 0.5)
- **Uporedni bar plot** — F2, Recall, ROC-AUC za sve modele

### 6. Feature importance (`src/feature.py`)

- Ekstrakcija važnosti atributa za **Logistic Regression** (koeficijenti), **Random Forest** i **Gradient Boosting** (feature importances)
- Bar plotovi Top 10 najvažnijih faktora po modelu
- **Heatmap** poređenja važnosti atributa između modela (Top 20, normalizovano na sumu=1)
- **Testiranje performansi** sa Top 10, Top 20 i svim atributima — poređenje F2 i Recall skora

### 7. Streamlit UI (`ui.py`)

Interaktivni korisnički interfejs za predikcije, napravljen u **Streamlit**-u. Omogućava korisniku da ručno unese podatke o klijentu i dobije predikciju bez potrebe za slanjem HTTP zahteva ili korišćenjem alata poput `curl`-a.

#### Funkcionalnosti

- **Selectbox** padajući meniji za sve kategorijske atribute:
  - *Lični podaci:* `job` (12 opcija), `marital` (4 opcije), `education` (8 opcija)
  - *Finansijski podaci:* `default`, `housing`, `loan` (po 3 opcije)
  - *Kontakt kampanja:* `contact` (2 opcije), `month` (12 opcija), `poutcome` (3 opcije)
- **Number input** polja za 7 numeričkih atributa sa definisanim opsezima i podrazumevanim vrednostima (`age`, `campaign`, `pdays`, `previous`, `cons.price.idx`, `cons.conf.idx`, `nr.employed`)
- **Selectbox za strategiju** (`max_f2` / `max_recall`) sa tooltip objašnjenjem
- **Predict dugme** koje pokreće predikciju i prikazuje:
  - YES/NO predikciju sa emoji indikatorom
  - Verovatnoću pretplate (YES) i odbijanja (NO) u decimalnom i procentualnom formatu
  - **Progress bar** za vizuelni prikaz verovatnoće
  - Tip modela koji je korišćen za predikciju

#### Strategije

| Strategija | Model | Opis |
|------------|-------|------|
| `max_f2` | `best_f2_model` | Balansira Recall i Precision (Recall ×2), hvata što više potencijalnih klijenata uz prihvatljiv broj lažnih pozitiva |
| `max_recall` | `best_recall_model` | Maksimizira Recall bez obzira na Precision — banka ne želi da propusti nijednog klijenta |

#### Pokretanje

```bash
streamlit run ui.py
```

> **Napomena:** UI zahteva da su prethodno pokrenuti `src/preprocessing.py` i `src/train.py` kako bi postojali `preprocessor.pkl` i trenirani modeli u `models/` direktorijumu.

## Kako pokrenuti

### Preduslovi

- Python ≥ 3.14
- Preporučeni menadžer paketa: **uv**

```bash
git clone https://github.com/andrejavucic/sausau_projekat.git
cd sausau_projekat

# Instalacija zavisnosti
uv sync
# ili: pip install -e .

# Postavite bank-additional-full.csv u data/raw/
```

### Redosled izvršavanja

```bash
# 1. Eksplorativna analiza (generiše EDA_figures/)
python -m src.EDA

# 2. Preprocesiranje (generiše data/processed/ i preprocessors/)
python -m src.preprocessing

# 3. Treniranje modela (generiše models/ i analysis/metrics_table.txt)
python -m src.train

# 4. Dubinska analiza (generiše analysis/figures/ i dijagnostiku)
python -m src.analysis

# 5. Feature importance analiza
python -m src.feature
```

## Zavisnosti

| Paket | Namena |
|-------|--------|
| `pandas` | Manipulacija podacima |
| `numpy` | Numeričke operacije |
| `scikit-learn` | Modeli, enkodiranje, metrike, GridSearch |
| `imbalanced-learn` | SMOTE oversampling |
| `matplotlib`, `seaborn` | Vizualizacija |
| `joblib` | Serijalizacija modela i preprocesora |