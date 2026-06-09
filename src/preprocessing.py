import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
import joblib
import os

# Kreiraj potrebne direktorijume ako ne postoje
os.makedirs('data', exist_ok=True)
os.makedirs('preprocessors', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

""" ========== UCITAVANJE PODATAKA ========== """
df = pd.read_csv('data/raw/bank-additional-full.csv', sep=';')
print(f"Učitano {df.shape[0]} redova i {df.shape[1]} kolona.")
#print(df.head()) #vraca prvih 5 redova

#print(df.isna().sum()) #ispisi koliko imas ukupno nedostajecih vrednosti
# NEMA IH
df = df.dropna()    #izbaci uzorak ako ima neku praznu kolonu
df = df.drop_duplicates()  #izbaci duplikate

# ukloni duration (poznat tek nakon zavrsetka poziva)
df = df.drop(columns=['duration'])   


""" ========== PREPROCESSING ========== """
# ------------ PDAYS --------------
# pravimo kolonu pdays_contatcet: 1 kontaktiran/ 0 nije (onaj sto ima 999)
# dobijamo info ko je stari, a ko novi klijent
df['pdays_contacted'] = (df['pdays'] != 999).astype(int)
# Postavljamo 999 na -1 
df['pdays'] = df['pdays'].replace(999, -1)
#print("pojavljivanja pdays=999: ", (df['pdays'] == 999).sum()) #nakon izbacivanja 
print(f"  pdays vrednosti: min = {df['pdays'].min()}, max = {df['pdays'].max()}")   #min i max vr
#print(f"  Broj kontaktiranih: {df['pdays_contacted'].sum()}") #br kontaktiranih

# prebroj koliko imamo unkown
#categorical_cols = ["job", "marital", "education", "default", "housing", "loan","contact", "poutcome", "month", "day_of_week" ]
#print((df[categorical_cols] == 'unknown').sum())

"""
ukloni previse slucajeva: 
    age: 473 outliera (donja granica=18, gornja granica=70)
    campaign: 2392 outliera (donja granica=1, gornja granica=6)
    previous: 282 outliera (gornja granica=2, 99. percentil)
    Ukupno uklonjeno redova sa outlierima: 3147
    Broj redova nakon uklanjanja outliera: 38029

# ---------------- OUTLINERI ------------------
total_outliers_removed = 0

# 1. AGE: starost ne može biti < 18
Q1_age = df['age'].quantile(0.25)   #25% ispod ove vr
Q3_age = df['age'].quantile(0.75)   #75% ispod ove vr
IQR_age = Q3_age - Q1_age       #50% sredine
# granice
lower_age = max(18, Q1_age - 1.5 * IQR_age)  #min br god je 18
upper_age = Q3_age + 1.5 * IQR_age
age_outliers = (df['age'] < lower_age) | (df['age'] > upper_age)
n_age = age_outliers.sum()
print(f"  age: {n_age} outliera (donja granica = {lower_age:.0f}, gornja granica = {upper_age:.0f})")
#ukloni oulinere
df = df[~age_outliers]
total_outliers_removed += n_age

# 2. CAMPAIGN: broj kontakata >= 1
Q1_camp = df['campaign'].quantile(0.25)
Q3_camp = df['campaign'].quantile(0.75)
IQR_camp = Q3_camp - Q1_camp
lower_camp = max(1, Q1_camp - 1.5 * IQR_camp)  
upper_camp = Q3_camp + 1.5 * IQR_camp
camp_outliers = (df['campaign'] < lower_camp) | (df['campaign'] > upper_camp)
n_camp = camp_outliers.sum()
print(f"  campaign: {n_camp} outliera (donja granica = {lower_camp:.0f}, gornja granica = {upper_camp:.0f})")
df = df[~camp_outliers]
total_outliers_removed += n_camp

# 3. PDAYS: Preskačemo IQR. pdays je već transformisan (999 -> -1),
#    a novi feature 'pdays_contacted' nosi informaciju o prethodnom kontaktu.
#    Zadržavamo sve vrednosti jer je anomalija (999) već rešena.
#    Eventualne ekstremne vrednosti za kontaktirane klijente (pdays > 27)
#    su retke i mogu biti validne (kampanja je trajala godinama).

# 4. PREVIOUS: 99. percentil (IQR bi uklonio sve > 0 jer je medijana = 0)
# vecina vr su 0, pa bi onda izbacio i ako neko ima 1
# cuva 99% baze poda, uklanja samo 1% cudnih vr
prev_p99 = df['previous'].quantile(0.99)
prev_outliers = df['previous'] > prev_p99
n_prev = prev_outliers.sum()
print(f"  previous: {n_prev} outliera (gornja granica={prev_p99:.0f}, 99. percentil)")
df = df[~prev_outliers]
total_outliers_removed += n_prev

print(f"  Ukupno uklonjeno redova sa outlierima: {total_outliers_removed}") """

# balans za y
print(df["y"].value_counts())

# ----------- ENKODIRANJE ------------- 
# CILJNA PROMENLJIVA
df['y'] = df['y'].map({'yes': 1, 'no': 0})

X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    random_state=42, stratify=y  #balansira odnos yes/no u train i test
)

print("Pre ciscenja anomalija:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

# -------------- ANOMALIJE ---------------
#izdvojimo samo numericke kolone
numeric_cols = X_train.select_dtypes(include=['number']).columns

# izbacujemo 3%
iso_forest = IsolationForest(contamination=0.03, random_state=42)

# Treniramo Isolation Forest i predviđamo anomalije SAMO na trening skupu
outlier_preds = iso_forest.fit_predict(X_train[numeric_cols])

# Zadržavamo samo normalne redove (tamo gde je rezultat 1) u X_train i y_train
# -1(anomalije) odbacujemo
X_train = X_train[outlier_preds == 1]
y_train = y_train[outlier_preds == 1]

# oko 1000 primera je uklonio -> moze se podesavati const
print("Nakon čišćenja anomalija pomoću Isolation Forest-a:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


nominal_features = [
    'job',
    'marital',
    'contact',
    'month',
    'day_of_week',
    'poutcome',
    'default',
    'housing',
    'loan'
]

education_order = [[
    'unknown',
    'illiterate',
    'basic.4y',
    'basic.6y',
    'basic.9y',
    'high.school',
    'professional.course',
    'university.degree'
]]

# handle_unknown - ne rusi program ako se pojavi nova kategorija
# drop - izbaci jednu kolonu (ta je podrazumevana ako su sve ostale 0)
preprocessor = ColumnTransformer(
    transformers=[
        (
            'nominal',
            OneHotEncoder(drop='first', handle_unknown='ignore'),
            nominal_features
        ),
        (
            'education',
            OrdinalEncoder(categories=education_order),
            ['education']
        )
    ],
    remainder='passthrough' #ako ima neke koje ovde nisu obradjene, ostavi takve
)

# transform - primeni naucena pravila (pozivamo enkoder)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# pokaze kako je skocio br kolona (zbog enkodiranja)
print("Train shape nakon preprocessing-a:", X_train_preprocessed.shape)
print("Test shape nakon preprocessing-a:", X_test_preprocessed.shape)


# ------------- CUVANJE PODATAKA ----------------
df.to_csv('data/processed/bank-additional-cleaned.csv', index=False)

# cuvanje train i test nekodiranih skupova
train_df = X_train.copy()
train_df['y'] = y_train
train_df.to_csv('data/processed/bank-additional-train.csv', index=False)

test_df = X_test.copy()
test_df['y'] = y_test
test_df.to_csv('data/processed/bank-additional-test.csv', index=False)


joblib.dump(preprocessor, 'preprocessors/preprocessor.pkl')    #sacuvaj preprocesor
joblib.dump(iso_forest, 'preprocessors/isolation_forest.pkl') #sacuvaj isolatin forest model

np.save('data/processed/X_train_preprocessed.npy', X_train_preprocessed)
np.save('data/processed/y_train.npy', y_train.values)
np.save('data/processed/X_test_preprocessed.npy', X_test_preprocessed)
np.save('data/processed/y_test.npy', y_test.values)