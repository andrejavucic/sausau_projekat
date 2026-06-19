import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
import joblib
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Kreiraj potrebne direktorijume ako ne postoje
os.makedirs('data', exist_ok=True)
os.makedirs('preprocessors', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
print()

""" ========== UCITAVANJE PODATAKA ========== """
df = pd.read_csv('data/raw/bank-additional-full.csv', sep=';')
print(f"Učitano {df.shape[0]} redova i {df.shape[1]} kolona.")
#print(df.head()) #vraca prvih 5 redova
print()

#print(df.isna().sum()) #ispisi koliko imas ukupno nedostajecih vrednosti
# NEMA IH
df = df.dropna()    #izbaci uzorak ako ima neku praznu kolonu
df = df.drop_duplicates()  #izbaci duplikate

# ukloni duration (poznat tek nakon zavrsetka poziva)
df = df.drop(columns=['duration', 'day_of_week'])   

"""
    IZ MATRICE ZAVISNOSTI (KORELACIJE):
    - veliku zavisnost imaju euribor3m i emp.var.rate (0.97)
    - isto i za euribor3m i nr.employed (0.95)
    - nr.employed i emp.var.rate (0.91)

    ZAKLJUCAK: mogu se izbaciti 2/3 -> prenose istu info
"""
df = df.drop(columns=['emp.var.rate', 'euribor3m'])      


""" ========== PREPROCESSING ========== """
# ------------ PDAYS -------------
# Postavljamo 999 na -1 
df['pdays'] = df['pdays'].replace(999, -1)
print("pojavljivanja pdays=999 : ", (df['pdays'] == 999).sum())     #nakon izbacivanja -> 0 
print(f"pdays vrednosti: min = {df['pdays'].min()}, max = {df['pdays'].max()}")   #min i max vr
print()

# prebroj koliko imamo unkown
# nema ih u month, poutcome i contact
# u ostalim ih ima podosta (najmanje u martial -> 80)
categorical_cols = ["job", "marital", "education", "default", "housing", "loan","contact", "poutcome", "month" ]
print((df[categorical_cols] == 'unknown').sum())

#poencijalni outlineri -> ne utice na izlaz (nema granica za yes po godinama)
print("Maloletni: ", df.loc[df['age'] < 18, ['age', 'y']])  
print("Prestari: ", df.loc[df['age'] > 90, ['age', 'y']])

"""
ukloni previse slucajeva: 
    age: 473 outliera (donja granica=18, gornja granica=70)
    campaign: 2392 outliera (donja granica=1, gornja granica=6)
    previous: 282 outliera (gornja granica=2, 99. percentil)
    Ukupno uklonjeno redova sa outlierima: 3147

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
print()

# stratify=y  balansira odnos yes/no u train i test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
# → 80/10/10

print("Pre enkodiranja:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
#print("y_train:", y_train.shape)
print()

# -------------- ANOMALIJE ---------------
#izdvojimo samo numericke kolone
numeric_cols = X_train.select_dtypes(include=['number']).columns

# izbacujemo 3%
#iso_forest = IsolationForest(contamination=0.03, random_state=42)

# Treniramo Isolation Forest i predviđamo anomalije SAMO na trening skupu
#outlier_preds = iso_forest.fit_predict(X_train[numeric_cols])

# Zadržavamo samo normalne redove (tamo gde je rezultat 1) u X_train i y_train
# -1(anomalije) odbacujemo
#X_train = X_train[outlier_preds == 1]
#y_train = y_train[outlier_preds == 1]

# oko 1000 primera je uklonio -> moze se podesavati const
#print("Nakon čišćenja anomalija pomoću Isolation Forest-a:")
#print("X_train:", X_train.shape)
#print("y_train:", y_train.shape)


nominal_features = [
    'job',
    'marital',
    'contact',
    'month',
    #'day_of_week',
    'poutcome',
    'default',
    'housing',
    'loan',
    'education'
]

# handle_unknown - ne rusi program ako se pojavi kategorija koju ranije nije video
# drop - izbaci jednu kolonu (ta je podrazumevana ako su sve ostale 0)
preprocessor = ColumnTransformer(
    transformers=[
        (
            'nominal',      # sparse_output -> da bih ga naterala da mi vrati gustu matricu (sta god to bilo)
            OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False),
            nominal_features
        ),
        # SKLARIRANJE 
        (
            'numeric',  
            StandardScaler(), 
            numeric_cols    #samo kolone sa br vr
        )
    ],
    remainder='passthrough' #ako ima neke koje ovde nisu obradjene, ostavi takve
)

# transform - primeni naucena pravila (pozivamo enkoder)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)

# pokaze kako je skocio br kolona (zbog enkodiranja)
print("Train shape nakon preprocessing-a:", X_train_preprocessed.shape)
print("Validation shape nakon preprocessing-a:", X_val_preprocessed.shape)
print("Test shape nakon preprocessing-a:", X_test_preprocessed.shape)
print()

# smote - vestacki pravi nove pod u datasetu kako bi izjednacio odnos yes/no za y
# ali samo za train skup
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# provera balansa y (nakon smote)
print(f"Pre SMOTE: {X_train_preprocessed.shape}, DA={sum(y_train==1)}, NE={sum(y_train==0)}" )
print(f"Posle SMOTE: {X_train_resampled.shape}, DA={sum(y_train_resampled==1)}, NE={sum(y_train_resampled==0)}")
print()

# ------------- CUVANJE PODATAKA ----------------
df.to_csv('data/processed/bank-additional-cleaned.csv', index=False)

# cuvanje train, test i val nekodiranih skupova
train_df = X_train.copy()
train_df['y'] = y_train
train_df.to_csv('data/processed/bank-additional-train.csv', index=False)

val_df = X_val.copy()
val_df['y'] = y_val
val_df.to_csv('data/processed/bank-additional-val.csv', index=False)

test_df = X_test.copy()
test_df['y'] = y_test
test_df.to_csv('data/processed/bank-additional-test.csv', index=False)

joblib.dump(preprocessor, 'preprocessors/preprocessor.pkl')    #sacuvaj preprocesor
#joblib.dump(iso_forest, 'preprocessors/isolation_forest.pkl') #sacuvaj isolatin forest model

# cuvamo nazive atributa, kasnije za poredjenje koji nam je najznacniji
feature_names = preprocessor.get_feature_names_out()
np.save('data/processed/feature_names.npy', feature_names)

np.save('data/processed/X_train_preprocessed.npy', X_train_preprocessed)
np.save('data/processed/y_train.npy', y_train)

np.save('data/processed/X_train_resampled.npy', X_train_resampled)
np.save('data/processed/y_train_resampled.npy', y_train_resampled)

np.save('data/processed/X_val_preprocessed.npy', X_val_preprocessed)   
np.save('data/processed/y_val.npy', y_val.values)

np.save('data/processed/X_test_preprocessed.npy', X_test_preprocessed)
np.save('data/processed/y_test.npy', y_test.values)