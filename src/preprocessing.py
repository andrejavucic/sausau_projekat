import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

""" ========== UCITAVANJE PODATAKA ========== """
df = pd.read_csv('data/bank-additional-full.csv', sep=';')
print(f"Učitano {df.shape[0]} redova i {df.shape[1]} kolona.")
df.head()


""" ========== POČETNO PREPROCESIRANJE - PROVERA NEDOSTAJUĆIH VREDNOSTI ========== """

print("\n========== 1. PROVERA PRAVIH NEDOSTAJUĆIH (NaN) VREDNOSTI ==========")
# nan -> nema podataka za tu kolonu
nan_counts = df.isnull().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) == 0:
    print("Nema NaN vrednosti ni u jednoj koloni.")
else:
    print("NaN vrednosti po kolonama:")
    print(nan_cols)

print("\n========== 2. PROVERA 'unknown' VREDNOSTI U SVIM KATEGORIJSKIM KOLONAMA ==========")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    unknown_count = df[col].value_counts().get('unknown', 0)
    if unknown_count > 0:
        print(f"  {col}: {unknown_count} unknown vrednosti ({unknown_count / len(df) * 100:.1f}%)")
    else:
        print(f"  {col}: 0 unknown vrednosti")

print("\n========== 3. TRETMAN 'unknown' VREDNOSTI ==========")
# Strategija:
#   - Za education: unknown se enkodira kao -1 (posebna kategorija)
#   - Za default, housing, loan: unknown je validna kategorija (klijent ne zna / nije odgovorio)
#   - Za job, marital, poutcome: unknown ostaje kao posebna kategorija kroz one-hot encoding
#     jer je procenat mali, a informacija da je vrednost nepoznata može biti signal za model
# Ove odluke su dokumentovane i sprovode se kroz enkodiranje u nastavku.

# Prikaz unknown raspodele za ključne kolone
unknown_summary = {}
unknown_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
for col in unknown_cols:
    unknown_summary[col] = df[col].value_counts().get('unknown', 0)
    print(f"  {col}: {unknown_summary[col]} unknown")

print("\n========== 4. DETEKCIJA ANOMALIJE: pdays = 999 ==========")
pdays_999_count = (df['pdays'] == 999).sum()
print(f"  pdays=999 (nikad nije kontaktiran): {pdays_999_count} redova ({pdays_999_count / len(df) * 100:.1f}%)")
print("  -> Kreiramo binarni indikator 'pdays_contacted' i postavljamo pdays=999 na -1")


""" ========== EKSPLORATIVNA ANALIZA PODATAKA (EDA) ========== """
# Pravimo kopiju za EDA (pre enkodiranja)
df_eda = df.copy()
df_eda['y_num'] = df_eda['y'].map({'yes': 1, 'no': 0})

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================
# 1. DISTRIBUCIJA CILJNE PROMENLJIVE
# ============================================
df_eda['y'].value_counts().plot(kind='bar', title='Distribucija pretplate (y)', color=['skyblue', 'salmon'])
plt.xlabel('Pretplata')
plt.ylabel('Broj klijenata')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('data/eda_target_distribution.png')
plt.show()
plt.close()

# ============================================
# 2. PRETPLATA PO KATEGORIJSKIM ATRIBUTIMA
# ============================================
# 2a. Pretplata po zanimanju (job)
plt.figure(figsize=(12, 6))
job_order = df_eda['job'].value_counts().index
sns.countplot(data=df_eda, x='job', hue='y', order=job_order)
plt.title('Pretplata po zanimanju', fontsize=14)
plt.xlabel('Zanimanje')
plt.ylabel('Broj klijenata')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Pretplata', labels=['Ne', 'Da'])
plt.tight_layout()
plt.savefig('data/eda_job.png')
plt.show()
plt.close()

# 2b. Pretplata po nivou obrazovanja
plt.figure(figsize=(10, 6))
education_order = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']
sns.countplot(data=df_eda, x='education', hue='y', order=education_order)
plt.title('Pretplata po nivou obrazovanja', fontsize=14)
plt.xlabel('Nivo obrazovanja')
plt.ylabel('Broj klijenata')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Pretplata', labels=['Ne', 'Da'])
plt.tight_layout()
plt.savefig('data/eda_education.png')
plt.show()
plt.close()

# 2c. Pretplata po bračnom statusu
plt.figure(figsize=(8, 6))
marital_order = df_eda['marital'].value_counts().index
sns.countplot(data=df_eda, x='marital', hue='y', order=marital_order)
plt.title('Pretplata po bračnom statusu', fontsize=14)
plt.xlabel('Bračni status')
plt.ylabel('Broj klijenata')
plt.legend(title='Pretplata', labels=['Ne', 'Da'])
plt.tight_layout()
plt.savefig('data/eda_marital.png')
plt.show()
plt.close()

# 2d. Pretplata po tipu kontakta
plt.figure(figsize=(6, 6))
sns.countplot(data=df_eda, x='contact', hue='y')
plt.title('Pretplata po tipu kontakta', fontsize=14)
plt.xlabel('Tip kontakta')
plt.ylabel('Broj klijenata')
plt.legend(title='Pretplata', labels=['Ne', 'Da'])
plt.tight_layout()
plt.savefig('data/eda_contact.png')
plt.show()
plt.close()

# 2e. Pretplata po mesecu
plt.figure(figsize=(12, 6))
month_order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
sns.countplot(data=df_eda, x='month', hue='y', order=month_order)
plt.title('Pretplata po mesecu', fontsize=14)
plt.xlabel('Mesec')
plt.ylabel('Broj klijenata')
plt.legend(title='Pretplata', labels=['Ne', 'Da'])
plt.tight_layout()
plt.savefig('data/eda_month.png')
plt.show()
plt.close()

# ============================================
# 3. NUMERIČKI ATRIBUTI U ODNOSU NA CILJNU PROMENLJIVU
# ============================================
# 3a. Starost (age) - boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_eda, x='y', y='age')
plt.title('Distribucija starosti u odnosu na pretplatu', fontsize=14)
plt.xlabel('Pretplata')
plt.ylabel('Starost (godine)')
plt.tight_layout()
plt.savefig('data/eda_age_boxplot.png')
plt.show()
plt.close()

# 3b. Starost - histogram poređenje
plt.figure(figsize=(10, 6))
sns.histplot(df_eda[df_eda['y'] == 'no']['age'], color='skyblue', label='Nema pretplatu', alpha=0.5)
sns.histplot(df_eda[df_eda['y'] == 'yes']['age'], color='salmon', label='Ima pretplatu', alpha=0.5)
plt.title('Distribucija starosti - poređenje', fontsize=14)
plt.xlabel('Starost (godine)')
plt.ylabel('Broj klijenata')
plt.legend()
plt.tight_layout()
plt.savefig('data/eda_age_histogram.png')
plt.show()
plt.close()

# 3c. Broj kontakata (campaign) - boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_eda, x='y', y='campaign')
plt.title('Broj kontakata tokom kampanje u odnosu na pretplatu', fontsize=14)
plt.xlabel('Pretplata')
plt.ylabel('Broj kontakata')
plt.tight_layout()
plt.savefig('data/eda_campaign_boxplot.png')
plt.show()
plt.close()

# 3d. Prethodni kontakti (previous) - boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_eda, x='y', y='previous')
plt.title('Broj prethodnih kontakata u odnosu na pretplatu', fontsize=14)
plt.xlabel('Pretplata')
plt.ylabel('Broj prethodnih kontakata')
plt.tight_layout()
plt.savefig('data/eda_previous_boxplot.png')
plt.show()
plt.close()

# 3e. pdays - boxplot (pre tretmana, da se vidi anomalija 999)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_eda, x='y', y='pdays')
plt.title('Broj dana od poslednjeg kontakta u odnosu na pretplatu (pre tretmana)', fontsize=14)
plt.xlabel('Pretplata')
plt.ylabel('Broj dana (999 = nikad kontaktiran)')
plt.tight_layout()
plt.savefig('data/eda_pdays_boxplot.png')
plt.show()
plt.close()

# ============================================
# 4. DODATNI GRAFICI ZA DUBLJU ANALIZU
# ============================================
# 4a. Stopa pretplate po starosnim grupama
df_eda['age_group'] = pd.cut(df_eda['age'], bins=[18, 30, 40, 50, 60, 100],
                              labels=['18-30', '30-40', '40-50', '50-60', '60+'])
subscription_rate = df_eda.groupby('age_group')['y_num'].mean() * 100
plt.figure(figsize=(8, 6))
subscription_rate.plot(kind='bar', color='skyblue')
plt.title('Stopa pretplate (%) po starosnim grupama', fontsize=14)
plt.xlabel('Starosna grupa')
plt.ylabel('Stopa pretplate (%)')
plt.xticks(rotation=0)
for i, v in enumerate(subscription_rate):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('data/eda_age_group_rate.png')
plt.show()
plt.close()

# 4b. Stopa pretplate po nivou obrazovanja (procentualno)
education_rate = df_eda.groupby('education')['y_num'].mean() * 100
plt.figure(figsize=(10, 6))
education_rate.sort_values().plot(kind='barh', color='lightcoral')
plt.title('Stopa pretplate (%) po nivou obrazovanja', fontsize=14)
plt.xlabel('Stopa pretplate (%)')
plt.ylabel('Nivo obrazovanja')
for i, v in enumerate(education_rate.sort_values().values):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
plt.tight_layout()
plt.savefig('data/eda_education_rate.png')
plt.show()
plt.close()

# 4c. Stopa pretplate po zanimanju
job_rate = df_eda.groupby('job')['y_num'].mean() * 100
plt.figure(figsize=(10, 8))
job_rate.sort_values().plot(kind='barh', color='lightgreen')
plt.title('Stopa pretplate (%) po zanimanju', fontsize=14)
plt.xlabel('Stopa pretplate (%)')
plt.ylabel('Zanimanje')
plt.tight_layout()
plt.savefig('data/eda_job_rate.png')
plt.show()
plt.close()

# ============================================
# 5. KORELACIONA MATRICA
# ============================================
print("\n========== 5. KORELACIONA MATRICA NUMERIČKIH ATRIBUTA ==========")
numeric_cols_raw = ['age', 'duration', 'campaign', 'pdays', 'previous',
                    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
# Napomena: duration uključen samo za EDA analizu, biće uklonjen za model
# Dodajemo y_num za korelaciju sa ciljnom promenljivom
corr_df = df_eda[numeric_cols_raw + ['y_num']].copy()

plt.figure(figsize=(12, 10))
corr_matrix = corr_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
plt.title('Korelaciona matrica numeričkih atributa (sa duration i y_num)', fontsize=14)
plt.tight_layout()
plt.savefig('data/eda_correlation_heatmap.png')
plt.show()
plt.close()

# Ispis korelacija sa ciljnom promenljivom (sortirano)
print("\nKorelacija numeričkih atributa sa ciljnom promenljivom (y_num):")
y_corr = corr_matrix['y_num'].drop('y_num').sort_values(key=abs, ascending=False)
for col, val in y_corr.items():
    print(f"  {col}: {val:+.3f}")

# Ispis jakih međusobnih korelacija (|r| > 0.7)
print("\nJake međusobne korelacije među atributima (|r| > 0.7):")
strong_corrs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
if strong_corrs:
    for c1, c2, val in strong_corrs:
        print(f"  {c1} <-> {c2}: {val:+.3f}")
else:
    print("  Nema jakih međusobnih korelacija iznad 0.7.")


""" ========== PREPROCESSING ========== """

# ============================================
# TRETMAN pdays ANOMALIJE
# ============================================
print("\n========== TRETMAN pdays = 999 ==========")
# Kreiramo binarni indikator: da li je klijent ikad kontaktiran ranije
# pravimo kolonu pdays_contatcet: 1 kontaktiran/ 0 nije (onaj sto ima 999)
# dobijamo info ko je stari, a ko novi klijent
df['pdays_contacted'] = (df['pdays'] != 999).astype(int)
# Postavljamo 999 na -1 (neutralna vrednost koja označava "nije kontaktiran")
df['pdays'] = df['pdays'].replace(999, -1)
print(f"  Kreiran 'pdays_contacted': 1 = kontaktiran, 0 = nije kontaktiran")
print(f"  pdays vrednosti: min={df['pdays'].min()}, max={df['pdays'].max()}")
print(f"  Broj kontaktiranih: {df['pdays_contacted'].sum()}")

# ============================================
# ENKODIRANJE KATEGORIJSKIH VARIJABLI
# ============================================
print("\n========== ENKODIRANJE ==========")

# ONE-HOT ENCODING za nominalne kolone
nominal_cols = ['job', 'marital', 'contact', 'month', 'day_of_week', 'poutcome', 'default', 'housing', 'loan']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
print(f"  One-hot encoded kolone: {nominal_cols}")
print(f"  Broj kolona nakon one-hot encodinga: {df.shape[1]}")

# LABEL ENCODING za education (ordinalna varijabla)
education_order = {
    'unknown': -1,           # nepoznato kao posebna kategorija
    'illiterate': 0,
    'basic.4y': 1,
    'basic.6y': 2,
    'basic.9y': 3,
    'high.school': 4,
    'professional.course': 5,
    'university.degree': 6
}
df['education_encoded'] = df['education'].map(education_order)
df = df.drop(columns=['education'])
print(f"  Label encoded: education -> education_encoded")

# CILJNA PROMENLJIVA
df['y'] = df['y'].map({'yes': 1, 'no': 0})
print(f"  Ciljna promenljiva y enkodirana: yes=1, no=0")

# ============================================
# UKLANJANJE ATRIBUTA KOJI NISU DOSTUPNI U TRENUTKU PREDIKCIJE
# ============================================
print("\n========== UKLANJANJE NEDOSTUPNIH ATRIBUTA ==========")
# duration: trajanje poziva - poznat tek nakon završetka poziva, nije dostupan unapred
df_model = df.drop(columns=['duration'])
print(f"  Uklonjen 'duration' (trajanje poziva nije dostupno u trenutku predikcije)")

# ODLUKA O MAKROEKONOMSKIM INDIKATORIMA:
# emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed su
# makroekonomski indikatori koji zavise od datuma kampanje. U realnom scenariju,
# oni nisu poznati za buduće datume. Međutim, za potrebe ovog projekta ih
# zadržavamo jer:
#   1. Dataset je iz fiksne kampanje (2008-2013) i model se evaluira na istom periodu
#   2. Omogućavaju modelu da uhvati ekonomski kontekst
#   3. U produkciji bi se koristile najnovije dostupne vrednosti ili prognoze
# Ako bi se model koristio za buduće kampanje, ovi atributi bi morali biti uklonjeni.
print(f"  Makroekonomski indikatori zadržani (videti komentar u kodu za obrazloženje)")

# ============================================
# TRETMAN EKSTREMNIH VREDNOSTI (OUTLIERS)
# ============================================
print("\n========== TRETMAN OUTLIERA ==========")
# VAŽNO: IQR metoda nije pogodna za pdays i previous jer su distribucije
# ekstremno skewovane (96% pdays=-1 nakon transformacije, vecina previous=0).
# Za njih koristimo percentilni pristup.

total_outliers_removed = 0

# 1. AGE: IQR sa realisticnim granicama (starost ne može biti < 18)
Q1_age = df_model['age'].quantile(0.25)
Q3_age = df_model['age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
lower_age = max(18, Q1_age - 1.5 * IQR_age)   # realisticna donja granica
upper_age = Q3_age + 1.5 * IQR_age
age_outliers = (df_model['age'] < lower_age) | (df_model['age'] > upper_age)
n_age = age_outliers.sum()
print(f"  age: {n_age} outliera (donja granica={lower_age:.0f}, gornja granica={upper_age:.0f})")
df_model = df_model[~age_outliers]
total_outliers_removed += n_age

# 2. CAMPAIGN: IQR sa realisticnim granicama (broj kontakata >= 1)
Q1_camp = df_model['campaign'].quantile(0.25)
Q3_camp = df_model['campaign'].quantile(0.75)
IQR_camp = Q3_camp - Q1_camp
lower_camp = max(1, Q1_camp - 1.5 * IQR_camp)   # realisticna donja granica
upper_camp = Q3_camp + 1.5 * IQR_camp
camp_outliers = (df_model['campaign'] < lower_camp) | (df_model['campaign'] > upper_camp)
n_camp = camp_outliers.sum()
print(f"  campaign: {n_camp} outliera (donja granica={lower_camp:.0f}, gornja granica={upper_camp:.0f})")
df_model = df_model[~camp_outliers]
total_outliers_removed += n_camp

# 3. PDAYS: Preskačemo IQR. pdays je već transformisan (999 -> -1),
#    a novi feature 'pdays_contacted' nosi informaciju o prethodnom kontaktu.
#    Zadržavamo sve vrednosti jer je anomalija (999) već rešena.
#    Eventualne ekstremne vrednosti za kontaktirane klijente (pdays > 27)
#    su retke i mogu biti validne (kampanja je trajala godinama).
print(f"  pdays: outlier detekcija preskočena (već transformisan, pdays_contacted nosi informaciju)")

# 4. PREVIOUS: 99. percentil (IQR bi uklonio sve > 0 jer je medijana = 0)
# vecina vr su 0, pa bi onda izbacio i ako neko ima 1
# cuva 99% baze poda, uklanja samo 1% cudnih vr
prev_p99 = df_model['previous'].quantile(0.99)
prev_outliers = df_model['previous'] > prev_p99
n_prev = prev_outliers.sum()
print(f"  previous: {n_prev} outliera (gornja granica={prev_p99:.0f}, 99. percentil)")
df_model = df_model[~prev_outliers]
total_outliers_removed += n_prev

print(f"  Ukupno uklonjeno redova sa outlierima: {total_outliers_removed}")
print(f"  Broj redova nakon uklanjanja outliera: {df_model.shape[0]}")

# ============================================
# PRIPREMA ZA MODEL (TRAIN/TEST SPLIT)
# ============================================
print("\n========== TRAIN/TEST SPLIT ==========")
X = df_model.drop(columns=['y'])
y = df_model['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Trening skup: {X_train.shape[0]} redova")
print(f"  Test skup: {X_test.shape[0]} redova")
print(f"  Broj feature-a (kolona u X): {X.shape[1]}")

# Provera balansa klasa
print(f"\n  Balans klasa u trening skupu:")
print(f"    y=0 (nema pretplatu): {y_train.value_counts().get(0, 0)} ({y_train.value_counts(normalize=True).get(0, 0)*100:.1f}%)")
print(f"    y=1 (ima pretplatu):  {y_train.value_counts().get(1, 0)} ({y_train.value_counts(normalize=True).get(1, 0)*100:.1f}%)")
print(f"  Balans klasa u test skupu:")
print(f"    y=0 (nema pretplatu): {y_test.value_counts().get(0, 0)} ({y_test.value_counts(normalize=True).get(0, 0)*100:.1f}%)")
print(f"    y=1 (ima pretplatu):  {y_test.value_counts().get(1, 0)} ({y_test.value_counts(normalize=True).get(1, 0)*100:.1f}%)")

# ============================================
# ČUVANJE PODATAKA
# ============================================
df_model.to_csv('data/bank-additional-preprocessed.csv', index=False)
print("\nPreprocesirani podaci sačuvani u data/bank-additional-preprocessed.csv")

# Čuvanje treniranih i test skupova
X_train.assign(y=y_train).to_csv('data/bank-additional-train.csv', index=False)
X_test.assign(y=y_test).to_csv('data/bank-additional-test.csv', index=False)
print("Trening skup sačuvan u data/bank-additional-train.csv")
print("Test skup sačuvan u data/bank-additional-test.csv")

print("\n========== PREPROCESSING ZAVRŠEN ==========")
print(f"""Rezime izmena nad podacima:
  - Proverene NaN vrednosti (nema ih)
  - Identifikovane i dokumentovane 'unknown' vrednosti
  - pdays=999 tretiran: kreiran binarni indikator, 999 zamenjeno sa -1
  - One-hot encoding: {nominal_cols}
  - Label encoding: education -> education_encoded
  - Uklonjen 'duration' (nedostupan u trenutku predikcije)
  - Makroekonomski indikatori zadržani uz obrazloženje
  - Outlieri uklonjeni IQR metodom ({total_outliers_removed} redova)
  - Train/test split 80/20 sa stratifikacijom
""")