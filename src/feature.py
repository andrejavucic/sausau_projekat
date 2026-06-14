import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.feature_selection import (
    SelectKBest, f_classif,
    RFE, RFECV,
    mutual_info_classif
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score, classification_report
)

import warnings
warnings.filterwarnings('ignore')

# ========== KREIRANJE FOLDERZA ==========
os.makedirs("analysis", exist_ok=True)
os.makedirs("analysis/figures", exist_ok=True)
sns.set_style("whitegrid")

# ========== 1. UČITAVANJE PODATAKA ==========
# Učitaj test podatke (samo da znamo shape za feature names)
X_test = np.load("data/processed/X_test_preprocessed.npy")
y_test = np.load("data/processed/y_test.npy")

# Učitaj originalne trening podatke (za shape ako zatreba)
X_train_orig = np.load("data/processed/X_train_preprocessed.npy")
y_train_orig = np.load("data/processed/y_train.npy")

X_val   = np.load("data/processed/X_val_preprocessed.npy")
y_val   = np.load("data/processed/y_val.npy")
 


# ========== 2. UČITAVANJE NAZIVA ATRIBUTA ==========
try:
    feature_names = np.load("data/processed/feature_names.npy", allow_pickle=True)
except FileNotFoundError:
    print("⚠️ Upozorenje: feature_names.npy ne postoji! Koristim generičke nazive.")
    feature_names = [f"Atribut_{i}" for i in range(X_train_orig.shape[1])]

# ========== 3. UČITAVANJE TRENIRANIH MODELA ==========
print("\n" + "="*60)
print("UČITAVANJE MODELA")
print("="*60)

models_dir = "models"
results = {}

# Lista modela koje očekujemo
model_names = [
    "Logistic Regression",
    "Random Forest"
]

for name in model_names:
    fname = f"{models_dir}/{name.lower().replace(' ', '_')}.pkl"
    if os.path.exists(fname):
        model = joblib.load(fname)
        results[name] = {"model": model}
        print(f"✅ Učitano: {name}")
    else:
        print(f"❌ Nije pronađeno: {fname}")

if len(results) == 0:
    raise FileNotFoundError("Nijedan model nije pronađen! Prvo pokreni train.py")


# ========== 7. FEATURE IMPORTANCE (interpretacija modela) ==========
# 7a. RANDOM FOREST - Feature Importance
if "Random Forest" in results:
    rf_model = results["Random Forest"]["model"]
    rf_importances = rf_model.feature_importances_
    
    # Sortiraj po važnosti
    rf_indices = np.argsort(rf_importances)[::-1]
    
    # Top 10 atributa
    # # DEFINISEMO PRAG !!! -> biramo top 10 -> selekcioni prag
    top_n = min(10, len(feature_names))
    rf_top_features = [feature_names[i] for i in rf_indices[:top_n]]
    rf_top_scores = [rf_importances[i] for i in rf_indices[:top_n]]
    
    # Plot za Random Forest
    plt.figure(figsize=(10, 6))
    plt.barh(rf_top_features[::-1], rf_top_scores[::-1], color='forestgreen', alpha=0.7)
    plt.xlabel("Važnost (Gini Importance)", fontsize=12)
    plt.ylabel("Atributi", fontsize=12)
    plt.title("Random Forest - Top 10 najvažnijih faktora za pretplatu", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("analysis/figures/rf_feature_importance.png", dpi=150)
    plt.close()
    
    print("\n📊 RANDOM FOREST - Najvažniji faktori:")
    for i, (feature, score) in enumerate(zip(rf_top_features, rf_top_scores), 1):
        print(f"   {i:2d}. {feature:<30} {score:.4f}")
    
    # Sačuvaj u CSV
    rf_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importances
    }).sort_values('Importance', ascending=False)
    rf_importance_df.to_csv("analysis/rf_feature_importance.csv", index=False)

# 7b. LOGISTIC REGRESION - Koeficijenti (apsolutna vrednost)
if "Logistic Regression" in results:
    lr_model = results["Logistic Regression"]["model"]
    lr_coefs = np.abs(lr_model.coef_[0])                    # APSOLUTNA VREDNOST !!!
    
    # Sortiraj po važnosti
    lr_indices = np.argsort(lr_coefs)[::-1]
    
    # Top 10 atributa
    top_n = min(10, len(feature_names))
    lr_top_features = [feature_names[i] for i in lr_indices[:top_n]]
    lr_top_scores = [lr_coefs[i] for i in lr_indices[:top_n]]
    
    # Plot za Logistic Regression
    plt.figure(figsize=(10, 6))
    plt.barh(lr_top_features[::-1], lr_top_scores[::-1], color='steelblue', alpha=0.7)
    plt.xlabel("Apsolutna vrednost koeficijenta", fontsize=12)
    plt.ylabel("Atributi", fontsize=12)
    plt.title("Logistic Regression - Top 10 najvažnijih faktora za pretplatu", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("analysis/figures/lr_feature_importance.png", dpi=150)
    plt.close()
    
    # apsolutni koeficijenti -> ampliuda, bez obzira na smer
    # moze biti pozitivno i negativno
    print("\n📊 LOGISTIC REGRESION - Najvažniji faktori (apsolutni koeficijenti):")
    for i, (feature, score) in enumerate(zip(lr_top_features, lr_top_scores), 1):
        # Dodaj i smer uticaja (+ ili -)
        original_coef = lr_model.coef_[0][lr_indices[i-1]]
        direction = "📈" if original_coef > 0 else "📉"
        print(f"   {i:2d}. {feature:<30} {score:.4f} {direction}")

# 7c. UPOREDNI PRIKAZ (ako imamo oba modela)
if "Random Forest" in results and "Logistic Regression" in results:
    # Normalizuj važnosti za poređenje
    rf_norm = rf_importances / rf_importances.max()
    lr_norm = lr_coefs / lr_coefs.max()
    
    # Uzmi top 10 iz RF (ili uniju top atributa)
    top_10_features_rf = [feature_names[i] for i in rf_indices[:10]]
    top_10_features_lr = [feature_names[i] for i in lr_indices[:10]]
    all_top_features = list(set(top_10_features_rf + top_10_features_lr))[:10]
    
    # Pripremi podatke za uporedni plot
    comparison_data = []
    for feat in all_top_features:
        idx = list(feature_names).index(feat)
        comparison_data.append({
            'Feature': feat,
            'Random Forest': rf_norm[idx],
            'Logistic Regression': lr_norm[idx]
        })
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df_melted = comp_df.melt(id_vars='Feature', var_name='Model', value_name='Normalized Importance')
    
    # Uporedni bar plot
    plt.figure(figsize=(12, 7))
    sns.barplot(data=comp_df_melted, x='Normalized Importance', y='Feature', hue='Model', palette='Set2')
    plt.xlabel("Normalizovana važnost (max=1)", fontsize=12)
    plt.ylabel("Atributi", fontsize=12)
    plt.title("Poređenje važnosti atributa: Random Forest vs Logistic Regression", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("analysis/figures/feature_importance_comparison.png", dpi=150)
    plt.close()
    
    print("\n📊 UPOREDNA ANALIZA - Ključni faktori po oba modela:")
    print("-" * 60)
    for i, row in comp_df.iterrows():
        print(f"   {row['Feature']:<30} RF: {row['Random Forest']:.3f} | LR: {row['Logistic Regression']:.3f}")

# 7d. Sačuvaj feature importance u TXT fajl
with open("analysis/feature_importance_analysis.txt", "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("ANALIZA NAJVAŽNIJIH FAKTORA ZA PRETPLATU\n")
    f.write("="*60 + "\n\n")
    
    if "Random Forest" in results:
        f.write("RANDOM FOREST - Top 10 faktora:\n")
        f.write("-"*40 + "\n")
        for i, (feature, score) in enumerate(zip(rf_top_features, rf_top_scores), 1):
            f.write(f"{i:2d}. {feature:<30} {score:.4f}\n")
        f.write("\n")
    
    if "Logistic Regression" in results:
        f.write("LOGISTIC REGRESION - Top 10 faktora:\n")
        f.write("-"*40 + "\n")
        for i, (feature, score) in enumerate(zip(lr_top_features, lr_top_scores), 1):
            original_coef = lr_model.coef_[0][lr_indices[i-1]]
            direction = "Pozitivan" if original_coef > 0 else "Negativan"
            f.write(f"{i:2d}. {feature:<30} {score:.4f} ({direction} uticaj)\n")
        f.write("\n")