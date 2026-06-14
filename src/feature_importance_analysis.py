import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score
)

# ============================================================
# KREIRANJE FOLDERA
# ============================================================
os.makedirs("analysis", exist_ok=True)
os.makedirs("analysis/figures", exist_ok=True)
sns.set_style("whitegrid")

# ============================================================
# 1. UČITAVANJE PODATAKA
# ============================================================
# SMOTE resampled trening podaci (isti na kojima su modeli trenirani)
X_train = np.load("data/processed/X_train_resampled.npy")
y_train = np.load("data/processed/y_train_resampled.npy")

X_val   = np.load("data/processed/X_val_preprocessed.npy")
y_val   = np.load("data/processed/y_val.npy")
X_test  = np.load("data/processed/X_test_preprocessed.npy")
y_test  = np.load("data/processed/y_test.npy")

try:
    feature_names = np.load("data/processed/feature_names.npy", allow_pickle=True)
    feature_names = list(feature_names)
except FileNotFoundError:
    feature_names = [f"Atribut_{i}" for i in range(X_train.shape[1])]

n_features = len(feature_names)

# ============================================================
# 2. UČITAVANJE SVIH 5 TRENIRANIH MODELA
# ============================================================
MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.pkl",
    # predugo mu treba
   # "KNN":                 "models/knn.pkl",
    "Decision Tree":       "models/decision_tree.pkl",
    "Random Forest":       "models/random_forest.pkl",
    "Gradient Boosting":   "models/gradient_boosting.pkl",
}

models = {}
for name, path in MODEL_FILES.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
        #print(f"✅ Učitan: {name}")
    else:
        print(f"❌ Nije pronađen: {path}")

if len(models) == 0:
    raise FileNotFoundError("Nijedan model nije pronađen! Prvo pokreni train.py.")

# ============================================================
# 3. IZVLAČENJE FEATURE IMPORTANCE ZA SVAKI MODEL
# ============================================================
importance_dict = {}  # {model_name: np.array(n_features,) normalizovano na [0,1]}

for model_name, model in models.items():
    print(f"\n  ▶ {model_name}...")

     # 👇 PRESKOČI KNN - ne treba ti za feature importance! 
     # predugo traje
    if model_name == "KNN":
        print("(preskačem - KNN nema feature importance)")
        importance_dict[model_name] = np.ones(X_train.shape[1])  # lažna vrednost
        continue

    if hasattr(model, 'feature_importances_'):
        # RF, DT, GB
        imp = model.feature_importances_
        print("(feature_importances_)")
    elif hasattr(model, 'coef_'):
        # Logistic Regression
        imp = np.abs(model.coef_[0])
        print("(apsolutni koeficijenti)")
    else:
        # KNN — Permutation Importance na validacionom skupu
        print("(Permutation Importance)...", end=" ")
        perm_res = permutation_importance(
            model, X_val, y_val,
            n_repeats=10, random_state=42, n_jobs=-1, scoring='roc_auc'
        )
        imp = np.clip(perm_res.importances_mean, 0, None)
        #print("✓")

    # Normalizuj na [0, 1]
    imp_max = imp.max()
    if imp_max > 0:
        imp_norm = imp / imp_max
    else:
        imp_norm = imp

    importance_dict[model_name] = imp_norm

# ============================================================
# 4. TABELA POREĐENJA TOP-15 ATRIBUTA PO MODELU
# ============================================================
TOP_N = 15

# Za svaki model: top-N atributi + skorovi
top_features_per_model = {}

for model_name, importances in importance_dict.items():
    indices = np.argsort(importances)[::-1][:TOP_N]
    top_feats = [(feature_names[i], importances[i]) for i in indices]
    top_features_per_model[model_name] = top_feats

# Kreiramo DataFrame gde su redovi atributi, kolone modeli
# Presek svih top atributa (unija top-15 iz svakog modela)
all_top_features = set()
for top_list in top_features_per_model.values():
    for feat, _ in top_list:
        all_top_features.add(feat)
all_top_features = sorted(all_top_features, key=lambda f: max(
    importance_dict[m][feature_names.index(f)] if f in feature_names else 0
    for m in importance_dict
), reverse=True)

comparison_data = []
for feat in all_top_features:
    idx = feature_names.index(feat)
    row = {'Feature': feat}
    for m_name in importance_dict:
        row[m_name] = round(importance_dict[m_name][idx], 4)
    # Prosečan skor (za sortiranje)
    row['Prosek'] = round(np.mean([row[m] for m in importance_dict]), 4)
    comparison_data.append(row)

comp_df = pd.DataFrame(comparison_data)
comp_df = comp_df.sort_values('Prosek', ascending=False).reset_index(drop=True)
comp_df['Rank'] = comp_df.index + 1

# Preuredi kolone: Rank, Feature, pa svi modeli, pa Prosek
model_cols = list(importance_dict.keys())
comp_df = comp_df[['Rank', 'Feature', 'Prosek'] + model_cols]

#print(f"\nUnija top atributa iz svih modela ({len(comp_df)} atributa):\n")
#print(comp_df.to_string(index=False))

#comp_df.to_csv("analysis/feature_importance_all_models.csv", index=False)

# ============================================================
# 5. VIZUALIZACIJA — HEATMAP (svi modeli × top atributi)
# ============================================================
# Uzimamo uniju top atributa (ograničenu na ~20-25 za čitljivost)
top_n_heat = min(25, len(all_top_features))
heat_features = comp_df.head(top_n_heat)['Feature'].tolist()

heat_data = comp_df.head(top_n_heat).set_index('Feature')[model_cols]

plt.figure(figsize=(12, max(8, top_n_heat * 0.35)))
sns.heatmap(
    heat_data,
    annot=True, fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={'label': 'Normalizovana važnost (0–1)'}
)
plt.title(f"Poređenje važnosti atributa — Top {top_n_heat} po svim modelima", fontsize=14, fontweight='bold')
plt.xlabel("Model", fontsize=12)
plt.ylabel("Atribut", fontsize=12)
plt.tight_layout()
plt.savefig("analysis/figures/feature_importance_heatmap_all_models.png", dpi=150)
plt.close()

# ============================================================
# 7. POREĐENJE PERFORMANSI: SVI vs. SELEKTOVANI (RF + LR)
# ============================================================
print("\n" + "=" * 60)
print("POREĐENJE: SVI ATRIBUTI vs. TOP-N — Random Forest + Logistic Regression")
print("=" * 60)

# Biramo 2 modela za poređenje: RF i LR
COMPARE_MODELS = ["Random Forest", "Logistic Regression"]

def get_model_params(model_name, loaded_model):
    """Vraća rečnik hiperparametara za kreiranje svežeg modela."""
    if model_name == "Random Forest":
        return {
            "n_estimators": loaded_model.n_estimators,
            "max_depth": loaded_model.max_depth,
            "max_features": getattr(loaded_model, 'max_features', 'sqrt'),
            "min_samples_split": loaded_model.min_samples_split,
            "min_samples_leaf": loaded_model.min_samples_leaf,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
    elif model_name == "Logistic Regression":
        return {
            "C": getattr(loaded_model, 'C', 1.0),
            "penalty": getattr(loaded_model, 'penalty', 'l2'),
            "solver": getattr(loaded_model, 'solver', 'lbfgs'),
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42
        }

def evaluate_model(model, X_tr, y_tr, X_te, y_te, label, n_feats):
    """Trenira model i vraća metrike na test skupu."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    return {
        'Varijanta': label,
        'Br_atributa': n_feats,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1':        round(f1_score(y_test, y_pred), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
    }

comparison_rows = []

for model_name in COMPARE_MODELS:
    if model_name not in models:
        print(f"  ⚠️ {model_name} nije dostupan — preskačem.")
        continue

    loaded_model = models[model_name]
    params = get_model_params(model_name, loaded_model)
    importances = importance_dict[model_name]

    # Indeksi svih atributa
    idx_all = list(range(n_features))

    # Top-N atributi po modelu (N = 5, 10, 15, 20)
    sorted_idx = np.argsort(importances)[::-1]
    for n_sub in [5, 10, 15, 20]:
        idx_top = sorted_idx[:n_sub].tolist()

        # Kreiraj svež model
        if model_name == "Random Forest":
            m = RandomForestClassifier(**params)
        else:
            m = LogisticRegression(**params)

        row = evaluate_model(
            m,
            X_train[:, idx_top], y_train,
            X_test[:, idx_top], y_test,
            label=f"{model_name} | Top {n_sub}",
            n_feats=n_sub
        )
        row['Model'] = model_name
        row['Podskup'] = f"Top {n_sub}"
        comparison_rows.append(row)
        print(f"    ✓ {model_name} | Top {n_sub:<3}  "
              f"F1={row['F1']:.4f}  AUC={row['ROC-AUC']:.4f}  Recall={row['Recall']:.4f}")

    # Varijanta sa SVIM atributima
    if model_name == "Random Forest":
        m_all = RandomForestClassifier(**params)
    else:
        m_all = LogisticRegression(**params)

    row_all = evaluate_model(
        m_all,
        X_train[:, idx_all], y_train,
        X_test[:, idx_all], y_test,
        label=f"{model_name} | Svi atributi",
        n_feats=n_features
    )
    row_all['Model'] = model_name
    row_all['Podskup'] = "Svi atributi"
    comparison_rows.append(row_all)
    print(f"    ✓ {model_name} | Svi ({n_features})  "
          f"F1={row_all['F1']:.4f}  AUC={row_all['ROC-AUC']:.4f}  Recall={row_all['Recall']:.4f}")

# ============================================================
# 8. ČUVANJE I VIZUALIZACIJA POREĐENJA
# ============================================================
comp_perf_df = pd.DataFrame(comparison_rows)
comp_perf_df.to_csv("analysis/feature_selection_performance.csv", index=False)

# Vizualizacija — za oba modela na istom plotu
metrics_to_plot = ['F1', 'ROC-AUC', 'Recall', 'Precision']
model_colors = {'Random Forest': 'forestgreen', 'Logistic Regression': 'steelblue'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Poređenje performansi: Svi atributi vs. Top-N (RF i LR)",
             fontsize=14, fontweight='bold', y=1.01)

for ax, metric in zip(axes.flatten(), metrics_to_plot):
    for model_name, color in model_colors.items():
        subset = comp_perf_df[comp_perf_df['Model'] == model_name]
        x_labels = subset['Podskup'].tolist()
        n_bars = len(x_labels)

        x_pos = [i + (0.2 if model_name == 'Random Forest' else -0.2) for i in range(n_bars)]
        bars = ax.bar(x_pos, subset[metric], width=0.35, color=color, alpha=0.8, label=model_name)

        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig("analysis/figures/feature_selection_performance.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 9. TXT IZVEŠTAJ
# ============================================================
with open("analysis/feature_importance_report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("IZVEŠTAJ: FEATURE IMPORTANCE I SELEKCIJA ATRIBUTA\n")
    f.write("=" * 70 + "\n\n")

    # Opis metoda
    f.write("MODELI I METODE ZA FEATURE IMPORTANCE:\n")
    f.write("-" * 40 + "\n")
    f.write("  • Logistic Regression  → apsolutni koeficijenti (|β|)\n")
    f.write("  • KNN                    → Permutation Importance (ROC-AUC, val skup)\n")
    f.write("  • Decision Tree          → Gini Importance (feature_importances_)\n")
    f.write("  • Random Forest          → Gini Importance (feature_importances_)\n")
    f.write("  • Gradient Boosting      → Gini Importance (feature_importances_)\n\n")

    # Top-15 po svakom modelu
    for model_name, top_feats in top_features_per_model.items():
        f.write(f"\nTOP 15 — {model_name.upper()}:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Rank':>4}  {'Atribut':<40} {'Važnost':>8}\n")
        f.write("-" * 50 + "\n")
        for i, (feat, score) in enumerate(top_feats, 1):
            f.write(f"  {i:>2}.  {feat:<40} {score:>8.4f}\n")

    # Poređenje performansi
    f.write("\n\n" + "=" * 70 + "\n")
    f.write("POREĐENJE PERFORMANSI: SVI vs. SELEKTOVANI ATRIBUTI\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"{'Varijanta':<45} {'F1':>7} {'AUC':>7} {'Recall':>8} {'Prec.':>7}\n")
    f.write("-" * 70 + "\n")
    for _, row in comp_perf_df.iterrows():
        f.write(f"  {row['Varijanta']:<43} "
                f"{row['F1']:>7.4f} {row['ROC-AUC']:>7.4f} "
                f"{row['Recall']:>8.4f} {row['Precision']:>7.4f}\n")

    # Zaključak po modelu
    f.write("\n\nZAKLJUČAK:\n")
    f.write("-" * 40 + "\n")
    for model_name in COMPARE_MODELS:
        m_df = comp_perf_df[comp_perf_df['Model'] == model_name]
        all_row = m_df[m_df['Podskup'] == 'Svi atributi']
        if len(all_row) == 0:
            continue
        all_f1 = all_row['F1'].values[0]
        best_row = m_df.loc[m_df['F1'].idxmax()]
        best_f1 = best_row['F1']
        diff = best_f1 - all_f1
        sign = '↑' if diff >= 0 else '↓'
        f.write(f"  {model_name}:\n")
        f.write(f"    Svi atributi ({all_row['Br_atributa'].values[0]}): F1 = {all_f1:.4f}\n")
        f.write(f"    Najbolji podskup: '{best_row['Podskup']}' ({best_row['Br_atributa']} atr.) "
                f"→ F1 = {best_f1:.4f} ({sign}{abs(diff):.4f})\n\n")

# ============================================================
# 10. KONAČAN PREGLED U TERMINALU
# ============================================================
print()
print("🏆 Najbolja varijanta po F1:")
best_overall = comp_perf_df.loc[comp_perf_df['F1'].idxmax()]
print(f"   {best_overall['Varijanta']} — F1={best_overall['F1']:.4f}, AUC={best_overall['ROC-AUC']:.4f}, Recall={best_overall['Recall']:.4f}")
print()