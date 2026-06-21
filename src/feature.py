import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score, fbeta_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# ========== KREIRANJE FOLDERZA ==========
os.makedirs("analysis", exist_ok=True)
os.makedirs("analysis/figures", exist_ok=True)
sns.set_style("whitegrid")

# ========== 1. UČITAVANJE PODATAKA ==========
X_test = np.load("data/processed/X_test_preprocessed.npy")
y_test = np.load("data/processed/y_test.npy")

# ========== 2. UČITAVANJE NAZIVA ATRIBUTA ==========
print()
try:
    feature_names = np.load("data/processed/feature_names.npy", allow_pickle=True)
    feature_names = list(feature_names)
    print(f"✅ Učitano {len(feature_names)} naziva atributa")
except FileNotFoundError:
    print("⚠️ feature_names.npy ne postoji! Koristim generičke nazive.")

n_features = len(feature_names)
print(f"\n📊 Ukupan broj atributa: {n_features}")

# ========== 3. UČITAVANJE MODELA ==========
models_dir = "models"
models = {}

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}

print()
for name, fname in model_files.items():
    path = os.path.join(models_dir, fname)
    if os.path.exists(path):
        model = joblib.load(path)
        models[name] = model
        print(f"✅ Učitano: {name}")
    else:
        print(f"❌ Nije pronađeno: {path}")

if len(models) == 0:
    raise FileNotFoundError("Nijedan model nije pronađen!")

# ========== 4. IZVLAČENJE FEATURE IMPORTANCE ==========
importance_dict = {}
model_estimators = {}  # Čuvamo referencu na stvarni estimator (za coef_)

for name, model in models.items():
    #print(f"\n  ▶ {name}...")
    
    imp = None
    estimator = None  # Stvarni estimator (može biti isti kao model ili last_step)
    
    # 1. Direktno iz modela
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        estimator = model
        #print(f"   ✓ feature_importances_ (direktno iz modela)")
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) == 2:
            imp = np.abs(model.coef_[0])
        else:
            imp = np.abs(model.coef_)
        estimator = model
        #print(f"   ✓ coef_ (direktno iz modela)")
    
    # 2. Ako je pipeline, izvuci iz poslednjeg koraka
    if imp is None and hasattr(model, 'named_steps'):
        #print(f"   🔍 Model je Pipeline, proveravam poslednji korak...")
        last_step = list(model.named_steps.values())[-1]
        estimator = last_step
        
        if hasattr(last_step, 'feature_importances_'):
            imp = last_step.feature_importances_
            #print(f"   ✓ feature_importances_ iz pipeline-a")
        elif hasattr(last_step, 'coef_'):
            if len(last_step.coef_.shape) == 2:
                imp = np.abs(last_step.coef_[0])
            else:
                imp = np.abs(last_step.coef_)
            #print(f"   ✓ coef_ iz pipeline-a")
    
    if imp is None:
        print(f"   ❌ NEMA FEATURE IMPORTANCE za {name}")
        continue
    
    # ASSERT: Provera dimenzija
    try:
        assert len(imp) == n_features, \
            f"Mismatch: {len(imp)} importance vs {n_features} feature names"
    except AssertionError as e:
        print(f"   ⚠️ {e}")
        if len(imp) < n_features:
            imp = np.pad(imp, (0, n_features - len(imp)))
            print(f"   → Dopunjeno sa nulama na {len(imp)}")
        else:
            imp = imp[:n_features]
            print(f"   → Skraćeno na {len(imp)}")
    
    # NORMALIZACIJA: suma = 1 (za heatmap poređenje)
    imp_sum = imp.sum()
    if imp_sum > 0:
        imp_norm_sum = imp / imp_sum  # relativna važnost (suma = 1)
    else:
        imp_norm_sum = imp
    
    # Sačuvaj obe verzije
    importance_dict[name] = {
        'sum_norm': imp_norm_sum,  # za heatmap (suma = 1)
        'raw': imp,                # originalne vrednosti
        'max_norm': imp / imp.max() if imp.max() > 0 else imp  # za bar plotove
    }
    model_estimators[name] = estimator
    #print(f"   ✅ Feature importance sačuvan (suma = {imp_norm_sum.sum():.2f})")

if len(importance_dict) == 0:
    print("\n❌ NIJEDAN MODEL NEMA FEATURE IMPORTANCE!")
    exit()

# ========== 5. VIZUALIZACIJA - Top 10 atributa ==========
print("\n" + "="*60)
print("VIZUALIZACIJA NAJVAŽNIJIH ATRIBUTA")
print("="*60)

TOP_N = 10
colors = {
    "Logistic Regression": "steelblue",
    "Random Forest": "forestgreen",
    "Gradient Boosting": "purple"
}

for model_name, imp_dict in importance_dict.items():
    # Koristimo max_norm za bar plot (svaki model ima max=1)
    imp_norm = imp_dict['max_norm']
    indices = np.argsort(imp_norm)[::-1][:TOP_N]
    top_features = [feature_names[i] for i in indices]
    top_scores = [imp_norm[i] for i in indices]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_scores[::-1], 
             color=colors.get(model_name, "gray"), alpha=0.7)
    plt.xlabel("Normalizovana važnost (max=1)", fontsize=12)
    plt.ylabel("Atributi", fontsize=12)
    plt.title(f"{model_name} - Top {TOP_N} najvažnijih faktora", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fname = model_name.lower().replace(' ', '_')
    plt.savefig(f"analysis/figures/{fname}_feature_importance.png", dpi=150)
    plt.close()
    
    print(f"\n📊 {model_name} - Top {TOP_N} atributa:")
    for i, (feature, score) in enumerate(zip(top_features, top_scores), 1):
        # ISPRAVKA: Koristimo sačuvani estimator za coef_
        if model_name == "Logistic Regression":
            estimator = model_estimators[model_name]
            if hasattr(estimator, 'coef_'):
                # indices[i-1] je stvarni indeks atributa
                if len(estimator.coef_.shape) == 2:
                    orig_coef = estimator.coef_[0][indices[i-1]]
                else:
                    orig_coef = estimator.coef_[indices[i-1]]
                print(f"   {i:2d}. {feature:<30} {score:.4f}")
            else:
                print(f"   {i:2d}. {feature:<30} {score:.4f}")
        else:
            print(f"   {i:2d}. {feature:<30} {score:.4f}")
    
    # Sačuvaj CSV
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': imp_norm,
        'Importance_sum': imp_dict['sum_norm']
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv(f"analysis/{fname}_feature_importance.csv", index=False)

# ========== 6. HEATMAP (koristi sum_norm) ==========
TOP_N_HEAT = 20
all_top_features = set()

for model_name, imp_dict in importance_dict.items():
    # Koristimo sum_norm za heatmap
    imp_norm = imp_dict['sum_norm']
    indices = np.argsort(imp_norm)[::-1][:TOP_N_HEAT]
    for i in indices:
        if i < len(feature_names):
            all_top_features.add(feature_names[i])

if len(all_top_features) == 0:
    print("   ⚠️ Nema atributa za heatmap, koristim prvih 20")
    all_top_features = set(feature_names[:TOP_N_HEAT])

all_top_features = sorted(all_top_features)
print(f"   Pronađeno {len(all_top_features)} atributa za heatmap")

heat_data = []
for feat in all_top_features:
    try:
        idx = feature_names.index(feat)
        row = {'Feature': feat}
        for model_name, imp_dict in importance_dict.items():
            row[model_name] = round(imp_dict['sum_norm'][idx], 4)
        heat_data.append(row)
    except ValueError:
        continue

if len(heat_data) > 0:
    heat_df = pd.DataFrame(heat_data)
    heat_df = heat_df.set_index('Feature')
    
    # Sortiraj po proseku
    heat_df['Prosek'] = heat_df.mean(axis=1)
    heat_df = heat_df.sort_values('Prosek', ascending=False)
    heat_df = heat_df.drop('Prosek', axis=1)
    
    plt.figure(figsize=(10, max(8, len(heat_df) * 0.35)))
    sns.heatmap(
        heat_df,
        annot=True, fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={'label': 'Relativna važnost (suma=1)'}
    )
    plt.title(f"Poređenje važnosti atributa — Top {len(heat_df)} (suma=1)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Atribut", fontsize=12)
    plt.tight_layout()
    plt.savefig("analysis/figures/feature_importance_heatmap_all_models.png", dpi=150)
    plt.close()

# ========== 7. TESTIRANJE: TOP 10 vs TOP 20 vs SVI ==========
print("\n" + "="*60)
print("TESTIRANJE PERFORMANSI (F2 i RECALL)")
print("="*60)

# Učitaj ORIGINALNE trening podatke (bez SMOTE-a)
X_train_pp = np.load("data/processed/X_train_preprocessed.npy")
y_train_pp = np.load("data/processed/y_train.npy")

TOP_OPTIONS = [10, 20]
comparison_results = []

# Čuvamo redosled modela za konzistentnost
model_order = []

for model_name, model in models.items():
    if model_name not in importance_dict:
        continue
    
    model_order.append(model_name)
    imp_dict = importance_dict[model_name]
    imp_norm = imp_dict['max_norm']
    
    print(f"\n🔍 {model_name}:")
    
    # Parametri ZA NOVI MODEL (isti kao u train.py)
    if model_name == "Logistic Regression":
        params = {
            "C": 0.01,
            "penalty": "l1",
            "solver": "saga",
            "max_iter": 2000,
            "random_state": 42
        }
        # KORISTI IMBPIPELINE (SMOTE UNUTAR PIPELINE-A)
        base_model = LogisticRegression(**params)
        
    elif model_name == "Random Forest":
        params = {
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1
        }
        base_model = RandomForestClassifier(**params)
        
    elif model_name == "Gradient Boosting":
        params = {
            "n_estimators": 100,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "subsample": 0.8,
            "random_state": 42
        }
        base_model = GradientBoostingClassifier(**params)
    
    # Testiraj Top N (KORISTI IMBPIPELINE ZA SVAKI SLUČAJ)
    for top_k in TOP_OPTIONS:
        top_k_indices = np.argsort(imp_norm)[::-1][:top_k]
        
        # Pipeline sa SMOTE (isto kao u train.py)
        pipe_top = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', base_model.__class__(**params))  # Novi model za svaki top_k
        ])
        
        # Treniraj na ORIGINALNIM podacima (bez SMOTE-a)
        pipe_top.fit(X_train_pp[:, top_k_indices], y_train_pp)
        
        # Predikcija na test skupu
        y_pred_top = pipe_top.predict(X_test[:, top_k_indices])
        y_prob_top = pipe_top.predict_proba(X_test[:, top_k_indices])[:, 1]
        
        metrics_top = {
            'Model': model_name,
            'Varijanta': f'Top {top_k}',
            'Br_atributa': top_k,
            'Accuracy': round(accuracy_score(y_test, y_pred_top), 4),
            'Precision': round(precision_score(y_test, y_pred_top), 4),
            'Recall': round(recall_score(y_test, y_pred_top), 4),
            'F2': round(fbeta_score(y_test, y_pred_top, beta=2), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob_top), 4)
        }
        comparison_results.append(metrics_top)
        print(f"   Top {top_k:2d}: F2={metrics_top['F2']:.4f}, Recall={metrics_top['Recall']:.4f}")
    
    # Testiraj SVE (opet sa ImbPipeline)
    pipe_all = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', base_model.__class__(**params))
    ])
    pipe_all.fit(X_train_pp, y_train_pp) 
    
    y_pred_all = pipe_all.predict(X_test)
    y_prob_all = pipe_all.predict_proba(X_test)[:, 1]
    
    metrics_all = {
        'Model': model_name,
        'Varijanta': 'Svi atributi',
        'Br_atributa': n_features,
        'Accuracy': round(accuracy_score(y_test, y_pred_all), 4),
        'Precision': round(precision_score(y_test, y_pred_all), 4),
        'Recall': round(recall_score(y_test, y_pred_all), 4),
        'F2': round(fbeta_score(y_test, y_pred_all, beta=2), 4),
        'ROC-AUC': round(roc_auc_score(y_test, y_prob_all), 4)
    }
    comparison_results.append(metrics_all)
    print(f"   Svi ({n_features}): F2={metrics_all['F2']:.4f}, Recall={metrics_all['Recall']:.4f}")

# ========== 8. VIZUALIZACIJA REZULTATA ==========
if len(comparison_results) > 0:
    comp_df = pd.DataFrame(comparison_results)
    
    # Bar plot - F2 i Recall
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Koristimo model_order za konzistentan redosled
    x = np.arange(len(model_order))
    width = 0.25
    varijante = ['Top 10', 'Top 20', 'Svi atributi']
    
    for idx, metric in enumerate(['F2', 'Recall']):
        ax = axes[idx]
        
        for i, varijanta in enumerate(varijante):
            # Filtriraj za svaki model posebno
            metric_values = []
            for model_name in model_order:
                subset = comp_df[(comp_df['Model'] == model_name) & (comp_df['Varijanta'] == varijanta)]
                if len(subset) > 0:
                    metric_values.append(subset[metric].values[0])
                else:
                    metric_values.append(0)
            
            bars = ax.bar(x + i*width, metric_values, width, label=varijanta, alpha=0.8)
            for bar, val in zip(bars, metric_values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Poređenje {metric} skora', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_order, rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig("analysis/figures/top10_vs_top20_vs_all_performance.png", dpi=150)
    plt.close()
    
    # Sačuvaj CSV
    comp_df.to_csv("analysis/top10_vs_top20_vs_all_performance.csv", index=False)
    
    # Ispiši najbolje po F2
    print("\n🏆 NAJBOLJE PO F2 SKORU:")
    best_f2 = comp_df.loc[comp_df['F2'].idxmax()]
    print(f"   {best_f2['Model']} - {best_f2['Varijanta']}: F2={best_f2['F2']:.4f}, Recall={best_f2['Recall']:.4f}")
    
    # Ispiši najbolje po Recall
    print("\n🏆 NAJBOLJE PO RECALL:")
    best_recall = comp_df.loc[comp_df['Recall'].idxmax()]
    print(f"   {best_recall['Model']} - {best_recall['Varijanta']}: Recall={best_recall['Recall']:.4f}, F2={best_recall['F2']:.4f}")
    
    # Tabelarni prikaz
    print("\n📊 TABELARNI PRIKAZ SVIH REZULTATA:")
    print("-" * 90)
    print(f"{'Model':<22} {'Varijanta':<15} {'F2':>8} {'Recall':>8} {'Prec.':>8} {'Acc.':>8} {'#Atr.':>8}")
    print("-" * 90)
    for _, row in comp_df.iterrows():
        print(f"{row['Model']:<22} {row['Varijanta']:<15} {row['F2']:>8.4f} {row['Recall']:>8.4f} {row['Precision']:>8.4f} {row['Accuracy']:>8.4f} {row['Br_atributa']:>8}")

print()