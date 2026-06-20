import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              precision_recall_curve, fbeta_score)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import RocCurveDisplay

import warnings
warnings.filterwarnings('ignore')

# ========== KREIRANJE FOLDERZA ==========
os.makedirs("analysis", exist_ok=True)
os.makedirs("analysis/figures", exist_ok=True)
sns.set_style("whitegrid")

# ========== 1. UČITAVANJE PODATAKA ==========
# Test podaci (isti za sve)
X_test = np.load("data/processed/X_test_preprocessed.npy")
y_test = np.load("data/processed/y_test.npy")

# Originalni trening podaci (bez SMOTE-a) - za dijagnostiku
X_train_orig = np.load("data/processed/X_train_preprocessed.npy")
y_train_orig = np.load("data/processed/y_train.npy")

# ========== 2. UČITAVANJE TRENIRANIH MODELA ==========
models_dir = "models"
results = {}

# Lista modela koje očekujemo
model_names = [
    "Logistic Regression",
    "KNN", 
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting"
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

# ========== 3. EVALUACIJA SVIH MODELA ==========
print("\n" + "="*60)
print("EVALUACIJA MODELA NA TEST SKUPU")
print("="*60)

for name, r in results.items():
    model = r["model"]
    
    # Predikcije
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrike (DODAT F2!)
    r["accuracy"] = accuracy_score(y_test, y_pred)
    r["precision"] = precision_score(y_test, y_pred)
    r["recall"] = recall_score(y_test, y_pred)
    r["f1"] = f1_score(y_test, y_pred)
    r["f2"] = fbeta_score(y_test, y_pred, beta=2)  # <<< DODATO
    r["roc_auc"] = roc_auc_score(y_test, y_prob)
    r["cm"] = confusion_matrix(y_test, y_pred)
    r["y_prob"] = y_prob  # sačuvaj za kasnije
    
    print(f"\n📊 {name}")
    print(f"   Accuracy:  {r['accuracy']:.4f}")
    print(f"   Precision: {r['precision']:.4f}")
    print(f"   Recall:    {r['recall']:.4f}")
    print(f"   F1:        {r['f1']:.4f}")
    print(f"   F2:        {r['f2']:.4f}")  # <<< DODATO
    print(f"   ROC-AUC:   {r['roc_auc']:.4f}")

# ========== 4. RANGIRANJE PO F2 (kao u train.py) ==========
print("\n" + "="*60)
print("🏆 RANG LISTA PO F2 SKORU (default prag 0.5)")
print("="*60)

# Sortiraj po F2 (opadajuće)
sorted_by_f2 = sorted(results.items(), key=lambda x: x[1]['f2'], reverse=True)

for i, (name, r) in enumerate(sorted_by_f2, 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"  {i}."
    print(f"{medal} {name}:")
    print(f"   F2 = {r['f2']:.4f}")
    print(f"   Recall = {r['recall']:.4f} ({r['recall']*100:.1f}%)")
    print(f"   Precision = {r['precision']:.4f}")
    print(f"   F1 = {r['f1']:.4f}")
    print()

# Sačuvaj rang listu u CSV
ranking_df = pd.DataFrame([
    {
        "Rank": i,
        "Model": name,
        "F2": round(r["f2"], 4),
        "Recall": round(r["recall"], 4),
        "Precision": round(r["precision"], 4),
        "F1": round(r["f1"], 4),
        "Accuracy": round(r["accuracy"], 4),
        "ROC_AUC": round(r["roc_auc"], 4)
    }
    for i, (name, r) in enumerate(sorted_by_f2, 1)
])
ranking_df.to_csv("analysis/ranking_by_f2_analysis.csv", index=False)
print("✅ Rang lista po F2 sačuvana u: analysis/ranking_by_f2_analysis.csv")

# ========== 5. POREĐENJE SA BASELINE MODELOM ==========
print("\n" + "="*60)
print("📊 POREĐENJE SA BASELINE MODELOM")
print("="*60)

# 5a. Stratified baseline (nasumično po distribuciji iz treninga)
stratified_baseline = DummyClassifier(strategy='stratified', random_state=42)
stratified_baseline.fit(X_train_orig, y_train_orig)

# Predikcije
y_pred_stratified = stratified_baseline.predict(X_test)
y_prob_stratified = stratified_baseline.predict_proba(X_test)[:, 1]

# Metrike za baseline
baseline_f1 = f1_score(y_test, y_pred_stratified)
baseline_f2 = fbeta_score(y_test, y_pred_stratified, beta=2)  # <<< DODATO
baseline_recall = recall_score(y_test, y_pred_stratified)
baseline_precision = precision_score(y_test, y_pred_stratified)
baseline_accuracy = accuracy_score(y_test, y_pred_stratified)
baseline_roc_auc = roc_auc_score(y_test, y_prob_stratified)
baseline_da_pct = y_pred_stratified.mean() * 100

print(f"\n📌 Stratified baseline (nasumično predviđanje):")
print(f"   Accuracy:  {baseline_accuracy:.4f}")
print(f"   Precision: {baseline_precision:.4f}")
print(f"   Recall:    {baseline_recall:.4f} ({baseline_recall*100:.1f}%)")
print(f"   F1:        {baseline_f1:.4f}")
print(f"   F2:        {baseline_f2:.4f}")  # <<< DODATO
print(f"   ROC-AUC:   {baseline_roc_auc:.4f}")
print(f"   Predviđeno DA: {baseline_da_pct:.1f}%")

# 5b. Poređenje sa Logistic Regression (ako postoji)
print("\n" + "-"*60)
print("🔍 POREĐENJE SA LOGISTIČKOM REGRESIJOM")
print("-"*60)

if "Logistic Regression" in results:
    lr = results["Logistic Regression"]
    lr_f1 = lr["f1"]
    lr_f2 = lr["f2"]
    lr_recall = lr["recall"]
    lr_precision = lr["precision"]
    lr_accuracy = lr["accuracy"]
    lr_roc_auc = lr["roc_auc"]
    
    print(f"\n📌 Logistic Regression:")
    print(f"   Accuracy:  {lr_accuracy:.4f}")
    print(f"   Precision: {lr_precision:.4f}")
    print(f"   Recall:    {lr_recall:.4f} ({lr_recall*100:.1f}%)")
    print(f"   F1:        {lr_f1:.4f}")
    print(f"   F2:        {lr_f2:.4f}")
    print(f"   ROC-AUC:   {lr_roc_auc:.4f}")
    
    print(f"\n📊 POREĐENJE (Logistic Regression vs Baseline):")
    print(f"   {'Metrika':<12} {'Baseline':>10} {'Logistic':>10} {'Razlika':>10}")
    print(f"   {'-'*42}")
    print(f"   {'Accuracy':<12} {baseline_accuracy:>10.4f} {lr_accuracy:>10.4f} {lr_accuracy - baseline_accuracy:>+10.4f}")
    print(f"   {'Precision':<12} {baseline_precision:>10.4f} {lr_precision:>10.4f} {lr_precision - baseline_precision:>+10.4f}")
    print(f"   {'Recall':<12} {baseline_recall:>10.4f} {lr_recall:>10.4f} {lr_recall - baseline_recall:>+10.4f}")
    print(f"   {'F1':<12} {baseline_f1:>10.4f} {lr_f1:>10.4f} {lr_f1 - baseline_f1:>+10.4f}")
    print(f"   {'F2':<12} {baseline_f2:>10.4f} {lr_f2:>10.4f} {lr_f2 - baseline_f2:>+10.4f}")
    print(f"   {'ROC-AUC':<12} {baseline_roc_auc:>10.4f} {lr_roc_auc:>10.4f} {lr_roc_auc - baseline_roc_auc:>+10.4f}")
    
    # Interpretacija
    if lr_f2 > baseline_f2:
        improvement = (lr_f2 - baseline_f2) / baseline_f2 * 100 if baseline_f2 > 0 else float('inf')
        print(f"\n✅ Logistic Regression je BOLJI od baseline modela")
        if baseline_f2 > 0:
            print(f"   Poboljšanje od +{improvement:.1f}% u F2 skoru")
    elif lr_f2 < baseline_f2:
        print(f"\n⚠️ Baseline je BOLJI od Logistic Regression-a!")
    else:
        print(f"\n📊 Modeli su JEDNAKI")
else:
    print("\n⚠️ Logistic Regression nije pronađen u results dictionary-ju")

# 5c. Poređenje sa najboljim modelom
print("\n" + "-"*60)
print("🏆 POREĐENJE SA NAJBOLJIM MODELOM")
print("-"*60)

best_model_name, best_model_data = sorted_by_f2[0]
best_f2 = best_model_data["f2"]
best_recall = best_model_data["recall"]
best_precision = best_model_data["precision"]
best_f1 = best_model_data["f1"]
best_accuracy = best_model_data["accuracy"]
best_roc_auc = best_model_data["roc_auc"]

print(f"\n📌 Najbolji model: {best_model_name}")
print(f"   Accuracy:  {best_accuracy:.4f}")
print(f"   Precision: {best_precision:.4f}")
print(f"   Recall:    {best_recall:.4f} ({best_recall*100:.1f}%)")
print(f"   F1:        {best_f1:.4f}")
print(f"   F2:        {best_f2:.4f}")
print(f"   ROC-AUC:   {best_roc_auc:.4f}")

print(f"\n📊 POREĐENJE (Najbolji model vs Baseline):")
print(f"   {'Metrika':<12} {'Baseline':>10} {'Najbolji':>10} {'Razlika':>10}")
print(f"   {'-'*42}")
print(f"   {'Accuracy':<12} {baseline_accuracy:>10.4f} {best_accuracy:>10.4f} {best_accuracy - baseline_accuracy:>+10.4f}")
print(f"   {'Precision':<12} {baseline_precision:>10.4f} {best_precision:>10.4f} {best_precision - baseline_precision:>+10.4f}")
print(f"   {'Recall':<12} {baseline_recall:>10.4f} {best_recall:>10.4f} {best_recall - baseline_recall:>+10.4f}")
print(f"   {'F1':<12} {baseline_f1:>10.4f} {best_f1:>10.4f} {best_f1 - baseline_f1:>+10.4f}")
print(f"   {'F2':<12} {baseline_f2:>10.4f} {best_f2:>10.4f} {best_f2 - baseline_f2:>+10.4f}")
print(f"   {'ROC-AUC':<12} {baseline_roc_auc:>10.4f} {best_roc_auc:>10.4f} {best_roc_auc - baseline_roc_auc:>+10.4f}")

if best_f2 > baseline_f2:
    improvement = (best_f2 - baseline_f2) / baseline_f2 * 100 if baseline_f2 > 0 else float('inf')
    print(f"\n✅ {best_model_name} je BOLJI od baseline modela")
    if baseline_f2 > 0:
        print(f"   Poboljšanje od +{improvement:.1f}% u F2 skoru")
elif best_f2 < baseline_f2:
    print(f"\n⚠️ Baseline je BOLJI od {best_model_name}!")
else:
    print(f"\n📊 Modeli su JEDNAKI")

# ========== 6. Vizuelna interpretacija ==========
# --- 6a. ROC krive za sve modele na jednom plotu ---
plt.figure(figsize=(10, 8))
ax = plt.gca()

for name, r in results.items():
    RocCurveDisplay.from_estimator(r["model"], X_test, y_test, 
                                   name=name, ax=ax)

# Dodaj dijagonalu (nasumičan model)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Nasumičan model (AUC=0.5)')

plt.title('ROC krive za sve modele', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specifičnost)', fontsize=12)
plt.ylabel('True Positive Rate (Recall / Osetljivost)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("analysis/figures/roc_curves.png", dpi=150)
plt.close()

# --- 6b. Confusion matrice za sve modele ---
num_models = len(results)
ncols = 2
nrows = (num_models + 1) // 2

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
if num_models == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, (name, r) in enumerate(results.items()):
    ax = axes[idx]
    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Ne (0)", "Da (1)"], yticklabels=["Ne (0)", "Da (1)"])
    ax.set_title(f"{name}\nF2={r['f2']:.3f} | Recall={r['recall']:.3f}", fontweight="bold")
    ax.set_xlabel("Predviđeno")
    ax.set_ylabel("Stvarno")

# Sakrij prazne subplotove
for idx in range(len(results), nrows * ncols):
    axes[idx].set_visible(False)

plt.suptitle("Confusion matrice - Poređenje modela", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("analysis/figures/confusion_matrices.png", dpi=150)
plt.close()

# --- 6c. Uporedni bar plot metrika (F2, Recall, ROC-AUC) ---
fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(results.keys())
f2_scores = [results[m]['f2'] for m in model_names]
recall_scores = [results[m]['recall'] for m in model_names]
roc_scores = [results[m]['roc_auc'] for m in model_names]

x = np.arange(len(model_names))
width = 0.25

bars1 = ax.bar(x - width, f2_scores, width, label='F2', color='steelblue')
bars2 = ax.bar(x, recall_scores, width, label='Recall', color='coral')
bars3 = ax.bar(x + width, roc_scores, width, label='ROC-AUC', color='seagreen')

ax.set_xlabel('Modeli', fontsize=12)
ax.set_ylabel('Skor', fontsize=12)
ax.set_title('Poređenje performansi modela', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

# Dodaj vrednosti na vrh stubića
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("analysis/figures/metrics_comparison.png", dpi=150)
plt.close()

# --- 6d. F2 bar plot sa rangiranjem ---
fig, ax = plt.subplots(figsize=(10, 6))

# Sortirani modeli po F2
sorted_names = [r[0] for r in sorted_by_f2]
sorted_f2 = [r[1]['f2'] for r in sorted_by_f2]

# ISPRAVLJENE BOJE (bronze → peru)
colors = ['gold' if i==0 else 'silver' if i==1 else 'peru' if i==2 else 'steelblue' 
          for i in range(len(sorted_names))]

bars = ax.barh(sorted_names, sorted_f2, color=colors)
ax.set_xlabel('F2 Score', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('Rang lista modela po F2 skoru', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

# Dodaj vrednosti na krajeve
for bar, score in zip(bars, sorted_f2):
    ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{score:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("analysis/figures/f2_ranking.png", dpi=150)
plt.close()

# ========== 7. ZAVRŠNI IZVEŠTAJ ==========
print("\n" + "="*60)
print("📋 ZAVRŠNI IZVEŠTAJ")
print("="*60)
print(f"✅ Broj testiranih modela: {len(results)}")
print(f"✅ Najbolji model (po F2): {sorted_by_f2[0][0]}")
print(f"   F2 = {sorted_by_f2[0][1]['f2']:.4f}")
print(f"   Recall = {sorted_by_f2[0][1]['recall']:.4f} ({sorted_by_f2[0][1]['recall']*100:.1f}%)")
print(f"✅ Svi rezultati sačuvani u folderu: analysis/")
print(f"✅ Svi grafikoni sačuvani u: analysis/figures/")
print(f"✅ Rang lista po F2: analysis/ranking_by_f2_analysis.csv")
print("="*60)