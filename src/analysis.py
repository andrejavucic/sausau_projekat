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
X_test = np.load("data/processed/X_test_preprocessed.npy")
y_test = np.load("data/processed/y_test.npy")

# Originalni trening podaci (bez SMOTE-a) - za dijagnostiku overfittinga
X_train_orig = np.load("data/processed/X_train_preprocessed.npy")
y_train_orig = np.load("data/processed/y_train.npy")

# ========== 2. UČITAVANJE TRENIRANIH MODELA ==========
models_dir = "models"
results = {}

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
        #print(f"✅ Učitano: {name}")
    else:
        print(f"❌ Nije pronađeno: {fname}")

if len(results) == 0:
    raise FileNotFoundError("Nijedan model nije pronađen! Prvo pokreni train.py")

# ========== 4. RANGIRANJE PO F2 (default prag 0.5) ==========
for name, r in results.items():
    model = r["model"]
    
    # Predikcije sa default pragom 0.5
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrike
    r["accuracy"] = accuracy_score(y_test, y_pred)
    r["precision"] = precision_score(y_test, y_pred)
    r["recall"] = recall_score(y_test, y_pred)
    r["f1"] = f1_score(y_test, y_pred)
    r["f2"] = fbeta_score(y_test, y_pred, beta=2)
    r["roc_auc"] = roc_auc_score(y_test, y_prob)
    r["cm"] = confusion_matrix(y_test, y_pred)
    r["y_prob"] = y_prob

sorted_by_f2 = sorted(results.items(), key=lambda x: x[1]['f2'], reverse=True)

# ========== 5. NOVO: ANALIZA RAZLIČITIH PRAGOVA (0.3, 0.35, 0.4, 0.5) ==========
print("\n" + "="*60)
print("🎯 ANALIZA RAZLIČITIH PRAGOVA")
print("="*60)

thresholds = [0.3, 0.35, 0.4, 0.5]
threshold_results = {}

for name, r in results.items():
    y_prob = r["y_prob"]
    threshold_results[name] = {}
    
    print(f"\n📊 {name}")
    print("-" * 40)
    
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        
        # Metrike za ovaj prag
        recall_thresh = recall_score(y_test, y_pred_thresh)
        f2_thresh = fbeta_score(y_test, y_pred_thresh, beta=2)
        precision_thresh = precision_score(y_test, y_pred_thresh)
        f1_thresh = f1_score(y_test, y_pred_thresh)
        
        # Sačuvaj
        threshold_results[name][thresh] = {
            "recall": recall_thresh,
            "f2": f2_thresh,
            "precision": precision_thresh,
            "f1": f1_thresh
        }
        
        print(f"   Prag {thresh:.2f}: Recall={recall_thresh:.4f}, F2={f2_thresh:.4f}, Precision={precision_thresh:.4f}")

# ========== 6. NOVO: HEATMAPE ZA PRAGOVE ==========
# 6a. Heatmap za Recall kroz pragove
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap 1: Recall
recall_data = []
for name in results.keys():
    row = []
    for thresh in thresholds:
        row.append(threshold_results[name][thresh]["recall"])
    recall_data.append(row)

recall_df = pd.DataFrame(recall_data, 
                         index=list(results.keys()),
                         columns=[f"{t:.2f}" for t in thresholds])

sns.heatmap(recall_df, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[0],
            cbar_kws={'label': 'Recall Score'})
axes[0].set_title('RECALL po pragovima\n', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Prag')
axes[0].set_ylabel('Model')

# Heatmap 2: F2
f2_data = []
for name in results.keys():
    row = []
    for thresh in thresholds:
        row.append(threshold_results[name][thresh]["f2"])
    f2_data.append(row)

f2_df = pd.DataFrame(f2_data,
                     index=list(results.keys()),
                     columns=[f"{t:.2f}" for t in thresholds])

sns.heatmap(f2_df, annot=True, fmt=".4f", cmap="Greens", ax=axes[1],
            cbar_kws={'label': 'F2 Score'})
axes[1].set_title('F2 SKOR po pragovima\n', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Prag')
axes[1].set_ylabel('Model')

plt.suptitle('Uticaj praga na performance modela', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("analysis/figures/threshold_heatmaps.png", dpi=150)
plt.close()

# ========== 8. NOVO: TRAIN VS TEST PLOT ==========
for name, r in results.items():
    model = r["model"]
    y_train_pred_orig = model.predict(X_train_orig)
    r['train_f1_original'] = f1_score(y_train_orig, y_train_pred_orig)
    r['train_f2_original'] = fbeta_score(y_train_orig, y_train_pred_orig, beta=2)
                                         
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: F1
models = list(results.keys())
train_f1_scores = [results[m]['train_f1_original'] for m in models]
test_f1_scores = [results[m]['f1'] for m in models]

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, train_f1_scores, width, label='Train F1 (originalni)', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, test_f1_scores, width, label='Test F1', color='coral', alpha=0.8)
axes[0].set_xlabel('Modeli', fontsize=12)
axes[0].set_ylabel('F1 Score', fontsize=12)
axes[0].set_title('Poređenje: Trening (original) vs Test F1', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].legend()
axes[0].set_ylim(0, 1)
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: F2
train_f2_scores = [results[m]['train_f2_original'] for m in models]
test_f2_scores = [results[m]['f2'] for m in models]

axes[1].bar(x - width/2, train_f2_scores, width, label='Train F2 (originalni)', color='steelblue', alpha=0.8)
axes[1].bar(x + width/2, test_f2_scores, width, label='Test F2', color='coral', alpha=0.8)
axes[1].set_xlabel('Modeli', fontsize=12)
axes[1].set_ylabel('F2 Score', fontsize=12)
axes[1].set_title('Poređenje: Trening (original) vs Test F2', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].legend()
axes[1].set_ylim(0, 1)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Dijagnostika overfittinga - Train vs Test', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("analysis/figures/train_vs_test_diagnostic.png", dpi=150)
plt.close()

# ========== 9. SAŽETAK DIJAGNOSTIKE ==========
print("\n" + "="*60)
print("SAŽETAK DIJAGNOSTIKE (svi modeli)")
print("="*60)

diagnostic_data = []
for name, r in results.items():
    train_f1 = r['train_f1_original']
    test_f1 = r['f1']
    gap = train_f1 - test_f1
    
    if gap > 0.15:
        status = "⚠️ Visok overfitting"
    elif gap > 0.08:
        status = "⚠️ Umeren overfitting"
    elif abs(gap) < 0.05:
        status = "✅ Dobra generalizacija"
    else:
        status = "✅ Generalizuje dobro"
    
    diagnostic_data.append({
        "Model": name,
        "Train F1 (orig)": round(train_f1, 4),
        "Test F1": round(test_f1, 4),
        "Gap": round(gap, 4),
        "Status": status,
        "Recall": round(r['recall'], 4),
        "F2": round(r['f2'], 4)
    })

diag_df = pd.DataFrame(diagnostic_data)
print(diag_df.to_string(index=False))
diag_df.to_csv("analysis/diagnostic_overfitting.csv", index=False)

# ========== 10. POREĐENJE SA BASELINE MODELOM ==========
print("\n" + "="*60)
print("📊 POREĐENJE SA BASELINE MODELOM")
print("="*60)

# Stratified baseline
stratified_baseline = DummyClassifier(strategy='stratified', random_state=42)
stratified_baseline.fit(X_train_orig, y_train_orig)

y_pred_stratified = stratified_baseline.predict(X_test)
y_prob_stratified = stratified_baseline.predict_proba(X_test)[:, 1]

baseline_f1 = f1_score(y_test, y_pred_stratified)
baseline_f2 = fbeta_score(y_test, y_pred_stratified, beta=2)
baseline_recall = recall_score(y_test, y_pred_stratified)
baseline_precision = precision_score(y_test, y_pred_stratified)
baseline_accuracy = accuracy_score(y_test, y_pred_stratified)
baseline_roc_auc = roc_auc_score(y_test, y_prob_stratified)

print(f"\n📌 Stratified baseline (nasumično predviđanje):")
print(f"   Accuracy:  {baseline_accuracy:.4f}")
print(f"   Precision: {baseline_precision:.4f}")
print(f"   Recall:    {baseline_recall:.4f} ({baseline_recall*100:.1f}%)")
print(f"   F1:        {baseline_f1:.4f}")
print(f"   F2:        {baseline_f2:.4f}")
print(f"   ROC-AUC:   {baseline_roc_auc:.4f}")

# Pronađi DVA NAJBOLJA modela:
# 1. Najbolji po F2 (Random Forest verovatno)
best_f2_model_name, best_f2_model_data = sorted_by_f2[0]

# 2. Najbolji po Recall-u (Logistic Regression verovatno)
best_recall_model_name = max(results.items(), key=lambda x: x[1]['recall'])[0]
best_recall_model_data = results[best_recall_model_name]

print(f"\n" + "="*60)
print(f"🏆 POREĐENJE SA DVA NAJBOLJA MODELA")
print("="*60)

# Poređenje 1: Najbolji po F2
print(f"\n📊 1. NAJBOLJI PO F2 SKORU: {best_f2_model_name}")
print("-" * 40)
print(f"   {'Metrika':<12} {'Baseline':>10} {'Model':>10} {'Razlika':>10}")
print(f"   {'-'*42}")
print(f"   {'F1':<12} {baseline_f1:>10.4f} {best_f2_model_data['f1']:>10.4f} {best_f2_model_data['f1'] - baseline_f1:>+10.4f}")
print(f"   {'F2':<12} {baseline_f2:>10.4f} {best_f2_model_data['f2']:>10.4f} {best_f2_model_data['f2'] - baseline_f2:>+10.4f}")
print(f"   {'Recall':<12} {baseline_recall:>10.4f} {best_f2_model_data['recall']:>10.4f} {best_f2_model_data['recall'] - baseline_recall:>+10.4f}")
print(f"   {'Precision':<12} {baseline_precision:>10.4f} {best_f2_model_data['precision']:>10.4f} {best_f2_model_data['precision'] - baseline_precision:>+10.4f}")

if best_f2_model_data['f2'] > baseline_f2:
    improvement = (best_f2_model_data['f2'] - baseline_f2) / baseline_f2 * 100 if baseline_f2 > 0 else float('inf')
    print(f"\n✅ {best_f2_model_name} je BOLJI od baseline modela")
    if baseline_f2 > 0:
        print(f"   Poboljšanje od +{improvement:.1f}% u F2 skoru")
elif best_f2_model_data['f2'] < baseline_f2:
    print(f"\n⚠️ Baseline je BOLJI od {best_f2_model_name}!")
else:
    print(f"\n📊 Modeli su JEDNAKI")

# Poređenje 2: Najbolji po Recall-u
print(f"\n\n📊 2. NAJBOLJI PO RECALL-U: {best_recall_model_name}")
print("-" * 40)
print(f"   {'Metrika':<12} {'Baseline':>10} {'Model':>10} {'Razlika':>10}")
print(f"   {'-'*42}")
print(f"   {'F1':<12} {baseline_f1:>10.4f} {best_recall_model_data['f1']:>10.4f} {best_recall_model_data['f1'] - baseline_f1:>+10.4f}")
print(f"   {'F2':<12} {baseline_f2:>10.4f} {best_recall_model_data['f2']:>10.4f} {best_recall_model_data['f2'] - baseline_f2:>+10.4f}")
print(f"   {'Recall':<12} {baseline_recall:>10.4f} {best_recall_model_data['recall']:>10.4f} {best_recall_model_data['recall'] - baseline_recall:>+10.4f}")
print(f"   {'Precision':<12} {baseline_precision:>10.4f} {best_recall_model_data['precision']:>10.4f} {best_recall_model_data['precision'] - baseline_precision:>+10.4f}")

if best_recall_model_data['recall'] > baseline_recall:
    improvement = (best_recall_model_data['recall'] - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else float('inf')
    print(f"\n✅ {best_recall_model_name} je BOLJI od baseline modela")
    if baseline_recall > 0:
        print(f"   Poboljšanje od +{improvement:.1f}% u Recall-u")
elif best_recall_model_data['recall'] < baseline_recall:
    print(f"\n⚠️ Baseline je BOLJI od {best_recall_model_name}!")
else:
    print(f"\n📊 Modeli su JEDNAKI")
print()

# ========== 11. Vizuelna interpretacija ==========

# --- 11a. ROC krive ---
plt.figure(figsize=(10, 8))
ax = plt.gca()

for name, r in results.items():
    RocCurveDisplay.from_estimator(r["model"], X_test, y_test, 
                                   name=name, ax=ax)

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Nasumičan model (AUC=0.5)')
plt.title('ROC krive za sve modele', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specifičnost)', fontsize=12)
plt.ylabel('True Positive Rate (Recall / Osetljivost)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("analysis/figures/roc_curves.png", dpi=150)
plt.close()

# --- 11b. Confusion matrice za sve modele (prag 0.5) ---
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

for idx in range(len(results), nrows * ncols):
    axes[idx].set_visible(False)

plt.suptitle("Confusion matrice - Poređenje modela (prag 0.5)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("analysis/figures/confusion_matrices.png", dpi=150)
plt.close()

# --- 11c. Uporedni bar plot metrika (F2, Recall, ROC-AUC) ---
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

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("analysis/figures/metrics_comparison.png", dpi=150)
plt.close()