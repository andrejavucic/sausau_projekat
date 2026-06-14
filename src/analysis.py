import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              precision_recall_curve)
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

# zbog vizuelizacije
# ========== 3. EVALUACIJA SVIH MODELA ==========
print("\n" + "="*60)
print("EVALUACIJA MODELA NA TEST SKUPU")
print("="*60)

for name, r in results.items():
    model = r["model"]
    
    # Predikcije
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrike
    r["accuracy"] = accuracy_score(y_test, y_pred)
    r["precision"] = precision_score(y_test, y_pred)
    r["recall"] = recall_score(y_test, y_pred)
    r["f1"] = f1_score(y_test, y_pred)
    r["roc_auc"] = roc_auc_score(y_test, y_prob)
    r["cm"] = confusion_matrix(y_test, y_pred)
    r["y_prob"] = y_prob  # sačuvaj za kasnije
    
    print(f"\n📊 {name}")
    print(f"   Accuracy:  {r['accuracy']:.4f}")
    print(f"   Precision: {r['precision']:.4f}")
    print(f"   Recall:    {r['recall']:.4f}")
    print(f"   F1:        {r['f1']:.4f}")
    print(f"   ROC-AUC:   {r['roc_auc']:.4f}")



# ========== 6. Vizuelna interpretacija ==========
# --- 6a. ROC krive za sve modele na jednom plotu (NAJVAŽNIJE) ---
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
    ax.set_title(f"{name}\nF1={r['f1']:.3f} | Recall={r['recall']:.3f}", fontweight="bold")
    ax.set_xlabel("Predviđeno")
    ax.set_ylabel("Stvarno")

# Sakrij prazne subplotove
for idx in range(len(results), nrows * ncols):
    axes[idx].set_visible(False)

plt.suptitle("Confusion matrice - Poređenje modela", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("analysis/figures/confusion_matrices.png", dpi=150)
plt.close()

# --- 6c. Uporedni bar plot metrika (F1, Recall, ROC-AUC) ---
fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(results.keys())
f1_scores = [results[m]['f1'] for m in model_names]
recall_scores = [results[m]['recall'] for m in model_names]
roc_scores = [results[m]['roc_auc'] for m in model_names]

x = np.arange(len(model_names))
width = 0.25

bars1 = ax.bar(x - width, f1_scores, width, label='F1', color='steelblue')
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


# ========== 8. ANALIZA REZULTATA PREDIKCIJE I ZAKLJUČCI ==========

# 8a. Poređenje sa "glupim" modelom (baseline)
"""
LOS PRIMER -> JER POREDIMO SA F1 -> KOJE CE UVEK BITI 0

# 1. Učitaj ORIGINALNE podatke (bez SMOTE-a)
# jer je tamo 50/50 odnos za yes/no u izlazu
X_train_original = np.load("data/processed/X_train_preprocessed.npy") 
y_train_original = np.load("data/processed/y_train.npy")  # original (89% NE, 11% DA)
y_test_original = np.load("data/processed/y_test.npy")    

# 2. Baseline
# nas basline -> uvek kaze ne (jer je to vecinska klasa)
unique, counts = np.unique(y_train_original, return_counts=True)
majority_class = unique[np.argmax(counts)]
baseline_pred = np.full_like(y_test_original, majority_class)
"""
# ========== STRATIFIED BASELINE  ==========
# nasumicno predvidja yes/no, ali oslanjajuci na odnos (distribucjiu) sa treninga
print("\n" + "="*60)
print("STRATIFIED BASELINE MODEL")
print("="*60)

# KORISTIMO DISTIBUCIJU ORIGINALNIH POD !!!

# Stratified baseline (nasumično po distribuciji iz treninga)
stratified_baseline = DummyClassifier(strategy='stratified', random_state=42)
stratified_baseline.fit(X_train_orig, y_train_orig)

# Predikcije
y_pred_stratified = stratified_baseline.predict(X_test)

# Metrike (definiši sve varijable!)
stratified_f1 = f1_score(y_test, y_pred_stratified)
stratified_recall = recall_score(y_test, y_pred_stratified)
stratified_precision = precision_score(y_test, y_pred_stratified)
stratified_da_pct = y_pred_stratified.mean() * 100

print(f"\nStratified baseline (nasumično predviđanje):")
print(f"   F1: {stratified_f1:.4f}")
print(f"   Recall: {stratified_recall:.4f} ({stratified_recall*100:.1f}%)")
print(f"   Precision: {stratified_precision:.4f} ({stratified_precision*100:.1f}%)")
print(f"   Predviđeno DA: {stratified_da_pct:.1f}%")

# Poređenje sa Random Forest-om
if "Random Forest" in results:
    rf_f1 = results["Random Forest"]["f1"]
    rf_recall = results["Random Forest"]["recall"]
    
    print(f"\n📊 POREĐENJE:")
    print(f"   Stratified baseline: F1={stratified_f1:.4f}, Recall={stratified_recall:.4f}")
    print(f"   Random Forest:       F1={rf_f1:.4f}, Recall={rf_recall:.4f}")
    
    if rf_f1 > stratified_f1:
        if stratified_f1 > 0:
            improvement = (rf_f1 - stratified_f1) / stratified_f1 * 100
            print(f"\n✅ Random Forest je BOLJI za +{improvement:.0f}%")
        else:
            print("\n✅ Random Forest je bolji od baseline modela")
    elif rf_f1 < stratified_f1:
        print(f"\n⚠️ Stratified baseline je BOLJI od Random Forest-a!")
    else:
        print(f"\n📊 Modeli su JEDNAKI")
else:
    print("\n⚠️ Random Forest nije pronađen u results dictionary-ju")


# 8c. Analiza gde model greši (confusion matrix interpretacija)
print("\n🔍 8c. ANALIZA GREŠAKA MODELA")
print("-" * 40)

for name, r in results.items():
    tn, fp, fn, tp = r['cm'].ravel()
    total = tn + fp + fn + tp
    
    # Tipovi grešaka
    false_positives = fp / total * 100  # lažno pozitivni (rekli DA, a nije)
    false_negatives = fn / total * 100  # lažno negativni (rekli NE, a jeste)
    
    print(f"\n{name}:")
    print(f"   ✅ Tačno pozitivnih: {tp} (klijenti za koje je tačno rekao da hoće)")
    print(f"   ❌ Lažno negativnih: {fn} ({false_negatives:.1f}%) - PROPUSTILI smo ih!")
    print(f"   ❌ Lažno pozitivnih: {fp} ({false_positives:.1f}%) - DŽABE ih zovemo")
    
    # Poslovna implikacija
    if fn > fp * 2:
        print(f"   ⚠️  Problem: PROPUŠTA mnogo potencijalnih klijenata! (visok FN)")
    elif fp > fn * 2:
        print(f"   ⚠️  Problem: GUBI VREME na loše klijente! (visok FP)")

# ========== 9. DIJAGNOSTIKA OVERFITTING / UNDERFITTING ==========
X_train_orig = np.load("data/processed/X_train_preprocessed.npy")
y_train_orig = np.load("data/processed/y_train.npy")  # 89% NE, 11% DA

print("\n" + "="*60)
print("ISPRAVNA DIJAGNOSTIKA (na ORIGINALNIM podacima)")
print("="*60)
print(f"Train original: {y_train_orig.mean()*100:.1f}% DA")
print(f"Test:           {y_test.mean()*100:.1f}% DA")
print("="*60)

for name, r in results.items():
    model = r["model"]
    
    # 1. Train na ORIGINALNIM podacima (ne SMOTE)
    y_train_pred_orig = model.predict(X_train_orig)
    train_f1_orig = f1_score(y_train_orig, y_train_pred_orig)
    train_acc_orig = accuracy_score(y_train_orig, y_train_pred_orig)
    
    # 2. Test metrike
    test_f1 = r['f1']
    test_acc = r['accuracy']
    
    # 3. Gap sada ima SMISLA
    f1_gap = train_f1_orig - test_f1
    
    print(f"\n📊 {name}")
    print(f"   Train F1 (originalni): {train_f1_orig:.4f}")
    print(f"   Test F1:               {test_f1:.4f}")
    print(f"   Gap:                   {f1_gap:+.4f}")
    
    # 4. Dijagnostika
    if name == "KNN" and train_f1_orig > 0.99:
        print(f"   🔴 OVERFITTING: KNN gotovo savršeno pamti trening podatke")
        print(f"      → Razmotri veći broj suseda (k)")

    elif f1_gap > 0.10:
        print(f"   ⚠️ Moguć overfitting (gap = {f1_gap:.3f})")
        print(f"      → Model radi znatno bolje na treningu nego na testu")

    elif abs(f1_gap) < 0.05:
        print(f"   ✅ Dobra generalizacija (gap = {f1_gap:.3f})")

    else:
        print(f"   ℹ️ Umeren pad performansi između treninga i testa")

    # Poslovna interpretacija
    if r['recall'] < 0.50:
        print(f"      ⚠️ Recall je nizak ({r['recall']:.3f}) → propuštaju se potencijalni klijenti")

    if r['precision'] < 0.50:
        print(f"      ⚠️ Precision je nizak ({r['precision']:.3f}) → mnogo nepotrebnih poziva")


    # Sačuvaj u results
    r['train_f1_original'] = train_f1_orig
    r['train_acc_original'] = train_acc_orig

# ========== 9b. SAŽETAK (tabela za sve modele) ==========
print("\n" + "="*60)
print("SAŽETAK DIJAGNOSTIKE (svi modeli)")
print("="*60)

# Napravi DataFrame za pregled
diagnostic_data = []
for name, r in results.items():
    train_orig = r['train_f1_original']
    test_f1 = r['f1']
    gap = train_orig - test_f1
    
    if gap > 0.15:
        status = "⚠️ Overfitting"
    elif train_orig < 0.35 and test_f1 < 0.35:
        status = "🔻 Underfitting"
    elif abs(gap) < 0.05:
        status = "✅ Dobra generalizacija"
    else:
        status = "✅ Generalizuje dobro"
    
    diagnostic_data.append({
        "Model": name,
        "Train F1 (orig)": round(train_orig, 4),
        "Test F1": round(test_f1, 4),
        "Gap": round(gap, 4),
        "Status": status,
        "Recall": round(r['recall'], 4)
    })

diag_df = pd.DataFrame(diagnostic_data)
print(diag_df.to_string(index=False))
diag_df.to_csv("analysis/diagnostic_overfitting.csv", index=False)

# ========== 9c. VIZUALIZACIJA (trening vs test - originalni podaci) ==========
fig, ax = plt.subplots(figsize=(10, 6))

models = list(results.keys())
train_f1_scores = [results[m]['train_f1_original'] for m in models]
test_f1_scores = [results[m]['f1'] for m in models]

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, train_f1_scores, width, label='Train F1 (originalni)', color='steelblue', alpha=0.8)
ax.bar(x + width/2, test_f1_scores, width, label='Test F1', color='coral', alpha=0.8)

ax.set_xlabel('Modeli', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Poređenje: Trening (original) vs Test F1', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("analysis/figures/train_vs_test_diagnostic.png", dpi=150)
plt.close()

"""
# ==================== PROVERA PRAGA ============
X_val = np.load("data/processed/X_val_preprocessed.npy")   
y_val = np.load("data/processed/y_val.npy")
threshold_results = {}

for name, r in results.items():
    model = r["model"]
    
    # ===== KORAK 1: Pronađi optimalni prag na VALIDACIONOM skupu =====
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    # Precision-Recall kriva na validacionom skupu
    precisions_val, recalls_val, thresholds_val = precision_recall_curve(y_val, y_val_prob)
    
    # F1 za svaki prag (na validaciji)
    f1_scores_val = 2 * (precisions_val[:-1] * recalls_val[:-1]) / (precisions_val[:-1] + recalls_val[:-1] + 1e-10)
    
    # Najbolji prag na VALIDACIJI
    best_idx_val = np.argmax(f1_scores_val)
    best_threshold = thresholds_val[best_idx_val]
    best_f1_val = f1_scores_val[best_idx_val]
    
    # ===== KORAK 2: Testiraj taj prag na TEST skupu =====
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred_optimized = (y_test_prob >= best_threshold).astype(int)
    
    # Metrike na TEST skupu sa optimalnim pragom (pronadjenim na validaciji)
    opt_accuracy = accuracy_score(y_test, y_test_pred_optimized)
    opt_precision = precision_score(y_test, y_test_pred_optimized)
    opt_recall = recall_score(y_test, y_test_pred_optimized)
    opt_f1 = f1_score(y_test, y_test_pred_optimized)
    
    # Default prag (0.5) na test skupu
    y_test_pred_default = (y_test_prob >= 0.5).astype(int)
    default_f1_test = f1_score(y_test, y_test_pred_default)
    default_recall_test = recall_score(y_test, y_test_pred_default)
    default_precision_test = precision_score(y_test, y_test_pred_default)
    
    # Sačuvaj rezultate
    threshold_results[name] = {
        "best_threshold": best_threshold,
        "best_threshold_val_f1": best_f1_val,  # F1 na validaciji sa optimalnim pragom
        
        # Metrike na TEST skupu sa default pragom (0.5)
        "default_f1": default_f1_test,
        "default_recall": default_recall_test,
        "default_precision": default_precision_test,
        
        # Metrike na TEST skupu sa optimalnim pragom (pronadjenim na validaciji)
        "optimized_f1": opt_f1,
        "optimized_recall": opt_recall,
        "optimized_precision": opt_precision,
        "improvement": opt_f1 - default_f1_test,
        
        # Za vizualizaciju
        "y_val_prob": y_val_prob,
        "y_test_prob": y_test_prob,
        "precisions_val": precisions_val,
        "recalls_val": recalls_val,
        "thresholds_val": thresholds_val,
        "f1_scores_val": f1_scores_val,
        "best_idx_val": best_idx_val
    }
    
    print(f"\n{name}:")
    print(f"  Optimalni prag (iz VALIDACIJE): {best_threshold:.4f} (F1 na validaciji: {best_f1_val:.4f})")
    print(f"  TEST - Default prag (0.5): F1={default_f1_test:.4f}, Recall={default_recall_test:.4f}")
    print(f"  TEST - Optimalni prag:     F1={opt_f1:.4f}, Recall={opt_recall:.4f}")
    improvement = opt_f1 - default_f1_test
    sign = "+" if improvement >= 0 else ""
    print(f"  Poboljšanje na TEST: {sign}{improvement:.4f} F1")

# ========== 7. NAJBOLJI MODEL SA OPTIMALNIM PRAGOM ==========
print("\n" + "="*60)
print("NAJBOLJI MODEL NAKON OPTIMIZACIJE PRAGA")
print("="*60)

# Poredi po optimizovanom F1 na TEST skupu
best_optimized = max(threshold_results.items(), key=lambda x: x[1]['optimized_f1'])
best_name, best_data = best_optimized

print(f"\n🏆 Najbolji model sa optimizovanim pragom: {best_name}")
print(f"   Optimalni prag (iz VALIDACIJE): {best_data['best_threshold']:.4f}")
print(f"   TEST F1: {best_data['optimized_f1']:.4f} (bio {best_data['default_f1']:.4f})")
print(f"   TEST Recall: {best_data['optimized_recall']:.4f} (bio {best_data['default_recall']:.4f})")
print(f"   TEST Precision: {best_data['optimized_precision']:.4f} (bio {best_data['default_precision']:.4f})")
print ()
"""