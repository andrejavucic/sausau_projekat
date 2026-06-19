import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from itertools import product
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import RocCurveDisplay
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import fbeta_score, make_scorer

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

os.makedirs("analysis", exist_ok=True)
os.makedirs("analysis/figures", exist_ok=True)
sns.set_style("whitegrid")

# ========== 1. Učitavanje ==========
#X_train = np.load("data/processed/X_train_resampled.npy")   # SMOTE pod
#y_train = np.load("data/processed/y_train_resampled.npy")

X_train = np.load("data/processed/X_train_preprocessed.npy")  # originalni, ne resampled
y_train = np.load("data/processed/y_train.npy")

X_val = np.load("data/processed/X_val_preprocessed.npy")   
y_val = np.load("data/processed/y_val.npy") 
X_test  = np.load("data/processed/X_test_preprocessed.npy")
y_test  = np.load("data/processed/y_test.npy")

# ========== 2. Modeli + gridovi za hiperparametre ==========
# K - fold unakrsna validacija, k=5 -> deli na 5 delova
# 5 puta se prolazi kroz trening, i na kraju se bira najbolji
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f2_scorer = make_scorer(fbeta_score, beta=2)  

# KORELACIJA: stabla odlucivanja i ansambl metode su znatno otpornije na koleralciju
# jer kad stablo deli pod, samo ce u tom trenutku izabrati atribut koji bolje deli podatke

# n_jobs -> koliko procesorskoh jezgra koristimo (-1 -> sva jezgra)
models_config = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000,random_state=42, #class_weight="balanced",
                                    C=0.01,
                                    penalty='l1',
                                    solver='saga'
                                    ),
       "grid" : {},
        # pod grid spadaju oni parametri za koje cemo probati grid da vidimo sta je naj
#        "grid": {
            # jacina regularizacije (C je suprotno lambda parametru)
            # 10 -> oslanja se minimizaciju greske (overfitting) 
            # ako gledamo naspram lambde, kada ga nemamo - ne kaznjavamo greske i tada moze doci do overfittinga
            # tkd malo lambda -> overfitting -> za C je samo obrnuto
            # 00.1 -> jace kaznjava velike koeficijente (underfitting) -> jednostavniji model
            # 
#            "C": [0.01, 0.1, 1, 10], 

            # l1 -> LASSO REGURALIZACIJA-> regulator se zasniva na apsolutnoj vr koeficijenata
            # moze svesti koeficijente tacno na 0, pronadji koji atribut ne pomaze modelu i izabaci ga
            # l2 -> RIDGE REGURALIZACIJA > reg kaznjava velike koeficijente -> ()^2
            # smanjuje koeficijente, ali ih najcesce ne svodi na 0\
            # smanjuje se varijnsa
#            "penalty": ["l1", "l2"],
            
            # nacin na koji se resava regularizacija
            # navodno neophodno zbog l1 regularizacije
            # oba nacina podrzavaju i l1 i l2
#            "solver": ["liblinear", "saga"],
#        },
    },
    "KNN": {
        "model": KNeighborsClassifier(n_jobs=-1, 
                                       n_neighbors=25,
                                        p=1,
                                       weights='uniform'
                                      ),
         "grid" : {},
#        "grid": {
            # izbor za k - br suseda (umesto Elbow metode)
            # malo K -> osetljiv na sum (overfitting)
            # veliko K -> niska varijnsa, veliki bias (underfitting)
            # bias - koliko je model pojednostavio problem
#            "n_neighbors": [3, 5, 7, 9, 11, 15, 19, 25],

            # tezina suseda:
            # uniform -> svi susedi iste tezine
            # distance -> blizi su znacajniji (imaju vecu tezinu)
#            "weights": ["uniform", "distance"],

            # nacin kako racunamo rastojanje:
            # p=1 -> Manhattan -> ono sa apsolutnom vr
            # putanja samo ortogonalnim koracima (strelice || sa x i y-osama)
            # p=2 -> Euklidsko -> koren kvadrata
            # najbliza vazdusna putanja
#            "p": [1, 2], 
#        },
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42, #class_weight="balanced",
                                        # najbolji kriterijumi -> izabrani gridom
                                        criterion='entropy',
                                        max_depth=5,
                                        min_samples_leaf=1,
                                        min_samples_split=2,
                                         ),
                                    
        "grid" : {},
#        "grid": {
            #velika -> overfitting, mala-> underfitting
#            "max_depth": [5, 10, 15, 20, None],
                # sled put treba staviti samo [4, 5, 6]
            
            #min br uzoraka za podatak
#           "min_samples_split": [2, 5, 10, 20],
            # [15, 20, 25] (ostalo mozemo izostaviti)
            
            # min br uzoraka u listu
#            "min_samples_leaf": [1, 2, 4, 8],

            #za merenje cistoce:
            # Gini: mera pomesanosti klasa u cvoru
            # Entropija: mera neuredjenosti(nestigurnosti) u cvoru
            # max pomesanost 50%/50%, max uredjenost 100%/0%
#            "criterion": ["gini", "entropy"],

            # ima jos i max_features - max br uzoraka za analizu

            # daje iste rez za oba
#            "class_weight": [None, "balanced"] 
#        },       
    },
    # =========================== ANSAMBL METODE ============================

    # BAGGING METODA: vise razliciih stabala koji se treniraju nezavinsno, uz uvodjenje slucajnosti
    # slucajnost u izboru pod i atributa - cini stabla medjusobno razlicitim
    # smanjenje varijanse, osetljiv na hiperparametre, umanjuje uticaj jednog singularnog stabla
    # kod klasifikacije -> svako stablo glasa za 1 klasu -> vecinska se bira
     "Random Forest": { 
        "model": RandomForestClassifier(random_state=42, n_jobs=-1,# class_weight="balanced",
                                        n_estimators=50,
                                        max_depth=5,
                                        min_samples_split=10,
                                        min_samples_leaf=1,
                                        max_features='sqrt'
                                        ),
        "grid" : {},
#        "grid": {
            # veci broj stabala = stabilniji model, ali sporiji
#            "n_estimators": [50, 100, 200],
            
#            "max_depth": [5, 10, 15, None],
            
            # min broj uzoraka za podelu cvora
#            "min_samples_split": [2, 5, 10],
            
            # min broj uzoraka u listu
#            "min_samples_leaf": [1, 2, 4],
            
            # br atributa koje razmatra za najbolju podelu (max br uzoraka za analizu)
            # manje atributa -> veca sansa da razlicita stabla biraju razlicite podele
            # 'sqrt' = sqrt(broj_atributa) - dobro za visoke dimenzije 
            # npr: 100 atributa -> sqrt(100) = 10 -> uzme 10 atributa 
            # 'log2' = log2(broj_atributa)
            # log(100) ~ 6,7 atributa 
            # None = svi atributi (moramo da poredimo sve atribute)
#           "max_features": ['sqrt', 'log2', None],
#        },
    },
    
    # BOOSTING METODA: modeli se treniraju sekvencijalno, svaki sled popravlja greske prethonog
    # postepeno se smanjuje greska
    "Gradient Boosting": {      
        "model": GradientBoostingClassifier(random_state=42,
                                            n_estimators=100,
                                            learning_rate=0.03,
                                            max_depth=2,
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            subsample=0.8
                                            ),
        "grid" : {},
#        "grid": {
#            "n_estimators": [100, 150],
            
            # korak ucenja -> koliko svako stablo utice na konacnu predikciju
            # manji learning_rate zahteva vise stabala -> kompenzacija parametara
            # kroz vise iteracija, ucimo manje, smanjuje sansu od overfittinga, uci polako-stabilnije
            # moze i obnuto, ali je rizicnije zbog oscilacija
#            "learning_rate": [0.03, 0.05, 0.1],
            
            # max dubina - obicno mala (3-5) jer GB ne voli duboka stabla
#            "max_depth": [2, 3, 4],
            
            # Subsampling - koji procenat tr podataka ce svako stablo koristiti za ucenje
            # jer se vrsi sekvencijalno, pod se ne ponavljaju -> cilj smanjenje overfittinga
#            "subsample": [0.8, 1.0],
            
            # min broj uzoraka za podelu
#            "min_samples_split": [2, 5],
            
            # min broj uzoraka u listu
#            "min_samples_leaf": [1, 2]
#        },
    },
}

# ========== 3. GridSearch + evaluacija ==========
results = {}

for name, cfg in models_config.items():
    # PIPELINE ZA K-FOLD 
    # ovde radimo smote umesto u preprocessingu
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', cfg["model"])
    ])

    param_grid_prefixed = {f"model__{k}": v for k, v in cfg["grid"].items()}

    # GridSearch sa k-fold cross-validacijom
    gs = GridSearchCV(
        pipe, 
        param_grid_prefixed, 
        cv=cv, 
        scoring=f2_scorer, 
        n_jobs=-1, 
        verbose=1
    )

    print(f"Broj kombinacija: {len(ParameterGrid(param_grid_prefixed))}")
    gs.fit(X_train, y_train)
    
    # Najbolji model (ceo pipeline sa SMOTE)
    best_model = gs.best_estimator_
    best_score = gs.best_score_  # Prosečan CV F1 kroz 5 foldova
    best_params = gs.best_params_
    
    print(f"\n✅ Najbolji CV F2: {best_score:.4f}")
    print(f"✅ Najbolji parametri: {best_params}")
    
    # Evaluacija na TEST skupu (test NIKAD ne prolazi kroz SMOTE)
    print(f"\nEvaluacija na TEST skupu...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # ========== 4. TESTIRANJE RAZLIČITIH PRAGOVA ==========
    thresholds = [0.3, 0.35, 0.4]
    threshold_results = {}
    
    print(f"\n{'='*60}")
    print(f"📊 TESTIRANJE PRAGOVA ZA {name}")
    print(f"{'='*60}")
    print(f"{'Prag':>8} {'F2':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print(f"{'-'*60}")
    
    for thresh in thresholds:
        # Konvertuj verovatnoće u predikcije na osnovu praga
        y_pred_thresh = (y_prob >= thresh).astype(int)
        
        # Izračunaj metrike za dati prag
        f2 = fbeta_score(y_test, y_pred_thresh, beta=2)
        recall = recall_score(y_test, y_pred_thresh)
        precision = precision_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        
        threshold_results[thresh] = {
            "f2": f2,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "accuracy": accuracy_score(y_test, y_pred_thresh),
            "cm": confusion_matrix(y_test, y_pred_thresh)
        }
        
        # Ispis u tabelarnom formatu
        print(f"{thresh:>8.2f} {f2:>10.4f} {recall:>10.4f} {precision:>10.4f} {f1:>10.4f}")

    f2_05 = fbeta_score(y_test, y_pred, beta=2)
    recall_05 = recall_score(y_test, y_pred)
    precision_05 = precision_score(y_test, y_pred)
    f1_05 = f1_score(y_test, y_pred)

    print(f"{'-'*60}")
    print(f"{'0.50':>8} {f2_05:>10.4f} {recall_05:>10.4f} {precision_05:>10.4f} {f1_05:>10.4f} (default)")
    print(f"{'='*60}\n")

    # Pronađi najbolji prag po F2
    best_thresh = max(threshold_results.keys(), key=lambda t: threshold_results[t]['f2'])
    best_f2 = threshold_results[best_thresh]['f2']
    best_recall = threshold_results[best_thresh]['recall']

    print(f"🏆 NAJBOLJI PRAG ZA {name}: {best_thresh}")
    print(f"   F2 = {best_f2:.4f}, Recall = {best_recall:.4f}")
    print(f"   Poboljšanje u odnosu na prag 0.5:")
    print(f"   F2: {best_f2 - f2_05:+.4f}, Recall: {best_recall - recall_05:+.4f}")
    print(f"{'='*60}\n")   

    results[name] = {
        "model": best_model,
        "best_params": best_params,
        "cv_score": best_score,
        "accuracy":  accuracy_score(y_test, y_pred),    # udeo tacnih pred

        # predvidimo da hoce da se pretplat, a zapravo nece -> banka gubi vreme da ga opet zove (nista spec)
        "precision": precision_score(y_test, y_pred),   # FP (lazno pozitivne pred skupe)
        
        # predvidimo da nece da se pretplate, a hoce -> veliki propust za banku
        "recall":    recall_score(y_test, y_pred),      # FN (kada je opasno propustiti pozitivan slucaj)
        
        "f1":        f1_score(y_test, y_pred),          # harmonijska sredina
        "f2":       fbeta_score(y_test, y_pred, beta=2),  # <<< F2 ZA POREĐENJE

        "roc_auc":   roc_auc_score(y_test, y_prob),     # kriva za prikazivanje ponasanja modela korz razlicite pragove odlucvanja
                                                        # prag -> utice na presicion i recall (sa kojom vr uzimamo podatke kao P(validne))
        "cm":        confusion_matrix(y_test, y_pred),

        "threshold_results": threshold_results,  # ← NOVO
        "best_threshold": best_thresh,           # ← NOVO
        "best_f2": best_f2,                      # ← NOVO
        "best_recall": best_recall               # ← NOVO
    }
    print(f"  Test F2: {results[name]['f2']:.4f}")
    print(f"  Test Recall: {results[name]['recall']:.4f}")
    print(f"  Test ROC-AUC: {results[name]['roc_auc']:.4f}")


# ========== 5. ČUVANJE MODELA I REZULTATA ==========
# 5a. Čuvanje modela
for name, r in results.items():
    fname = f"models/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(r["model"], fname)

# Čuvanje najboljeg modela
# prema val skupu (ovo je bilo pre, sada cuvamo naj prema k-fold)
best_cv_score = 0
best_name = ""
best_model_obj = None

for name, r in results.items():
    if r['cv_score'] > best_cv_score:
        best_cv_score = r['cv_score']
        best_name = name
        best_model_obj = r['model']

print()
joblib.dump(best_model_obj, "models/best_model.pkl")
print(f"🏆 Najbolji model ({best_name}) sačuvan kao: models/best_model.pkl")



# 5b. Čuvanje CSV tabele (sa svim metrikama)
metrics_df = pd.DataFrame([
    {
        "Model": n, 
        "Best_params": str(r["best_params"]),
        "CV_F2": round(r["cv_score"], 4),
        "Accuracy": round(r["accuracy"], 4), 
        "Precision": round(r["precision"], 4),
        "Recall": round(r["recall"], 4), 
        "F1": round(r["f1"], 4), 
        "F2": round(r["f2"], 4),    
        "ROC-AUC": round(r["roc_auc"], 4)
    }
    for n, r in results.items()
])

# Sortiraj po F1 (bolji prvi)
metrics_df = metrics_df.sort_values("F2", ascending=False)
metrics_df.to_csv("analysis/metrics_comparison.csv", index=False)

"""
# 5c. Čuvanje TXT tabele (za brzi pregled)
with open("analysis/metrics_table.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("BANK MARKETING - REZULTATI KLASIFIKACIJE\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Model':<22} {'CV_F2':>10} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F2':>8} {'ROC':>10}\n")
    f.write("-"*80 + "\n")
    for name, r in results.items():
        f.write(f"{name:<22} {r['cv_score']:>10.4f} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['f2']:>8.4f} {r['roc_auc']:>10.4f}\n")
    f.write("-"*80 + "\n\n")
    
    # Rang lista
    f.write("RANG LISTA (prema F2 skoru):\n")  
    f.write("-"*40 + "\n")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f2'], reverse=True)  
    for i, (name, r) in enumerate(sorted_results, 1):
        f.write(f"{i}. {name}: F1={r['f1']:.4f}, F2={r['f2']:.4f}, Recall={r['recall']:.4f}, CV_F2={r['cv_score']:.4f}\n")

        """
sorted_results = sorted(results.items(), key=lambda x: x[1]['f2'], reverse=True)

# 5d. Kratak ispis u terminalu
print("\n" + "-"*50)
print("RANG LISTA (prema F2):") 
for i, (name, r) in enumerate(sorted_results, 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"  {i}."
    print(f"{medal} {name}: F1={r['f1']:.4f}, F2={r['f2']:.4f}, Recall={r['recall']:.4f}, CV_F2={r['cv_score']:.4f}") 
print ()



# ========== 6. TABELA SA REZULTATIMA PRAGOVA ==========
# 6a. CSV tabela sa F2 i Recall za pragove
threshold_comparison = []
for name, r in results.items():
    for thresh, metrics in r["threshold_results"].items():
        threshold_comparison.append({
            "Model": name,
            "Threshold": thresh,
            "F2": round(metrics["f2"], 4),
            "Recall": round(metrics["recall"], 4),
            "Precision": round(metrics["precision"], 4),
            "F1": round(metrics["f1"], 4)
        })
    
    # Dodaj i default prag 0.5
    threshold_comparison.append({
        "Model": name,
        "Threshold": 0.5,
        "F2": round(r["f2"], 4),
        "Recall": round(r["recall"], 4),
        "Precision": round(r["precision"], 4),
        "F1": round(r["f1"], 4)
    })

threshold_df = pd.DataFrame(threshold_comparison)
threshold_df = threshold_df.sort_values(["Model", "Threshold"])
threshold_df.to_csv("analysis/threshold_comparison.csv", index=False)

# 6b. Tabela sa najboljim pragovima po modelu
best_thresholds = []
for name, r in results.items():
    best_thresholds.append({
        "Model": name,
        "Best_Threshold": r["best_threshold"],
        "Best_F2": round(r["best_f2"], 4),
        "Best_Recall": round(r["best_recall"], 4),
        "Default_F2": round(r["f2"], 4),
        "Default_Recall": round(r["recall"], 4),
        "F2_Improvement": round(r["best_f2"] - r["f2"], 4),
        "Recall_Improvement": round(r["best_recall"] - r["recall"], 4)
    })

best_thresholds_df = pd.DataFrame(best_thresholds)
best_thresholds_df = best_thresholds_df.sort_values("Best_F2", ascending=False)
best_thresholds_df.to_csv("analysis/best_thresholds.csv", index=False)

# 6c. Tekstualni izveštaj
with open("analysis/metrics_table.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("BANK MARKETING - REZULTATI KLASIFIKACIJE SA PRAGOVIMA\n")
    f.write("="*80 + "\n\n")
    
    # Glavna tabela sa default pragom
    f.write("📊 REZULTATI SA DEFAULT PRAGOM (0.5):\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<22} {'CV_F2':>10} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F2':>8}\n")
    f.write("-"*80 + "\n")
    for name, r in results.items():
        f.write(f"{name:<22} {r['cv_score']:>10.4f} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['f2']:>8.4f}\n")
    f.write("-"*80 + "\n\n")
    
    # Tabela sa F2 i Recall za različite pragove
    f.write("📊 UPOREDNI PRIKAZ F2 I RECALL ZA RAZLIČITE PRAGOVE:\n")
    f.write("="*80 + "\n")
    
    for name, r in results.items():
        f.write(f"\n{name}:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Prag':>8} {'F2':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}\n")
        f.write("-"*60 + "\n")
        
        for thresh in sorted(r["threshold_results"].keys()):
            m = r["threshold_results"][thresh]
            f.write(f"{thresh:>8.2f} {m['f2']:>10.4f} {m['recall']:>10.4f} "
                   f"{m['precision']:>10.4f} {m['f1']:>10.4f}\n")
        
        # Default prag
        f.write(f"{'-'*60}\n")
        f.write(f"{'0.50':>8} {r['f2']:>10.4f} {r['recall']:>10.4f} "
               f"{r['precision']:>10.4f} {r['f1']:>10.4f} (default)\n")
        
        # Najbolji prag
        f.write(f"\n🏆 NAJBOLJI PRAG: {r['best_threshold']}\n")
        f.write(f"   F2 = {r['best_f2']:.4f} (poboljšanje: {r['best_f2'] - r['f2']:+.4f})\n")
        f.write(f"   Recall = {r['best_recall']:.4f} (poboljšanje: {r['best_recall'] - r['recall']:+.4f})\n")
        f.write("-"*60 + "\n")
    
    # Rang lista po najboljem F2
    f.write("\n" + "="*80 + "\n")
    f.write("🏆 RANG LISTA (prema NAJBOLJEM F2 SKORU):\n")
    f.write("-"*40 + "\n")
    sorted_by_best_f2 = sorted(results.items(), key=lambda x: x[1]['best_f2'], reverse=True)
    for i, (name, r) in enumerate(sorted_by_best_f2, 1):
        f.write(f"{i}. {name}: Best_F2={r['best_f2']:.4f} (prag={r['best_threshold']}), "
               f"Best_Recall={r['best_recall']:.4f}, Default_F2={r['f2']:.4f}\n")

# ========== 7. GRAFIČKI PRIKAZ ==========
# 7a. F2 i Recall po pragovima
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, r) in enumerate(results.items()):
    if idx >= 6:
        break
    
    thresholds_list = sorted(r["threshold_results"].keys())
    f2_scores = [r["threshold_results"][t]["f2"] for t in thresholds_list]
    recall_scores = [r["threshold_results"][t]["recall"] for t in thresholds_list]
    
    # Dodajemo i default prag 0.5
    all_thresholds = thresholds_list + [0.5]
    all_f2 = f2_scores + [r["f2"]]
    all_recall = recall_scores + [r["recall"]]
    
    ax = axes[idx]
    
    # F2 linija
    ax.plot(all_thresholds, all_f2, 'o-', label='F2 Score', linewidth=2, color='blue', markersize=8)
    # Recall linija
    ax.plot(all_thresholds, all_recall, 's-', label='Recall', linewidth=2, color='red', markersize=8)
    
    # Označi najbolji F2
    best_idx = np.argmax(all_f2)
    best_thresh = all_thresholds[best_idx]
    best_f2 = all_f2[best_idx]
    best_recall = all_recall[best_idx]
    
    ax.plot(best_thresh, best_f2, 'o', color='green', markersize=12, 
            label=f'Best F2={best_f2:.3f}')
    ax.annotate(f'F2={best_f2:.3f}\nRecall={best_recall:.3f}', 
                xy=(best_thresh, best_f2),
                xytext=(5, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(f'{name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.25, 0.55)

# Sakrij prazne subplotove
for idx in range(len(results.items()), 6):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig("analysis/figures/f2_recall_by_threshold.png", dpi=300, bbox_inches='tight')
plt.show()

# 7b. Heatmap F2 po modelima i pragovima
thresholds_for_heatmap = [0.3, 0.35, 0.4, 0.5]
f2_matrix = []
recall_matrix = []

for name, r in results.items():
    f2_row = []
    recall_row = []
    for thresh in thresholds_for_heatmap:
        if thresh == 0.5:
            f2_row.append(r["f2"])
            recall_row.append(r["recall"])
        else:
            if thresh in r.get("threshold_results", {}):
                f2_row.append(r["threshold_results"][thresh]["f2"])
                recall_row.append(r["threshold_results"][thresh]["recall"])
            else:
                f2_row.append(np.nan)
                recall_row.append(np.nan)
    f2_matrix.append(f2_row)
    recall_matrix.append(recall_row)

# Heatmap za F2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

f2_df = pd.DataFrame(f2_matrix, 
                     index=[name for name in results.keys()],
                     columns=[f"Prag {t}" for t in thresholds_for_heatmap])

sns.heatmap(f2_df, annot=True, fmt='.4f', cmap='RdYlGn', 
            cbar_kws={'label': 'F2 Score'}, 
            linewidths=0.5, linecolor='black', ax=ax1)
ax1.set_title('F2 Skorovi po Modelima i Pragovima', fontsize=14, fontweight='bold')
ax1.set_xlabel('Prag odlučivanja', fontsize=12)
ax1.set_ylabel('Model', fontsize=12)

# Heatmap za Recall
recall_df = pd.DataFrame(recall_matrix, 
                         index=[name for name in results.keys()],
                         columns=[f"Prag {t}" for t in thresholds_for_heatmap])

sns.heatmap(recall_df, annot=True, fmt='.4f', cmap='Blues', 
            cbar_kws={'label': 'Recall Score'}, 
            linewidths=0.5, linecolor='black', ax=ax2)
ax2.set_title('Recall Skorovi po Modelima i Pragovima', fontsize=14, fontweight='bold')
ax2.set_xlabel('Prag odlučivanja', fontsize=12)
ax2.set_ylabel('Model', fontsize=12)

plt.tight_layout()
plt.savefig("analysis/figures/f2_recall_heatmaps.png", dpi=300, bbox_inches='tight')
plt.show()

# 7c. Bar plot - Poboljšanje F2 i Recall
fig, ax = plt.subplots(figsize=(12, 6))

models = list(results.keys())
f2_improvements = [r["best_f2"] - r["f2"] for r in results.values()]
recall_improvements = [r["best_recall"] - r["recall"] for r in results.values()]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, f2_improvements, width, label='F2 Poboljšanje', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, recall_improvements, width, label='Recall Poboljšanje', color='red', alpha=0.7)

ax.set_xlabel('Model')
ax.set_ylabel('Poboljšanje')
ax.set_title('Poboljšanje F2 i Recall pri optimalnom pragu vs default (0.5)')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# Dodaj vrednosti na vrh stubića
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom')

plt.tight_layout()
plt.savefig("analysis/figures/improvement_barplot.png", dpi=300, bbox_inches='tight')
plt.show()