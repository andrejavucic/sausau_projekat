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

#X_val = np.load("data/processed/X_val_preprocessed.npy")   
#y_val = np.load("data/processed/y_val.npy") 
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

    print("\n" + "-"*50)
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

# Sortiraj po F2 (bolji prvi)
metrics_df = metrics_df.sort_values("F2", ascending=False)
#metrics_df.to_csv("analysis/metrics_comparison.csv", index=False)

# 5c. Čuvanje TXT tabele (za brzi pregled)
sorted_results = sorted(results.items(), key=lambda x: x[1]['f2'], reverse=True)

with open("analysis/metrics_table.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("BANK MARKETING - REZULTATI KLASIFIKACIJE\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Model':<22} {'F2':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'ROC-AUC':>10}\n") 
    f.write("-"*80 + "\n")
    for name, r in results.items():
        f.write(f"{name:<22} {r['f2']:>10.4f} {r['recall']:>10.4f} "
                f"{r['precision']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}\n")
    f.write("-"*80 + "\n\n")
    
    # Rang lista
    f.write("RANG LISTA (prema F2 skoru):\n")
    f.write("-"*40 + "\n")
    for i, (name, r) in enumerate(sorted_results, 1):
        f.write(f"{i}. {name}: F2={r['f2']:.4f}, Recall={r['recall']:.4f}\n")

# 5d. Kratak ispis u terminalu
print("\n" + "-"*50)
print("RANG LISTA (prema F2):")
for i, (name, r) in enumerate(sorted_results, 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"  {i}."
    print(f"{medal} {name}: F2={r['f2']:.4f}, Recall={r['recall']:.4f}")
print()