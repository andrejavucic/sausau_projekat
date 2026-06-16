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

import warnings
warnings.filterwarnings('ignore')

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
 #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# KORELACIJA: stabla odlucivanja i ansambl metode su znatno otpornije na koleralciju
# jer kad stablo deli pod, samo ce u tom trenutku izabrati atribut koji bolje deli podatke

# n_jobs -> koliko procesorskoh jezgra koristimo (-1 -> sva jezgra)
models_config = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000,random_state=42, class_weight="balanced",
            #                        C=0.1,
             #                       penalty='l1',
              #                      solver='liblinear'
                                    ),
  #      "grid" : {},
        # pod grid spadaju oni parametri za koje cemo probati grid da vidimo sta je naj
        "grid": {
            # jacina regularizacije (C je suprotno lambda parametru)
            # 10 -> oslanja se minimizaciju greske (overfitting) 
            # ako gledamo naspram lambde, kada ga nemamo - ne kaznjavamo greske i tada moze doci do overfittinga
            # tkd malo lambda -> overfitting -> za C je samo obrnuto
            # 00.1 -> jace kaznjava velike koeficijente (underfitting) -> jednostavniji model
            # 
            "C": [0.01, 0.1, 1, 10], 

            # l1 -> LASSO REGURALIZACIJA-> regulator se zasniva na apsolutnoj vr koeficijenata
            # moze svesti koeficijente tacno na 0, pronadji koji atribut ne pomaze modelu i izabaci ga
            # l2 -> RIDGE REGURALIZACIJA > reg kaznjava velike koeficijente -> ()^2
            # smanjuje koeficijente, ali ih najcesce ne svodi na 0\
            # smanjuje se varijnsa
            "penalty": ["l1", "l2"],
            
            # nacin na koji se resava regularizacija
            # navodno neophodno zbog l1 regularizacije
            # oba nacina podrzavaju i l1 i l2
            "solver": ["liblinear", "saga"],
        },
    },
    "KNN": {
        "model": KNeighborsClassifier(n_jobs=-1, 
          #                              n_neighbors=19,
           #                             p=1,
            #                           weights='distance'
                                      ),
 #       "grid" : {},
        "grid": {
            # izbor za k - br suseda (umesto Elbow metode)
            # malo K -> osetljiv na sum (overfitting)
            # veliko K -> niska varijnsa, veliki bias (underfitting)
            # bias - koliko je model pojednostavio problem
            "n_neighbors": [3, 5, 7, 9, 11, 15, 19, 25],

            # tezina suseda:
            # uniform -> svi susedi iste tezine
            # distance -> blizi su znacajniji (imaju vecu tezinu)
            "weights": ["uniform", "distance"],

            # nacin kako racunamo rastojanje:
            # p=1 -> Manhattan -> ono sa apsolutnom vr
            # putanja samo ortogonalnim koracima (strelice || sa x i y-osama)
            # p=2 -> Euklidsko -> koren kvadrata
            # najbliza vazdusna putanja
            "p": [1, 2], 
        },
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42, class_weight="balanced",
              #                          # najbolji kriterijumi -> izabrani gridom
               #                         criterion='gini',
                #                        max_depth=5,
                 #                       min_samples_leaf=1,
                  #                      min_samples_split=2,
                                         ),
                                    
   #     "grid" : {},
        "grid": {
            #velika -> overfitting, mala-> underfitting
            "max_depth": [5, 10, 15, 20, None],
                # sled put treba staviti samo [4, 5, 6]
            
            #min br uzoraka za podatak
           "min_samples_split": [2, 5, 10, 20],
            # [15, 20, 25] (ostalo mozemo izostaviti)
            
            # min br uzoraka u listu
            "min_samples_leaf": [1, 2, 4, 8],

            #za merenje cistoce:
            # Gini: mera pomesanosti klasa u cvoru
            # Entropija: mera neuredjenosti(nestigurnosti) u cvoru
            # max pomesanost 50%/50%, max uredjenost 100%/0%
            "criterion": ["gini", "entropy"],

            # ima jos i max_features - max br uzoraka za analizu

            # daje iste rez za oba
#            "class_weight": [None, "balanced"] 
        },       
    },
    # =========================== ANSAMBL METODE ============================

    # BAGGING METODA: vise razliciih stabala koji se treniraju nezavinsno, uz uvodjenje slucajnosti
    # slucajnost u izboru pod i atributa - cini stabla medjusobno razlicitim
    # smanjenje varijanse, osetljiv na hiperparametre, umanjuje uticaj jednog singularnog stabla
    # kod klasifikacije -> svako stablo glasa za 1 klasu -> vecinska se bira
     "Random Forest": { 
        "model": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced",
               #                         n_estimators=50,
                #                        max_depth=15,
                 #                       min_samples_split=2,
                  #                      min_samples_leaf=4,
                   #                     max_features='log2'
                                        ),
 #       "grid" : {},
        "grid": {
            # veci broj stabala = stabilniji model, ali sporiji
            "n_estimators": [50, 100, 200],
            
            "max_depth": [5, 10, 15, None],
            
            # min broj uzoraka za podelu cvora
            "min_samples_split": [2, 5, 10],
            
            # min broj uzoraka u listu
            "min_samples_leaf": [1, 2, 4],
            
            # br atributa koje razmatra za najbolju podelu (max br uzoraka za analizu)
            # manje atributa -> veca sansa da razlicita stabla biraju razlicite podele
            # 'sqrt' = sqrt(broj_atributa) - dobro za visoke dimenzije 
            # npr: 100 atributa -> sqrt(100) = 10 -> uzme 10 atributa 
            # 'log2' = log2(broj_atributa)
            # log(100) ~ 6,7 atributa 
            # None = svi atributi (moramo da poredimo sve atribute)
           "max_features": ['sqrt', 'log2', None],
            
            # OVO NISAM NI TESTIRALA (zbog vremena)
            # da li koristiti bootstrap uzorkovanje
            # uzorak sa vracanjem ili trening na celom skupu
            # sa vracanjem -> neki uzorci se ponavljaju, neki nisu nikad uzeti (mogu da se koriste kao free validacija)
            # npr: sqrt(100)=10, i bootstrap=true, on uzme najbolju komb atributa od tih 10 (mogu se ponavljati neki, ili da ih nema)
            # celo stablo -> stabla su slicinija, jedino max_features pravi razliku koji atributi se razmatraju
    #        "bootstrap": [True, False]
        },
    },
    
    # BOOSTING METODA: modeli se treniraju sekvencijalno, svaki sled popravlja greske prethonog
    # postepeno se smanjuje greska
    "Gradient Boosting": {       #NISAM TESTIRALA PARAMETRE -> PREDUGO
        "model": GradientBoostingClassifier(random_state=42,
#                                            n_estimators=100,
#                                            learning_rate=0.1,
#                                            max_depth=3,
#                                            min_samples_split=2,
#                                            min_samples_leaf=1,
#                                            subsample=0.8
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
    # predlog da ne radimo svm -> previse vremena oduzima, teze interpretabilan -> vrv nema veliku prednost ovde
#   "SVM": {
#        "model": SVC(probability=True, random_state=42, class_weight="balanced"),
#        "grid" : {},
#        "grid": {
            # koliko model kaznjava greske (slicno kao lambda kod regresije)
            # manja kazna -> sira margina, vise dozvoljenih gresaka -> underfitting
            # veca kazna -> uza margina, manje dozvoljenih gresaka -> overfitting
            #"C": [1, 10], 

            # f- ja koja omogucava SVM da formira nelinearnu granicu odlucivanja
            # podaci se posmatraju u drugacijem prostoru u kom ih je lakse razdvojiti
            # linear - linearna gr odlucivanja
            # poly - polinomijalna
            # rbf - fleksibilna nelinearna gr odlucivanja
          #  "kernel": ["linear", "rbf", "poly"],

            #fleksibilnost granice kod RBF kernela
            # radijus uticaja jedne tacke 
           # "gamma": ["scale", "auto", 0.1, 1],
#       },
#    },
}

# ========== 3. GridSearch + evaluacija ==========
results = {}

for name, cfg in models_config.items():
    # GRID SEARCH
    # prvi primer sa K-fold unakrsnom validacijom 
    # F1 - metrika po kojoj biram modele (harmonijska sredina)
    #gs = GridSearchCV(cfg["model"], cfg["grid"], cv=cv, scoring="f1", n_jobs=-1, verbose=0)
    #gs = GridSearchCV(cfg["model"], cfg["grid"], scoring="f1", n_jobs=-1, verbose=0)

    best_score = 0
    best_params = {}
    best_model = None
    print()

    # pravimo sve kombinacije parametara
    param_grid= ParameterGrid(cfg["grid"])
    total = len(param_grid)
    
    print(f"Testiram {total} kombinacija parametara...")
    
    for i, params in enumerate(param_grid):
        # Kreiraj model sa trenutnim parametrima
        model = cfg["model"].set_params(**params)
        
        # Treniraj na TRENING skupu
        model.fit(X_train, y_train)
        
        # Evaluacija na VALIDACIONOM skupu
        y_val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)
        
        # Ako je bolji, zapamti ga
        if val_f1 > best_score:
            best_score = val_f1
            best_params = params
            best_model = model
        
        # Ispis svakih 20 kombinacija (da ne zatrpa terminal)
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [{i+1:3d}/{total}] Trenutno najbolji F1: {best_score:.4f}")


    print(f"\n NAJBOLJI PARAMETRI: {best_params}")
    print(f" VALIDATION F1: {best_score:.4f}")

    # na najboljem modelu radimo X_test
    print(f"\nEvaluacija na TEST skupu...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": best_model,
        "best_params": best_params,
        "val_f1": best_score,
        #"cv_score": gs.best_score_,
        "accuracy":  accuracy_score(y_test, y_pred),    # udeo tacnih pred

        # predvidimo da hoce da se pretplat, a zapravo nece -> banka gubi vreme da ga opet zove (nista spec)
        "precision": precision_score(y_test, y_pred),   # FP (lazno pozitivne pred skupe)
        
        # predvidimo da nece da se pretplate, a hoce -> veliki propust za banku
        "recall":    recall_score(y_test, y_pred),      # FN (kada je opasno propustiti pozitivan slucaj)
        
        "f1":        f1_score(y_test, y_pred),          # harmonijska sredina
        "roc_auc":   roc_auc_score(y_test, y_prob),     # kriva za prikazivanje ponasanja modela korz razlicite pragove odlucvanja
                                                        # prag -> utice na presicion i recall (sa kojom vr uzimamo podatke kao P(validne))
        "cm":        confusion_matrix(y_test, y_pred),
    }
    print(f"  Test F1: {results[name]['f1']:.4f}")
    print(f"  Test Recall: {results[name]['recall']:.4f}")
    print(f"  Test ROC-AUC: {results[name]['roc_auc']:.4f}")


# ========== 5. ČUVANJE MODELA I REZULTATA ==========
# 5a. Čuvanje modela
for name, r in results.items():
    fname = f"models/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(r["model"], fname)

# Čuvanje najboljeg modela
# prema val skupu
best_val_f1 = 0
best_name = ""
best_model_obj = None
for name, r in results.items():
    if r['val_f1'] > best_val_f1:
        best_val_f1 = r['val_f1']
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
        "Val_F1": round(r["val_f1"], 4),
        "Accuracy": round(r["accuracy"], 4), 
        "Precision": round(r["precision"], 4),
        "Recall": round(r["recall"], 4), 
        "F1": round(r["f1"], 4), 
        "ROC-AUC": round(r["roc_auc"], 4)
    }
    for n, r in results.items()
])

# Sortiraj po F1 (bolji prvi)
metrics_df = metrics_df.sort_values("F1", ascending=False)
metrics_df.to_csv("analysis/metrics_comparison_2.csv", index=False)

# 5c. Čuvanje TXT tabele (za brzi pregled)
with open("analysis/metrics_table.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("BANK MARKETING - REZULTATI KLASIFIKACIJE\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Model':<22} {'Val F1':>10} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ROC':>10}\n")
    f.write("-"*80 + "\n")
    for name, r in results.items():
        f.write(f"{name:<22} {r['val_f1']:>10.4f} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>10.4f}\n")
    f.write("-"*80 + "\n\n")
    
    # Rang lista
    f.write("RANG LISTA (prema F1 skoru):\n")
    f.write("-"*40 + "\n")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (name, r) in enumerate(sorted_results, 1):
        f.write(f"{i}. {name}: F1={r['f1']:.4f}, Recall={r['recall']:.4f}\n")

# 5d. Kratak ispis u terminalu
print("\n" + "-"*50)
print("RANG LISTA (prema F1):")
for i, (name, r) in enumerate(sorted_results, 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"  {i}."
    print(f"{medal} {name}: F1={r['f1']:.4f}, Recall={r['recall']:.4f}")
print()