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

import warnings
warnings.filterwarnings('ignore')

os.makedirs("models", exist_ok=True)
os.makedirs("models/figures", exist_ok=True)
sns.set_style("whitegrid")

# ========== 1. Učitavanje ==========
X_train = np.load("data/processed/X_train_preprocessed.npy")
y_train = np.load("data/processed/y_train.npy")
X_test  = np.load("data/processed/X_test_preprocessed.npy")
y_test  = np.load("data/processed/y_test.npy")

# ========== 2. Modeli + gridovi za hiperparametre ==========
# K - fold unakrsna validacija, k=5 -> deli na 5 delova
# 5 puta se prolazi kroz trening, i na kraju se bira najbolji
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# n_jobs -> koliko procesorskoh jezgra koristimo (-1 -> sva jezgra)
models_config = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42,
                                    C=0.1,
                                    penalty='l1',
                                    solver='liblinear'),
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
                                        n_neighbors=5,
                                        p=2,
                                        weights='uniform'),
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
        "model": DecisionTreeClassifier(random_state=42, class_weight="balanced",
                                        # najbolji kriterijumi -> izabrani gridom
                                        criterion='gini',
                                        max_depth=5,
                                        min_samples_leaf=4,
                                        min_samples_split=20,),
                                    
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
#        },       
#    },
#    "SVM": {
#        "model": SVC(probability=True, random_state=42, class_weight="balanced"),
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
        },
#    },
}

# ========== 3. GridSearch + evaluacija ==========
results = {}

for name, cfg in models_config.items():
    # GRID SEARCH
    # prvi primer sa K-fold unakrsnom validacijom 
    # F1 - metrika po kojoj biram modele (harmonijska sredina)
    #gs = GridSearchCV(cfg["model"], cfg["grid"], cv=cv, scoring="f1", n_jobs=-1, verbose=0)
    gs = GridSearchCV(cfg["model"], cfg["grid"], scoring="f1", n_jobs=-1, verbose=0)

    gs.fit(X_train, y_train)    # pronadji parametre koji najbolje odgovraju podacima za obucavanje

    # na najboljem modelu radimo X_test
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": best_model,
        "best_params": gs.best_params_,
        #"cv_score": gs.best_score_,
        "accuracy":  accuracy_score(y_test, y_pred),    # udeo tacnih pred
        "precision": precision_score(y_test, y_pred),   # FP (lazno pozitivne pred skupe)
        "recall":    recall_score(y_test, y_pred),      # FN (kada je opasno propustiti pozitivan slucaj)
        "f1":        f1_score(y_test, y_pred),          # harmonijska sredina
        "roc_auc":   roc_auc_score(y_test, y_prob),     # kriva za prikazivanje ponasanja modela korz razlicite pragove odlucvanja
                                                        # prag -> utice na presicion i recall (sa kojom vr uzimamo podatke kao P(validne))
        "cm":        confusion_matrix(y_test, y_pred),
    }
    print(f"  Najbolji param: {gs.best_params_}")
    #print(f"  CV F1: {gs.best_score_:.4f}  |  Test ROC-AUC: {results[name]['roc_auc']:.4f}\n")

# ========== 4. Pregledna tabela ==========
print("=" * 70)
print(f"{'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ROC':>7}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<20} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['roc_auc']:>7.4f}")
print()

# ========== 5. Čuvanje ==========
for name, r in results.items():
    fname = f"models/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(r["model"], fname)
   

pd.DataFrame([
   # {"Model": n, "Best params": str(r["best_params"]), "CV F1": f"{r['cv_score']:.4f}",
   #  "Accuracy": r["accuracy"], "Precision": r["precision"],
   #  "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["roc_auc"]}
     {"Model": n, "Best params": str(r["best_params"]),
     "Accuracy": r["accuracy"], "Precision": r["precision"],
     "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["roc_auc"]}
    for n, r in results.items()
]).to_csv("models/metrics_comparison.csv", index=False)

# ========== 6. Vizuelna interpretacija ==========
# --- 6a. PCA 2D granice odluke ---
# originalne atribute predstavi pomocu novih osa
# uzima sve nase atribute i skalira ih u 2 ose (tako da izgubi sto manje info) -> 2D vizuelizacija
num_models = len(results)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train)

x_min, x_max = X_pca[:, 0].min()-1, X_pca[:, 0].max()+1
y_min, y_max = X_pca[:, 1].min()-1, X_pca[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Dinamički broj redova za subplot (2 po redu)
ncols = 2
nrows = (num_models + 1) // 2  # Zaokruživanje naviše

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
# Ako je samo jedan model, axes treba da bude lista
if num_models == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, (name, r) in enumerate(results.items()):
    ax = axes[idx]
    m2d = type(r["model"])()
    
    # Filtriramo samo parametre koje model podržava
    valid_params = {k: v for k, v in r["best_params"].items() 
                    if k in m2d.get_params()}
    m2d.set_params(**valid_params)
    
    # RandomState dodajemo SAMO ako model podržava
    if "random_state" in m2d.get_params():
        m2d.set_params(random_state=42)
    
    m2d.fit(X_pca, y_train)
    Z = m2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.15, cmap="RdBu")
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap="RdBu", alpha=0.4, s=5)
    ax.set_title(f"{name}\nAcc={r['accuracy']:.3f}  F1={r['f1']:.3f}", fontweight="bold")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")

# Sakrij prazne subplotove ako ih ima
for idx in range(len(results), nrows * ncols):
    axes[idx].set_visible(False)

fig.suptitle("Granice odluke na PCA (2D) projekciji", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("models/figures/decision_boundaries_pca.png", dpi=150)
plt.close()

# --- 6b. Confusion matrice ---
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
                xticklabels=["Ne", "Da"], yticklabels=["Ne", "Da"])
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Predviđeno"); ax.set_ylabel("Stvarno")

# Sakrij prazne subplotove
for idx in range(len(results), nrows * ncols):
    axes[idx].set_visible(False)

fig.suptitle("Confusion matrice", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("models/figures/confusion_matrices.png", dpi=150)
plt.close()

print("\nGotovo!")