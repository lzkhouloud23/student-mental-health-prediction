# ============================================================
#  PROJET ML - Détection précoce du burnout / anxiété chez les étudiants
#  Enseignante : Dr. N. OUERHANI | 4 ING GL 2025-2026
#  Dataset réel : Student Mental Health (Kaggle)
# ============================================================
# Ce fichier fait TOUT :
#   1. Charge les données réelles (Student_Mental_health.csv)
#   2. Analyse exploratoire (EDA)
#   3. Prétraitement
#   4. Entraîne 2 modèles (Logistic Regression + Random Forest)
#   5. Évalue et compare
#   6. Sauvegarde le meilleur modèle pour l'appli web
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
import joblib
import os

os.makedirs("outputs", exist_ok=True)

# ============================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES RÉELLES
# ============================================================
print("=" * 60)
print("  ÉTAPE 1 : Chargement des données réelles")
print("=" * 60)

df = pd.read_csv("Student_Mental_health.csv")
# Renommer les colonnes pour simplifier
df.columns = [
    "timestamp", "gender", "age", "course", "year",
    "cgpa", "married", "depression", "anxiety", "panic_attack", "treatment"
]

print(f"✅ Dataset chargé : {len(df)} étudiants, {len(df.columns)} colonnes")
print(df.head())


# ============================================================
# ÉTAPE 2 : ANALYSE EXPLORATOIRE (EDA)
# ============================================================
print("\n" + "=" * 60)
print("  ÉTAPE 2 : Analyse exploratoire (EDA)")
print("=" * 60)

print("\n📊 Types et valeurs manquantes :")
print(df.info())
print(f"\n❓ Valeurs manquantes par colonne :\n{df.isnull().sum()}")

print("\n📈 Distribution de l'anxiété (variable cible) :")
print(df["anxiety"].value_counts())

# --- Graphiques EDA ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Analyse Exploratoire — Santé Mentale des Étudiants", fontsize=15, y=1.01)

# 1. Distribution anxiété
axes[0, 0].bar(["Non", "Oui"],
               df["anxiety"].value_counts().reindex(["No", "Yes"]).values,
               color=["#2ecc71", "#e74c3c"], edgecolor="white")
axes[0, 0].set_title("Prévalence de l'anxiété")
axes[0, 0].set_ylabel("Nombre d'étudiants")

# 2. Genre vs Anxiété
anxiety_gender = df.groupby(["gender", "anxiety"]).size().unstack(fill_value=0)
anxiety_gender.plot(kind="bar", ax=axes[0, 1], color=["#2ecc71", "#e74c3c"],
                    edgecolor="white", rot=0)
axes[0, 1].set_title("Anxiété par genre")
axes[0, 1].set_xlabel("")
axes[0, 1].legend(["Non", "Oui"], title="Anxiété")

# 3. Année d'étude vs Anxiété
year_order = ["year 1", "year 2", "year 3", "year 4"]
df["year_clean"] = df["year"].str.lower().str.strip()
anxiety_year = df.groupby(["year_clean", "anxiety"]).size().unstack(fill_value=0)
anxiety_year = anxiety_year.reindex([y for y in year_order if y in anxiety_year.index])
anxiety_year.plot(kind="bar", ax=axes[0, 2], color=["#2ecc71", "#e74c3c"],
                  edgecolor="white", rot=0)
axes[0, 2].set_title("Anxiété par année d'étude")
axes[0, 2].set_xlabel("")
axes[0, 2].legend(["Non", "Oui"], title="Anxiété")

# 4. CGPA vs Anxiété
cgpa_order = ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"]
df["cgpa_clean"] = df["cgpa"].str.strip()
anxiety_cgpa = df.groupby(["cgpa_clean", "anxiety"]).size().unstack(fill_value=0)
anxiety_cgpa = anxiety_cgpa.reindex([c for c in cgpa_order if c in anxiety_cgpa.index])
anxiety_cgpa.plot(kind="bar", ax=axes[1, 0], color=["#2ecc71", "#e74c3c"],
                  edgecolor="white", rot=15)
axes[1, 0].set_title("Anxiété par CGPA")
axes[1, 0].set_xlabel("")
axes[1, 0].legend(["Non", "Oui"], title="Anxiété")

# 5. Co-occurrence : Dépression, Anxiété, Attaque panique
venn_data = {
    "Dépression": (df["depression"] == "Yes").sum(),
    "Anxiété":    (df["anxiety"] == "Yes").sum(),
    "Panique":    (df["panic_attack"] == "Yes").sum(),
}
axes[1, 1].bar(venn_data.keys(), venn_data.values(),
               color=["#9b59b6", "#e74c3c", "#f39c12"], edgecolor="white")
axes[1, 1].set_title("Prévalence par condition mentale")
axes[1, 1].set_ylabel("Nombre d'étudiants")

# 6. Traitement selon anxiété
treat_anx = df.groupby(["anxiety", "treatment"]).size().unstack(fill_value=0)
treat_anx.plot(kind="bar", ax=axes[1, 2], color=["#e74c3c", "#2ecc71"],
               edgecolor="white", rot=0)
axes[1, 2].set_title("Traitement selon anxiété")
axes[1, 2].set_xlabel("Anxiété")
axes[1, 2].legend(["Non traité", "Traité"], title="Traitement")

plt.tight_layout()
plt.savefig("outputs/eda_graphiques.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅ Graphiques EDA sauvegardés : outputs/eda_graphiques.png")

# ============================================================
# ÉTAPE 3 : PRÉTRAITEMENT
# ============================================================
print("\n" + "=" * 60)
print("  ÉTAPE 3 : Prétraitement des données")
print("=" * 60)

df_model = df.copy()

# Supprimer colonnes inutiles
df_model.drop(columns=["timestamp"], inplace=True)

# Nettoyer et normaliser
df_model["year"] = df_model["year"].str.lower().str.strip()
df_model["cgpa"] = df_model["cgpa"].str.strip()

# Gérer les valeurs manquantes
df_model["age"].fillna(df_model["age"].median(), inplace=True)
df_model.dropna(subset=["gender", "cgpa", "year"], inplace=True)

print(f"📊 Lignes après nettoyage : {len(df_model)}")
# --- BLOC DE NETTOYAGE D'URGENCE ---
# 1. Nettoyer la colonne 'year' (Transformer "year 1" en 1)
df['year'] = df['year'].str.extract('(\d+)').astype(float)

# 2. Nettoyer la colonne 'cgpa' (Prendre la moyenne de la fourchette, ex: "3.00 - 3.49" -> 3.25)
def clean_cgpa(x):
    try:
        if '-' in str(x):
            parts = str(x).split('-')
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        return float(x)
    except:
        return 0.0

df['cgpa'] = df['cgpa'].apply(clean_cgpa)

# 3. Remplir les valeurs manquantes pour l'âge
df['age'] = df['age'].fillna(df['age'].median())
# ----------------------------------

# Encoder les variables catégorielles
binary_cols = ["gender", "married", "depression", "anxiety", "panic_attack", "treatment"]
for col in binary_cols:
    df_model[col] = df_model[col].map({"Yes": 1, "No": 0,
                                        "Female": 0, "Male": 1})

# Encoder CGPA (ordinal)
cgpa_map = {
    "0 - 1.99": 0, "2.00 - 2.49": 1, "2.50 - 2.99": 2,
    "3.00 - 3.49": 3, "3.50 - 4.00": 4
}
df_model["cgpa"] = df_model["cgpa"].map(cgpa_map)
df_model["cgpa"].fillna(df_model["cgpa"].median(), inplace=True)

# Encoder l'année d'étude (ordinal)
year_map = {"year 1": 1, "year 2": 2, "year 3": 3, "year 4": 4}
df_model["year"] = df_model["year"].map(year_map)
df_model["year"].fillna(1, inplace=True)

# Encoder le cours (label encoding)
le_course = LabelEncoder()
df_model["course"] = df_model["course"].str.lower().str.strip()
df_model["course"] = le_course.fit_transform(df_model["course"])

# Sauvegarder l'encodeur de cours
joblib.dump(le_course, "outputs/course_encoder.pkl")

print("✅ Encodage des variables catégorielles terminé")
print(df_model.dtypes)

# Séparer X et y
# On prédit l'ANXIÉTÉ comme variable cible principale
X = df_model.drop(columns=["anxiety"])
y = df_model["anxiety"]

print(f"\n📊 Distribution de la cible :\n{y.value_counts()}")

# ============================================================
# ÉTAPE 3 (SUITE) : PRÉPARATION FINALE DES DONNÉES
# ============================================================

# 1. Ne garder QUE les colonnes numériques (pour éviter l'erreur 'year 1')
X = df_model.select_dtypes(include=['int64', 'float64'])

# 2. Définir la cible (y) et l'enlever de X
y = df_model["anxiety"]
if 'anxiety' in X.columns:
    X = X.drop(columns=['anxiety'])

# 3. BOUCHER LES TROUS (NaN) par 0 (pour éviter l'erreur Input X contains NaN)
X = X.fillna(0)

print(f"✅ Colonnes utilisées pour l'entraînement : {list(X.columns)}")

# 4. Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split train/test (stratifié)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Split terminé : {len(X_train)} train / {len(X_test)} test")

# 6. SAUVEGARDE CRUCIALE pour l'application Web
import os
if not os.path.exists('outputs'): os.makedirs('outputs')
joblib.dump(list(X.columns), "outputs/feature_names.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")

# ============================================================
# ÉTAPE 4 : ENTRAÎNEMENT DES MODÈLES
# ============================================================
print("\n" + "=" * 60)
print("  ÉTAPE 4 : Entraînement des modèles")
print("=" * 60)

# --- Modèle 1 : Régression Logistique ---
print("\n🔵 Modèle 1 : Régression Logistique")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
cv_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring="accuracy").mean()
print(f"   Accuracy Test  : {acc_lr:.4f} ({acc_lr*100:.1f}%)")
print(f"   Cross-Val (5x) : {cv_lr:.4f} ({cv_lr*100:.1f}%)")

# --- Modèle 2 : Random Forest ---
print("\n🟢 Modèle 2 : Random Forest")
rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42,
                             class_weight="balanced")
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
cv_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy").mean()
print(f"   Accuracy Test  : {acc_rf:.4f} ({acc_rf*100:.1f}%)")
print(f"   Cross-Val (5x) : {cv_rf:.4f} ({cv_rf*100:.1f}%)")

# ============================================================
# ÉTAPE 5 : ÉVALUATION COMPARATIVE
# ============================================================
print("\n" + "=" * 60)
print("  ÉTAPE 5 : Évaluation et comparaison")
print("=" * 60)

print("\n--- Rapport Régression Logistique ---")
print(classification_report(y_test, y_pred_lr, target_names=["Non anxieux", "Anxieux"]))

print("\n--- Rapport Random Forest ---")
print(classification_report(y_test, y_pred_rf, target_names=["Non anxieux", "Anxieux"]))

# Graphiques de comparaison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Comparaison des Modèles — Prédiction d'Anxiété", fontsize=14)

# Matrice de confusion - LR
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', ax=axes[0], cmap="Blues",
            xticklabels=["Non", "Oui"], yticklabels=["Non", "Oui"])
axes[0].set_title(f"Régression Logistique\nAccuracy: {acc_lr*100:.1f}%")
axes[0].set_ylabel("Réel")
axes[0].set_xlabel("Prédit")

# Matrice de confusion - RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[1], cmap="Greens",
            xticklabels=["Non", "Oui"], yticklabels=["Non", "Oui"])
axes[1].set_title(f"Random Forest\nAccuracy: {acc_rf*100:.1f}%")
axes[1].set_ylabel("Réel")
axes[1].set_xlabel("Prédit")

# Importance des features
importances = rf.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importances)[::-1]
axes[2].barh(range(len(feature_names)),
             importances[sorted_idx],
             color="#3498db", alpha=0.8)
axes[2].set_yticks(range(len(feature_names)))
axes[2].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
axes[2].set_title("Importance des variables\n(Random Forest)")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("outputs/comparaison_modeles.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅ Graphiques sauvegardés : outputs/comparaison_modeles.png")

# ============================================================
# ÉTAPE 6 : SAUVEGARDE DU MEILLEUR MODÈLE
# ============================================================
print("\n" + "=" * 60)
print("  ÉTAPE 6 : Sauvegarde")
print("=" * 60)

best_model = rf if acc_rf >= acc_lr else lr
best_name = "Random Forest" if acc_rf >= acc_lr else "Régression Logistique"
print(f"🏆 Meilleur modèle : {best_name} ({max(acc_rf, acc_lr)*100:.1f}%)")

joblib.dump(best_model, "outputs/best_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")

# Sauvegarder les métriques pour l'app web
import json
metrics = {
    "logistic_regression": {"accuracy": round(acc_lr * 100, 1), "cv": round(cv_lr * 100, 1)},
    "random_forest":        {"accuracy": round(acc_rf * 100, 1), "cv": round(cv_rf * 100, 1)},
    "best_model": best_name,
    "n_samples": len(df_model),
    "anxiety_prevalence": round((y == 1).sum() / len(y) * 100, 1)
}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f)

print("💾 Modèle   : outputs/best_model.pkl")
print("💾 Scaler   : outputs/scaler.pkl")
print("💾 Métriques: outputs/metrics.json")

print("\n" + "=" * 60)
print("  ✅ PROJET TERMINÉ AVEC SUCCÈS !")
print(f"  Régression Logistique : {acc_lr*100:.1f}%")
print(f"  Random Forest         : {acc_rf*100:.1f}%")
print(f"  Meilleur modèle       : {best_name}")
print("=" * 60)