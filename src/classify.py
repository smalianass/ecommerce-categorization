# ============================================================
# 1. IMPORTS
# ============================================================

import os
import ast
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
NOTEBOOKS_DIR = os.path.join(PROJECT_DIR, "notebooks")

CONFIDENCE_CSV = os.path.join(
    INPUT_DIR,
    "dataset_confiance_final_pour_classification.csv"
)

# CSV contenant les embeddings déjà calculés
EMBEDDINGS_CSV = os.path.join(
    OUTPUT_DIR,
    "embeddings_produits_uniques.csv"
)

# Dataset complet à classifier
FULL_DATASET_CSV = os.path.join(
    NOTEBOOKS_DIR,
    "résultats.csv"
)

RESULTS_DIR = os.path.join(
    OUTPUT_DIR,
    "classification_results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "linear_svm_final.joblib"
)

LABEL_ENCODER_PATH = os.path.join(
    RESULTS_DIR,
    "label_encoder_final.joblib"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "resultats_final_classification.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 3. CHARGEMENT DATASET DE CONFIANCE
# ============================================================

print("Chargement du dataset de confiance...")

df_conf = pd.read_csv(
    CONFIDENCE_CSV,
    low_memory=False
)

print(f"Nombre de lignes : {len(df_conf):,}")
print(f"Nombre de colonnes : {len(df_conf.columns)}")


# ============================================================
# 4. VERIFICATION DES 586 CLASSES
# ============================================================

df_conf["Nature"] = (
    df_conf["Nature"]
    .astype(str)
    .str.strip()
)

classes = sorted(df_conf["Nature"].dropna().unique())

print(f"\nNombre de classes : {len(classes)}")

if len(classes) != 586:
    raise ValueError(
        f"ERREUR : le dataset contient {len(classes)} classes "
        f"au lieu des 586 classes attendues."
    )

print("✓ Les 586 classes valides sont présentes.")


# ============================================================
# 5. NETTOYAGE DU TEXTE
# ============================================================

df_conf["Libellé produit"] = (
    df_conf["Libellé produit"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df_conf = df_conf[
    df_conf["Libellé produit"] != ""
].copy()

print(
    f"Lignes après nettoyage : {len(df_conf):,}"
)


# ============================================================
# 6. CHARGEMENT DES EMBEDDINGS EXISTANTS
# ============================================================

print("\nChargement des embeddings existants...")

df_embeddings = pd.read_csv(
    EMBEDDINGS_CSV,
    low_memory=False
)

print(
    f"Produits uniques avec embeddings : "
    f"{len(df_embeddings):,}"
)

print(df_embeddings.columns.tolist())


# ============================================================
# 7. PARSING DES EMBEDDINGS
# ============================================================

def parse_embedding(value):

    if isinstance(value, np.ndarray):
        return value.astype(np.float32)

    if isinstance(value, list):
        return np.asarray(
            value,
            dtype=np.float32
        )

    return np.asarray(
        ast.literal_eval(str(value)),
        dtype=np.float32
    )


df_embeddings["Embedding"] = (
    df_embeddings["Embedding"]
    .apply(parse_embedding)
)


# ============================================================
# 8. VERIFICATION DIMENSION EMBEDDING
# ============================================================

embedding_dim = len(
    df_embeddings["Embedding"].iloc[0]
)

print(
    f"Dimension des embeddings : {embedding_dim}"
)

if embedding_dim != 1024:
    raise ValueError(
        f"Dimension inattendue : {embedding_dim}. "
        f"BGE-M3 attendu : 1024."
    )


# ============================================================
# 9. CREATION DU DICTIONNAIRE
# ============================================================

print("\nCréation du mapping produit → embedding...")

embedding_map = dict(
    zip(
        df_embeddings["Libellé produit"],
        df_embeddings["Embedding"]
    )
)

print(
    f"Embeddings disponibles : "
    f"{len(embedding_map):,}"
)


# ============================================================
# 10. RECUPERATION DES EMBEDDINGS DU DATASET DE CONFIANCE
# ============================================================

print("\nAssociation des embeddings au dataset de confiance...")

df_conf["Embedding"] = (
    df_conf["Libellé produit"]
    .map(embedding_map)
)

missing_conf = df_conf["Embedding"].isna().sum()

print(
    f"Embeddings manquants : {missing_conf:,}"
)

if missing_conf > 0:

    print(
        "\nExemples de produits sans embedding :"
    )

    print(
        df_conf.loc[
            df_conf["Embedding"].isna(),
            "Libellé produit"
        ]
        .drop_duplicates()
        .head(20)
        .to_list()
    )

    raise ValueError(
        "Certains produits du dataset de confiance "
        "n'ont pas d'embedding sauvegardé."
    )


# ============================================================
# 11. CONSTRUCTION DE X ET y
# ============================================================

X = np.vstack(
    df_conf["Embedding"].values
).astype(np.float32)

y_text = df_conf["Nature"].values


print(
    f"\nX : {X.shape}"
)
print(
    f"y : {y_text.shape}"
)


# ============================================================
# 12. ENCODAGE DES LABELS
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_text
)

print(
    f"Nombre de labels encodés : "
    f"{len(label_encoder.classes_)}"
)

if len(label_encoder.classes_) != 586:
    raise ValueError(
        "Le LabelEncoder ne contient pas les 586 classes."
    )


# ============================================================
# 13. DISTRIBUTION DES CLASSES
# ============================================================

class_counts = (
    df_conf["Nature"]
    .value_counts()
)

print("\nDistribution des classes :")
print(
    class_counts.describe()
)


# ============================================================
# 14. SPLIT TRAIN / TEST
# ============================================================

# Les classes avec une seule observation
# doivent rester dans le train.

singletons = class_counts[
    class_counts < 2
].index

multi_classes = class_counts[
    class_counts >= 2
].index

print(
    f"\nClasses avec >= 2 exemples : "
    f"{len(multi_classes)}"
)

print(
    f"Classes singleton : "
    f"{len(singletons)}"
)


# Indices des classes pouvant être stratifiées

mask_multi = df_conf["Nature"].isin(
    multi_classes
)

indices_multi = np.where(
    mask_multi.values
)[0]

indices_single = np.where(
    ~mask_multi.values
)[0]


# Split uniquement des classes ayant >= 2 exemples

train_idx_multi, test_idx_multi = train_test_split(
    indices_multi,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y[indices_multi]
)


# Les singletons vont uniquement dans le train

train_idx = np.concatenate(
    [
        train_idx_multi,
        indices_single
    ]
)

test_idx = test_idx_multi


X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]


print("\nSplit final :")
print(
    f"Train : {X_train.shape}"
)
print(
    f"Test  : {X_test.shape}"
)

print(
    f"Labels train : "
    f"{len(np.unique(y_train))}"
)

print(
    f"Labels test : "
    f"{len(np.unique(y_test))}"
)


# ============================================================
# 15. ENTRAINEMENT LINEAR SVM
# ============================================================

print("\nEntraînement LinearSVC...")

clf = LinearSVC(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    max_iter=3000
)

clf.fit(
    X_train,
    y_train
)

print("✓ Modèle entraîné.")


# ============================================================
# 16. EVALUATION
# ============================================================

y_pred = clf.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("RESULTATS LINEAR SVM")
print("=" * 60)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy  : {balanced_accuracy:.4f}"
)

print(
    f"Precision Macro    : {precision_macro:.4f}"
)

print(
    f"Recall Macro       : {recall_macro:.4f}"
)

print(
    f"F1 Macro           : {f1_macro:.4f}"
)

print(
    f"F1 Weighted        : {f1_weighted:.4f}"
)


# ============================================================
# 17. RAPPORT DETAILLE
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(
        len(label_encoder.classes_)
    ),
    target_names=label_encoder.classes_,
    zero_division=0
)

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

print(
    f"\nRapport sauvegardé : {report_path}"
)


# ============================================================
# 18. RE-ENTRAINEMENT FINAL SUR LES 586 CLASSES
# ============================================================

print("\n" + "=" * 60)
print("RE-ENTRAINEMENT FINAL")
print("=" * 60)

print(
    f"Entraînement sur {len(df_conf):,} lignes"
)

print(
    f"et {len(label_encoder.classes_)} classes."
)


final_model = LinearSVC(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    max_iter=3000
)

final_model.fit(
    X,
    y
)

print(
    "✓ Modèle final entraîné sur 100 % "
    "du dataset de confiance."
)


# ============================================================
# 19. SAUVEGARDE MODELE + LABEL ENCODER
# ============================================================

joblib.dump(
    final_model,
    MODEL_PATH
)

joblib.dump(
    label_encoder,
    LABEL_ENCODER_PATH
)

print(
    f"\n✓ Modèle sauvegardé : {MODEL_PATH}"
)

print(
    f"✓ LabelEncoder sauvegardé : "
    f"{LABEL_ENCODER_PATH}"
)


# ============================================================
# 20. RECHARGEMENT DU MODELE SAUVEGARDE
# ============================================================

print("\nRechargement des artefacts...")

saved_model = joblib.load(
    MODEL_PATH
)

saved_label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

print("✓ Modèle rechargé.")
print("✓ LabelEncoder rechargé.")


# ============================================================
# 21. CHARGEMENT DU DATASET COMPLET
# ============================================================

print("\nChargement du dataset complet...")

df_full = pd.read_csv(
    FULL_DATASET_CSV,
    low_memory=False
)

print(
    f"Dataset complet : "
    f"{len(df_full):,} lignes"
)


# ============================================================
# 22. ASSOCIATION DES EMBEDDINGS EXISTANTS
# ============================================================

df_full["Libellé produit"] = (
    df_full["Libellé produit"]
    .fillna("")
    .astype(str)
    .str.strip()
)

print(
    "\nAssociation avec les embeddings "
    "déjà sauvegardés..."
)

df_full["Embedding"] = (
    df_full["Libellé produit"]
    .map(embedding_map)
)


# ============================================================
# 23. VERIFICATION EMBEDDINGS MANQUANTS
# ============================================================

missing_full = (
    df_full["Embedding"]
    .isna()
    .sum()
)

print(
    f"Embeddings manquants : "
    f"{missing_full:,}"
)

if missing_full > 0:

    missing_products = (
        df_full.loc[
            df_full["Embedding"].isna(),
            "Libellé produit"
        ]
        .drop_duplicates()
    )

    print(
        f"Produits uniques sans embedding : "
        f"{len(missing_products):,}"
    )

    print(
        missing_products
        .head(20)
        .to_list()
    )

    raise ValueError(
        "Le dataset complet contient des produits "
        "sans embedding sauvegardé."
    )


# ============================================================
# 24. PREDICTION PAR PRODUITS UNIQUES
# ============================================================

print(
    "\nClassification des produits uniques..."
)

unique_products = (
    df_full["Libellé produit"]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(
    f"Produits uniques à classifier : "
    f"{len(unique_products):,}"
)


# Récupération des embeddings uniquement
# pour les produits uniques

X_unique = np.vstack(
    unique_products.map(
        embedding_map
    ).values
).astype(np.float32)

print(
    f"Matrice embeddings uniques : "
    f"{X_unique.shape}"
)


# ============================================================
# 25. PREDICTION
# ============================================================

pred_encoded = saved_model.predict(
    X_unique
)

pred_natures = saved_label_encoder.inverse_transform(
    pred_encoded
)


# ============================================================
# 26. DICTIONNAIRE PRODUIT → PREDICTION
# ============================================================

prediction_map = dict(
    zip(
        unique_products,
        pred_natures
    )
)


# ============================================================
# 27. AJOUT DE LA COLONNE
# ============================================================

df_full["Classification_Prediction"] = (
    df_full["Libellé produit"]
    .map(prediction_map)
)


# ============================================================
# 28. VERIFICATION
# ============================================================

missing_predictions = (
    df_full["Classification_Prediction"]
    .isna()
    .sum()
)

print(
    f"\nPrédictions manquantes : "
    f"{missing_predictions:,}"
)

if missing_predictions > 0:
    raise ValueError(
        "Certaines lignes n'ont pas reçu "
        "de Classification_Prediction."
    )


print("\nExemple :")

print(
    df_full[
        [
            "Libellé produit",
            "Nature",
            "Prediction",
            "Classification_Prediction"
        ]
    ].head(20)
)


# ============================================================
# 29. SUPPRESSION DE LA COLONNE EMBEDDING
# ============================================================

# On ne met PAS les vecteurs 1024D dans le CSV final.

df_full.drop(
    columns=["Embedding"],
    inplace=True
)


# ============================================================
# 30. SAUVEGARDE CSV FINAL
# ============================================================

df_full.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("TERMINE")
print("=" * 60)

print(
    f"CSV final : {OUTPUT_CSV}"
)

print(
    f"Nombre de lignes : {len(df_full):,}"
)

print(
    "Nouvelle colonne : Classification_Prediction"
)