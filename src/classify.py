"""
Classification supervisée des produits e-commerce.

Pipeline :
1. Chargement du dataset de confiance enrichi
2. Nettoyage des données
3. Analyse de la distribution des classes
4. Split train/test stratifié
5. Génération des embeddings BGE-M3
6. Entraînement de plusieurs classifieurs
7. Gestion du déséquilibre des classes
8. Évaluation avec plusieurs métriques
9. Sauvegarde des modèles et résultats

IMPORTANT :
Le fichier utilisé ici sera le dataset de confiance enrichi après
annotation manuelle des classes rares ou absentes.

Exemple :
    data/output/dataset_confiance_enrichi.csv
"""

import os
import ast
import json
import joblib

import numpy as np
import pandas as pd
import ollama

from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "bge-m3"
)

CLASSIFICATION_BATCH_SIZE = int(
    os.getenv(
        "CLASSIFICATION_BATCH_SIZE",
        "32"
    )
)

TEST_SIZE = float(
    os.getenv(
        "CLASSIFICATION_TEST_SIZE",
        "0.20"
    )
)

RANDOM_STATE = int(
    os.getenv(
        "CLASSIFICATION_RANDOM_STATE",
        "42"
    )
)

MIN_CLASS_SAMPLES = int(
    os.getenv(
        "CLASSIFICATION_MIN_CLASS_SAMPLES",
        "15"
    )
)

ollama_client = ollama.Client(
    host=OLLAMA_HOST
)


# ============================================================
# 2. CHEMINS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "output"
)

# Dataset qui sera créé après annotation manuelle
DATASET_FILE = os.path.join(
    DATA_OUTPUT_DIR,
    "dataset_confiance.csv"
)

EMBEDDINGS_FILE = os.path.join(
    DATA_OUTPUT_DIR,
    "embeddings_classification.csv"
)

RESULTS_DIR = os.path.join(
    DATA_OUTPUT_DIR,
    "classification_results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 3. CHARGEMENT DU DATASET
# ============================================================

def load_dataset(file_path):
    """
    Charge le dataset de confiance enrichi.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nDataset introuvable : {file_path}\n\n"
            "Le fichier dataset_confiance_enrichi.csv doit être "
            "créé après l'annotation manuelle des classes rares "
            "ou absentes."
        )

    data = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("CHARGEMENT DU DATASET")
    print("=" * 70)

    print(f"Nombre de lignes : {len(data):,}")
    print(f"Nombre de colonnes : {len(data.columns)}")

    print("\nColonnes :")
    print(list(data.columns))

    return data


# ============================================================
# 4. PRETRAITEMENT
# ============================================================

def preprocess_dataset(data):
    """
    Nettoyage et préparation du dataset.
    """

    data = data.copy()

    required_columns = [
        "Libellé produit",
        "Nature"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    # --------------------------------------------------------
    # Nettoyage du texte
    # --------------------------------------------------------

    data["Libellé produit"] = (
        data["Libellé produit"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"[\x00-\x1F\x7F]",
            " ",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    # --------------------------------------------------------
    # Nettoyage de la cible
    # --------------------------------------------------------

    data["Nature"] = (
        data["Nature"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Supprimer les lignes sans texte ou sans classe
    data = data[
        (data["Libellé produit"] != "")
        &
        (data["Nature"] != "")
    ].copy()

    # --------------------------------------------------------
    # Suppression des doublons exacts
    # --------------------------------------------------------

    data = data.drop_duplicates(
        subset=[
            "Libellé produit",
            "Nature"
        ]
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("PRETRAITEMENT")
    print("=" * 70)

    print(
        f"Nombre de lignes après nettoyage : {len(data):,}"
    )

    print(
        f"Nombre de classes : {data['Nature'].nunique():,}"
    )

    return data


# ============================================================
# 5. ANALYSE DU DESÉQUILIBRE
# ============================================================

def analyze_class_distribution(data):
    """
    Analyse la distribution des classes.
    """

    distribution = (
        data["Nature"]
        .value_counts()
        .rename_axis("Nature")
        .reset_index(name="Nombre")
    )

    distribution["Pourcentage"] = (
        distribution["Nombre"]
        / len(data)
        * 100
    )

    print("\n" + "=" * 70)
    print("DISTRIBUTION DES CLASSES")
    print("=" * 70)

    print(
        f"Nombre de classes : {len(distribution)}"
    )

    print(
        f"Classe la plus représentée : "
        f"{distribution.iloc[0]['Nature']} "
        f"({distribution.iloc[0]['Nombre']:,})"
    )

    print(
        f"Classe la moins représentée : "
        f"{distribution.iloc[-1]['Nature']} "
        f"({distribution.iloc[-1]['Nombre']:,})"
    )

    print("\nStatistiques :")

    print(
        distribution["Nombre"].describe()
    )

    print("\n10 classes les plus représentées :")

    print(
        distribution.head(10).to_string(
            index=False
        )
    )

    print("\nClasses avec moins de 10 exemples :")

    print(
        distribution[
            distribution["Nombre"] < 10
        ].to_string(index=False)
    )


    return distribution


# ============================================================
# 6. FILTRAGE POUR LE SPLIT STRATIFIÉ
# ============================================================

def prepare_classes_for_split(
    data,
    min_class_samples=2
):
    """
    Retire uniquement les classes ne permettant pas de faire
    un split stratifié.

    Une classe doit avoir au minimum 2 exemples pour apparaître
    dans train et/ou test lors du split.

    Les classes rares devront idéalement être enrichies
    manuellement avant cette étape.
    """

    counts = data["Nature"].value_counts()

    valid_classes = counts[
        counts >= min_class_samples
    ].index

    removed_classes = counts[
        counts < min_class_samples
    ]

    data_filtered = data[
        data["Nature"].isin(valid_classes)
    ].copy()

    print("\n" + "=" * 70)
    print("PRÉPARATION DES CLASSES")
    print("=" * 70)

    print(
        f"Classes avant filtrage : {data['Nature'].nunique()}"
    )

    print(
        f"Classes conservées : {data_filtered['Nature'].nunique()}"
    )

    print(
        f"Classes avec moins de "
        f"{min_class_samples} exemples : "
        f"{len(removed_classes)}"
    )

    if len(removed_classes) > 0:

        print("\nClasses retirées temporairement :")

        print(
            removed_classes.to_string()
        )


    return data_filtered


# ============================================================
# 7. SPLIT TRAIN / TEST
# ============================================================

def split_dataset(data):
    """
    Split stratifié train/test.
    """

    X = data["Libellé produit"]
    y = data["Nature"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Train : {len(X_train):,} "
        f"({len(X_train) / len(data) * 100:.2f} %)"
    )

    print(
        f"Test  : {len(X_test):,} "
        f"({len(X_test) / len(data) * 100:.2f} %)"
    )

    print(
        f"Classes train : {y_train.nunique()}"
    )

    print(
        f"Classes test : {y_test.nunique()}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 8. ENCODAGE DES LABELS
# ============================================================

def encode_labels(
    y_train,
    y_test
):
    """
    Encode les catégories Nature en entiers.
    """

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(
        y_train
    )

    y_test_encoded = encoder.transform(
        y_test
    )

    print("\n" + "=" * 70)
    print("ENCODAGE DES LABELS")
    print("=" * 70)

    print(
        f"Nombre de classes : "
        f"{len(encoder.classes_)}"
    )

    return (
        y_train_encoded,
        y_test_encoded,
        encoder
    )


# ============================================================
# 9. EMBEDDINGS
# ============================================================

def calculate_embeddings(
    texts,
    batch_size=32
):
    """
    Calcule les embeddings BGE-M3 avec Ollama.
    """

    texts = list(texts)

    embeddings = []

    total = len(texts)

    print("\n" + "=" * 70)
    print("CALCUL DES EMBEDDINGS")
    print("=" * 70)

    print(
        f"Nombre de textes : {total:,}"
    )

    print(
        f"Modèle : {OLLAMA_EMBEDDING_MODEL}"
    )

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start + batch_size,
            total
        )

        batch = texts[start:end]

        response = ollama_client.embed(
            model=OLLAMA_EMBEDDING_MODEL,
            input=batch
        )

        batch_embeddings = response[
            "embeddings"
        ]

        embeddings.extend(
            batch_embeddings
        )

        print(
            f"Embeddings : "
            f"{end:,}/{total:,}"
        )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# 10. SAUVEGARDE / CHARGEMENT DES EMBEDDINGS
# ============================================================

def save_embeddings(
    texts,
    embeddings,
    file_path
):
    """
    Sauvegarde les embeddings.
    """

    df_embeddings = pd.DataFrame({
        "Libellé produit": texts,
        "Embedding": [
            embedding.tolist()
            for embedding in embeddings
        ]
    })

    df_embeddings.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nEmbeddings sauvegardés : "
        f"{file_path}"
    )


# ============================================================
# 11. MÉTRIQUES
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
    label_encoder
):
    """
    Évalue un classifieur avec plusieurs métriques.
    """

    y_pred = model.predict(
        X_test
    )

    metrics = {
        "Model": model_name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_test,
            y_pred
        ),

        "Precision Macro": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Recall Macro": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "F1 Macro": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "F1 Weighted": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    print("\n" + "=" * 70)
    print(f"ÉVALUATION : {model_name}")
    print("=" * 70)

    for metric, value in metrics.items():

        if metric != "Model":

            print(
                f"{metric:<20}: "
                f"{value:.4f}"
            )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(
            len(label_encoder.classes_)
        ),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"{model_name}_classification_report.csv"
        ),
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Matrice de confusion
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            f"{model_name}_confusion_matrix.npy"
        ),
        cm
    )

    return metrics


# ============================================================
# 12. ENTRAÎNEMENT DES CLASSIFIEURS
# ============================================================

def train_classifiers(
    X_train,
    y_train,
    X_test,
    y_test,
    label_encoder
):
    """
    Entraîne plusieurs modèles.

    Les modèles linéaires utilisent class_weight='balanced'
    afin de prendre en compte le déséquilibre des classes.
    """

    models = {

        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000
        ),

        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE
        )
    }

    all_metrics = []

    for model_name, model in models.items():

        print("\n" + "=" * 70)
        print(
            f"ENTRAÎNEMENT : {model_name}"
        )
        print("=" * 70)

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model_name,
            model,
            X_test,
            y_test,
            label_encoder
        )

        all_metrics.append(
            metrics
        )

        model_path = os.path.join(
            RESULTS_DIR,
            f"{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"\nModèle sauvegardé : "
            f"{model_path}"
        )

    results_df = pd.DataFrame(
        all_metrics
    )

    results_df = results_df.sort_values(
        "F1 Macro",
        ascending=False
    )

    results_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "model_comparison.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    return results_df


# ============================================================
# 13. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION SUPERVISÉE DES NATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    data = load_dataset(
        DATASET_FILE
    )

    # --------------------------------------------------------
    # Prétraitement
    # --------------------------------------------------------

    data = preprocess_dataset(
        data
    )

    # --------------------------------------------------------
    # Analyse distribution
    # --------------------------------------------------------

    analyze_class_distribution(
        data
    )

    # --------------------------------------------------------
    # Préparation des classes
    # --------------------------------------------------------

    data = prepare_classes_for_split(
        data,
        MIN_CLASS_SAMPLES
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    ) = split_dataset(
        data
    )

    # --------------------------------------------------------
    # Encodage des labels
    # --------------------------------------------------------

    (
        y_train_encoded,
        y_test_encoded,
        label_encoder
    ) = encode_labels(
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    X_train_embeddings = calculate_embeddings(
        X_train_text,
        batch_size=CLASSIFICATION_BATCH_SIZE
    )

    X_test_embeddings = calculate_embeddings(
        X_test_text,
        batch_size=CLASSIFICATION_BATCH_SIZE
    )

    print("\nDimension des embeddings :")
    print(
        X_train_embeddings.shape
    )

    # --------------------------------------------------------
    # Sauvegarde des embeddings
    # --------------------------------------------------------

    train_embeddings_file = os.path.join(
        RESULTS_DIR,
        "embeddings_train.csv"
    )

    test_embeddings_file = os.path.join(
        RESULTS_DIR,
        "embeddings_test.csv"
    )

    save_embeddings(
        X_train_text,
        X_train_embeddings,
        train_embeddings_file
    )

    save_embeddings(
        X_test_text,
        X_test_embeddings,
        test_embeddings_file
    )

    # --------------------------------------------------------
    # Entraînement + évaluation
    # --------------------------------------------------------

    results = train_classifiers(
        X_train_embeddings,
        y_train_encoded,
        X_test_embeddings,
        y_test_encoded,
        label_encoder
    )

    # --------------------------------------------------------
    # Sauvegarde du LabelEncoder
    # --------------------------------------------------------

    joblib.dump(
        label_encoder,
        os.path.join(
            RESULTS_DIR,
            "label_encoder.joblib"
        )
    )

    # --------------------------------------------------------
    # Résultats finaux
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPARAISON FINALE DES MODÈLES")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("MEILLEUR MODÈLE")
    print("=" * 70)

    best_model = results.iloc[0]

    print(
        f"Modèle : {best_model['Model']}"
    )

    print(
        f"F1 Macro : "
        f"{best_model['F1 Macro']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{best_model['Balanced Accuracy']:.4f}"
    )

    print("\nClassification terminée.")


# ============================================================
# 14. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
