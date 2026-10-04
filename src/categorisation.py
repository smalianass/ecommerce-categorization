import pandas as pd
import numpy as np
import ollama
import os
import ast

from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "../data/input/20210614 Ecommerce sales.xlsb"

NATURE_EMBEDDINGS_FILE = (
    "../data/output/embeddings_natures.csv"
)

PRODUCT_EMBEDDINGS_FILE = (
    "../data/output/embeddings_produits_uniques.csv"
)

PREDICTIONS_FILE = (
    "../data/output/predictions_525034.csv"
)

CONFIDENCE_FILE = (
    "../data/output/dataset_confiance.csv"
)

MISSING_NATURES_FILE = (
    "../data/output/natures_manquantes.csv"
)

DISTRIBUTION_FILE = (
    "../data/output/distribution_natures_confiance.csv"
)

BATCH_SIZE = 32

# Pour la construction du dataset de confiance,
# nous utilisons Prediction == Nature.
# ============================================================
# CHARGEMENT DE LA CONFIGURATION .ENV
# ============================================================

load_dotenv()

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_LLM_MODEL = os.getenv(
    "OLLAMA_LLM_MODEL",
    "llama3.2:3b"
)

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "bge-m3"
)

BATCH_SIZE = int(
    os.getenv(
        "OLLAMA_BATCH_SIZE",
        "32"
    )
)

SIMILARITY_BATCH_SIZE = int(
    os.getenv(
        "SIMILARITY_BATCH_SIZE",
        "1000"
    )
)

# Configuration du client Ollama
ollama_client = ollama.Client(
    host=OLLAMA_HOST
)

print("=" * 70)
print("CONFIGURATION OLLAMA")
print("=" * 70)
print("OLLAMA_HOST              :", OLLAMA_HOST)
print("OLLAMA_LLM_MODEL         :", OLLAMA_LLM_MODEL)
print("OLLAMA_EMBEDDING_MODEL   :", OLLAMA_EMBEDDING_MODEL)
print("OLLAMA_BATCH_SIZE        :", BATCH_SIZE)
print("SIMILARITY_BATCH_SIZE    :", SIMILARITY_BATCH_SIZE)


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

print("\n" + "=" * 70)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 70)

data = pd.read_excel(
    INPUT_FILE
)

print(
    "Nombre de lignes :",
    len(data)
)

if len(data) != 525034:
    print(
        "⚠ Attention : le nombre de lignes attendu était 525 034."
    )
    print(
        "Nombre réellement chargé :",
        len(data)
    )


# ============================================================
# 2. EXTRACTION DES NATURES
# ============================================================

print("\n" + "=" * 70)
print("2. EXTRACTION DES NATURES")
print("=" * 70)

natures_originales = (
    data["Nature"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

print(
    "Nombre de Natures uniques :",
    len(natures_originales)
)

natures = natures_originales


# ============================================================
# 3. GÉNÉRATION DES DÉFINITIONS
# ============================================================

def generate_nature_definitions(natures):

    definitions = []

    for i, nature in enumerate(natures):

        prompt = f"""
Tu es un expert en classification de produits e-commerce.

La catégorie produit est :
"{nature}"

Donne une définition courte, précise et adaptée à la classification
e-commerce de cette catégorie.

La définition doit :
- expliquer ce que désigne cette catégorie ;
- mentionner les caractéristiques permettant de reconnaître les produits ;
- utiliser un vocabulaire pertinent pour identifier les produits ;
- rester générale et ne pas inventer de marques ;
- faire 1 à 3 phrases maximum.

Réponds uniquement avec la définition.
"""

        response = ollama_client.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        definition = (
            response["message"]["content"]
            .strip()
        )

        definitions.append(definition)

        print(
            f"Définition {i + 1}/{len(natures)} : {nature}"
        )

    return definitions


# ============================================================
# 4. CALCUL DES EMBEDDINGS DES NATURES
# ============================================================

def calculate_embedding_natures(nature_texts):

    embedding_natures = []

    for i, text in enumerate(nature_texts):

        response = ollama_client.embed(
            model=OLLAMA_EMBEDDING_MODEL,
            input=text
        )

        embedding = (
            response["embeddings"][0]
        )

        embedding_natures.append(
            embedding
        )

        print(
            f"Embedding Nature "
            f"{i + 1}/{len(nature_texts)}"
        )

    return embedding_natures


# ============================================================
# 5. SAUVEGARDE DES NATURES
# ============================================================

def save_embeddings_to_csv(
    natures,
    definitions,
    embeddings,
    output_file
):

    df = pd.DataFrame({
        "Nature": natures,
        "Definition": definitions,
        "Embedding": embeddings
    })

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 6. CHARGEMENT OU CALCUL DES EMBEDDINGS DES NATURES
# ============================================================

print("\n" + "=" * 70)
print("6. DÉFINITIONS + EMBEDDINGS DES NATURES")
print("=" * 70)

if os.path.exists(
    NATURE_EMBEDDINGS_FILE
):

    print(
        "Fichier existant détecté :",
        NATURE_EMBEDDINGS_FILE
    )

    nature_embeddings_df = pd.read_csv(
        NATURE_EMBEDDINGS_FILE
    )

    natures_loaded = (
        nature_embeddings_df["Nature"]
        .astype(str)
        .values
    )

    generated_definitions = (
        nature_embeddings_df["Definition"]
        .fillna("")
        .tolist()
    )

    embeddings_natures = (
        nature_embeddings_df["Embedding"]
        .apply(ast.literal_eval)
        .tolist()
    )

    natures = natures_loaded

    print(
        "Nombre de Natures chargées :",
        len(natures)
    )

    print(
        "Nombre de définitions chargées :",
        len(generated_definitions)
    )

    print(
        "Nombre d'embeddings chargés :",
        len(embeddings_natures)
    )

    print(
        "Dimension des embeddings :",
        len(embeddings_natures[0])
    )

else:

    print(
        "Aucun fichier d'embeddings trouvé."
    )

    print(
        "Génération des définitions avec Ollama..."
    )

    generated_definitions = (
        generate_nature_definitions(
            natures
        )
    )

    nature_texts = [
        (
            f"Catégorie : {nature}. "
            f"Définition : {definition}"
        )
        for nature, definition
        in zip(
            natures,
            generated_definitions
        )
    ]

    print(
        "\nGénération des embeddings BGE-M3..."
    )

    embeddings_natures = (
        calculate_embedding_natures(
            nature_texts
        )
    )

    save_embeddings_to_csv(
        natures,
        generated_definitions,
        embeddings_natures,
        NATURE_EMBEDDINGS_FILE
    )

    print(
        "Embeddings des Natures sauvegardés."
    )


# ============================================================
# 7. VÉRIFICATION DES NATURES
# ============================================================

print("\n" + "=" * 70)
print("7. VÉRIFICATION DES NATURES")
print("=" * 70)

natures_originales_set = set(
    str(x).strip()
    for x in natures_originales
)

natures_embeddings_set = set(
    str(x).strip()
    for x in natures
)

missing_natures_embeddings = (
    natures_originales_set
    - natures_embeddings_set
)

if missing_natures_embeddings:

    print(
        "⚠ Natures présentes dans le fichier source "
        "mais absentes des embeddings :"
    )

    for nature in sorted(
        missing_natures_embeddings
    ):
        print(
            "-",
            nature
        )

    raise ValueError(
        "Les embeddings des Natures ne couvrent "
        "pas toutes les Natures du dataset."
    )

else:

    print(
        "✓ Toutes les Natures du dataset possèdent "
        "un embedding."
    )

    print(
        "Nombre de Natures couvertes :",
        len(natures)
    )


# ============================================================
# 8. PRÉTRAITEMENT DES PRODUITS
# ============================================================

print("\n" + "=" * 70)
print("8. PRÉTRAITEMENT DES PRODUITS")
print("=" * 70)


def preprocess_data(data):

    data = data.copy()

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

    return data


preprocessed_data = (
    preprocess_data(data)
)

print(
    "Prétraitement terminé."
)


# ============================================================
# 9. EXTRACTION DES PRODUITS UNIQUES
# ============================================================

print("\n" + "=" * 70)
print("9. PRODUITS UNIQUES")
print("=" * 70)

unique_products = (
    preprocessed_data[
        "Libellé produit"
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(
    "Nombre de lignes originales :",
    len(preprocessed_data)
)

print(
    "Nombre de produits uniques :",
    len(unique_products)
)

print(
    "Nombre de doublons évités :",
    len(preprocessed_data)
    - len(unique_products)
)


# ============================================================
# 10. CALCUL DES EMBEDDINGS DES PRODUITS
# ============================================================

def calculate_embeddings_for_products(
    products,
    batch_size=32
):

    embeddings = []

    products = list(products)

    for i in range(
        0,
        len(products),
        batch_size
    ):

        batch = products[
            i:i + batch_size
        ]

        response = ollama_client.embed(
            model=OLLAMA_EMBEDDING_MODEL,
            input=batch
        )

        embeddings.extend(
            response["embeddings"]
        )

        print(
            f"Produits : "
            f"{min(i + batch_size, len(products))}"
            f"/{len(products)}"
        )

    return embeddings

# ============================================================
# 11. CHARGEMENT / CALCUL DES EMBEDDINGS PRODUITS
# ============================================================

print("\n" + "=" * 70)
print("11. EMBEDDINGS DES PRODUITS UNIQUES")
print("=" * 70)

if os.path.exists(
    PRODUCT_EMBEDDINGS_FILE
):

    print(
        "Cache des embeddings produits détecté :"
    )

    print(
        PRODUCT_EMBEDDINGS_FILE
    )

    product_embeddings_df = pd.read_csv(
        PRODUCT_EMBEDDINGS_FILE
    )

    cached_products = (
        product_embeddings_df[
            "Libellé produit"
        ]
        .tolist()
    )

    cached_embeddings = (
        product_embeddings_df[
            "Embedding"
        ]
        .apply(ast.literal_eval)
        .tolist()
    )

    print(
        "Produits présents dans le cache :",
        len(cached_products)
    )

    # --------------------------------------------------------
    # Dictionnaire produit -> embedding
    # --------------------------------------------------------

    embedding_dict = dict(
        zip(
            cached_products,
            cached_embeddings
        )
    )

    # --------------------------------------------------------
    # Recherche des produits absents du cache
    # --------------------------------------------------------

    missing_products = [
        product
        for product in unique_products
        if product not in embedding_dict
    ]

    print(
        "Produits restant à calculer :",
        len(missing_products)
    )

    if missing_products:

        new_embeddings = (
            calculate_embeddings_for_products(
                missing_products,
                batch_size=BATCH_SIZE
            )
        )

        for product, embedding in zip(
            missing_products,
            new_embeddings
        ):

            embedding_dict[
                product
            ] = embedding

        # ----------------------------------------------------
        # Sauvegarde du cache complet
        # ----------------------------------------------------

        product_embeddings_df = pd.DataFrame({
            "Libellé produit": list(
                embedding_dict.keys()
            ),
            "Embedding": list(
                embedding_dict.values()
            )
        })

        product_embeddings_df.to_csv(
            PRODUCT_EMBEDDINGS_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "Nouveau cache sauvegardé."
        )

else:

    print(
        "Aucun cache trouvé."
    )

    print(
        "Calcul des embeddings des",
        len(unique_products),
        "produits uniques..."
    )

    unique_embeddings = (
        calculate_embeddings_for_products(
            unique_products,
            batch_size=BATCH_SIZE
        )
    )

    embedding_dict = dict(
        zip(
            unique_products,
            unique_embeddings
        )
    )

    product_embeddings_df = pd.DataFrame({
        "Libellé produit": list(
            embedding_dict.keys()
        ),
        "Embedding": list(
            embedding_dict.values()
        )
    })

    product_embeddings_df.to_csv(
        PRODUCT_EMBEDDINGS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "Cache des embeddings sauvegardé dans :",
        PRODUCT_EMBEDDINGS_FILE
    )


print(
    "Nombre total d'embeddings disponibles :",
    len(embedding_dict)
)


# ============================================================
# 12. CONSTRUCTION DE LA MATRICE DES EMBEDDINGS PRODUITS
# ============================================================

print("\n" + "=" * 70)
print("12. CONSTRUCTION DES EMBEDDINGS")
print("=" * 70)

product_embeddings = np.asarray(
    [
        embedding_dict[product]
        for product in unique_products
    ],
    dtype=np.float32
)

nature_embeddings = np.asarray(
    embeddings_natures,
    dtype=np.float32
)

print(
    "Shape embeddings produits :",
    product_embeddings.shape
)

print(
    "Shape embeddings Natures :",
    nature_embeddings.shape
)


# ============================================================
# 13. NORMALISATION DES EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("13. NORMALISATION")
print("=" * 70)


def normalize_embeddings(
    embeddings
):

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms = np.maximum(
        norms,
        1e-12
    )

    return (
        embeddings / norms
    )


product_embeddings_normalized = (
    normalize_embeddings(
        product_embeddings
    )
)

nature_embeddings_normalized = (
    normalize_embeddings(
        nature_embeddings
    )
)


# ============================================================
# 14. CLASSIFICATION PAR SIMILARITÉ COSINUS
# ============================================================

print("\n" + "=" * 70)
print("14. CLASSIFICATION DES PRODUITS")
print("=" * 70)

# ------------------------------------------------------------
# Important :
#
# 55 830 x 596 = environ 33 millions de similarités.
#
# On traite les produits par blocs pour ne pas créer une
# matrice énorme en mémoire.
# ------------------------------------------------------------

SIMILARITY_BATCH_SIZE = 1000

predicted_natures_unique = []
predicted_scores_unique = []

number_products = (
    len(product_embeddings_normalized)
)

for start in range(
    0,
    number_products,
    SIMILARITY_BATCH_SIZE
):

    end = min(
        start + SIMILARITY_BATCH_SIZE,
        number_products
    )

    product_batch = (
        product_embeddings_normalized[
            start:end
        ]
    )

    # Cosine similarity
    similarities = (
        product_batch
        @ nature_embeddings_normalized.T
    )

    # Meilleure Nature
    best_indices = np.argmax(
        similarities,
        axis=1
    )

    best_scores = similarities[
        np.arange(
            len(best_indices)
        ),
        best_indices
    ]

    best_natures = (
        np.asarray(natures)[
            best_indices
        ]
    )

    predicted_natures_unique.extend(
        best_natures.tolist()
    )

    predicted_scores_unique.extend(
        best_scores.tolist()
    )

    print(
        f"Classification : "
        f"{end}/{number_products}"
    )


# ============================================================
# 15. TABLE DES PRÉDICTIONS UNIQUES
# ============================================================

print("\n" + "=" * 70)
print("15. PRÉDICTIONS DES PRODUITS UNIQUES")
print("=" * 70)

unique_predictions = pd.DataFrame({

    "Libellé produit":
        unique_products,

    "Prediction":
        predicted_natures_unique,

    "Score":
        predicted_scores_unique

})

print(
    unique_predictions.head(10).to_string(
        index=False
    )
)

print(
    "Nombre de produits classifiés :",
    len(unique_predictions)
)


# ============================================================
# 16. REPORT DES PRÉDICTIONS SUR LES 525 034 LIGNES
# ============================================================

print("\n" + "=" * 70)
print("16. REPORT DES PRÉDICTIONS SUR TOUTES LES LIGNES")
print("=" * 70)

# Dictionnaires pour retrouver rapidement la prédiction
prediction_dict = dict(
    zip(
        unique_predictions[
            "Libellé produit"
        ],
        unique_predictions[
            "Prediction"
        ]
    )
)

score_dict = dict(
    zip(
        unique_predictions[
            "Libellé produit"
        ],
        unique_predictions[
            "Score"
        ]
    )
)

preprocessed_data["Prediction"] = (
    preprocessed_data[
        "Libellé produit"
    ].map(
        prediction_dict
    )
)

preprocessed_data["Score"] = (
    preprocessed_data[
        "Libellé produit"
    ].map(
        score_dict
    )
)

print(
    "Nombre de lignes avec Prediction :",
    preprocessed_data[
        "Prediction"
    ].notna().sum()
)

print(
    "Nombre de lignes sans Prediction :",
    preprocessed_data[
        "Prediction"
    ].isna().sum()
)


# ============================================================
# 17. CONSTRUCTION DU DATASET DE RÉSULTATS
# ============================================================

print("\n" + "=" * 70)
print("17. DATASET DE RÉSULTATS")
print("=" * 70)

results = preprocessed_data[
    [
        "Libellé produit",
        "Prediction",
        "Score",
        "Nature"
    ]
].copy()


# ============================================================
# 18. COMPARAISON PREDICTION / NATURE
# ============================================================

print("\n" + "=" * 70)
print("18. COMPARAISON PREDICTION / NATURE")
print("=" * 70)

results["Correct"] = (
    results["Prediction"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    ==
    results["Nature"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

print(
    "Nombre de lignes Prediction == Nature :",
    results["Correct"].sum()
)

print(
    "Nombre de lignes Prediction != Nature :",
    (~results["Correct"]).sum()
)

print(
    "Taux d'accord :",
    f"{results['Correct'].mean():.2%}"
)


# ============================================================
# 19. SAUVEGARDE DES PRÉDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("19. SAUVEGARDE DES PRÉDICTIONS")
print("=" * 70)

results.to_csv(
    PREDICTIONS_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    "Résultats sauvegardés dans :",
    PREDICTIONS_FILE
)


# ============================================================
# 20. CONSTRUCTION DU DATASET DE CONFIANCE
# ============================================================

print("\n" + "=" * 70)
print("20. DATASET DE CONFIANCE")
print("=" * 70)

dataset_confiance = (
    results[
        results["Correct"]
    ]
    .copy()
)

print(
    "Nombre de lignes du dataset de confiance :",
    len(dataset_confiance)
)

print(
    "Taux de conservation :",
    f"{len(dataset_confiance) / len(results):.2%}"
)


# ============================================================
# 21. VÉRIFICATION DES NATURES
# ============================================================

print("\n" + "=" * 70)
print("21. VÉRIFICATION DES 596 NATURES")
print("=" * 70)

# ------------------------------------------------------------
# Natures originales
# ------------------------------------------------------------

natures_originales_set = set(
    data["Nature"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

# ------------------------------------------------------------
# Natures dans dataset confiance
# ------------------------------------------------------------

natures_confiance_set = set(
    dataset_confiance[
        "Nature"
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

# ------------------------------------------------------------
# Natures manquantes
# ------------------------------------------------------------

natures_manquantes = sorted(
    natures_originales_set
    -
    natures_confiance_set
)

print(
    "Natures originales :",
    len(natures_originales_set)
)

print(
    "Natures représentées dans confiance :",
    len(natures_confiance_set)
)

print(
    "Natures manquantes :",
    len(natures_manquantes)
)


# ============================================================
# 22. AFFICHAGE DES NATURES MANQUANTES
# ============================================================

if len(natures_manquantes) == 0:

    print(
        "\n✓ TOUTES LES NATURES SONT PRÉSENTES."
    )

    print(
        "Les",
        len(natures_originales_set),
        "Natures sont représentées."
    )

else:

    print(
        "\n⚠ NATURES MANQUANTES :"
    )

    for i, nature in enumerate(
        natures_manquantes,
        start=1
    ):

        print(
            f"{i:3d}. {nature}"
        )


# ============================================================
# 23. DISTRIBUTION DES NATURES
# ============================================================

print("\n" + "=" * 70)
print("23. DISTRIBUTION DES NATURES")
print("=" * 70)

distribution_natures = (
    dataset_confiance[
        "Nature"
    ]
    .value_counts()
    .sort_index()
)

print(
    distribution_natures.to_string()
)


# ============================================================
# 24. NATURES AVEC PEU D'EXEMPLES
# ============================================================

print("\n" + "=" * 70)
print("24. NATURES AVEC PEU D'EXEMPLES")
print("=" * 70)

natures_moins_10 = (
    distribution_natures[
        distribution_natures < 10
    ]
    .sort_values()
)

print(
    "Nombre de Natures avec moins de 10 exemples :",
    len(natures_moins_10)
)

if len(natures_moins_10) > 0:

    print(
        "\nNatures concernées :"
    )

    print(
        natures_moins_10.to_string()
    )

else:

    print(
        "✓ Toutes les Natures ont au moins 10 exemples."
    )


# ============================================================
# 25. NATURES AVEC MOINS DE 5 EXEMPLES
# ============================================================

print("\n" + "=" * 70)
print("25. NATURES AVEC MOINS DE 5 EXEMPLES")
print("=" * 70)

natures_moins_5 = (
    distribution_natures[
        distribution_natures < 5
    ]
    .sort_values()
)

print(
    "Nombre :",
    len(natures_moins_5)
)

if len(natures_moins_5) > 0:

    print(
        natures_moins_5.to_string()
    )

else:

    print(
        "✓ Aucune Nature avec moins de 5 exemples."
    )


# ============================================================
# 26. SAUVEGARDE DU DATASET DE CONFIANCE
# ============================================================

print("\n" + "=" * 70)
print("26. SAUVEGARDE DATASET DE CONFIANCE")
print("=" * 70)

dataset_confiance.to_csv(
    CONFIDENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    "Dataset de confiance sauvegardé dans :",
    CONFIDENCE_FILE
)


# ============================================================
# 27. SAUVEGARDE DES NATURES MANQUANTES
# ============================================================

pd.DataFrame({
    "Nature": natures_manquantes
}).to_csv(
    MISSING_NATURES_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    "Natures manquantes sauvegardées dans :",
    MISSING_NATURES_FILE
)


# ============================================================
# 28. SAUVEGARDE DE LA DISTRIBUTION
# ============================================================

distribution_df = (
    distribution_natures
    .rename("Nombre_exemples")
    .reset_index()
)

distribution_df.columns = [
    "Nature",
    "Nombre_exemples"
]

distribution_df.to_csv(
    DISTRIBUTION_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    "Distribution sauvegardée dans :",
    DISTRIBUTION_FILE
)


# ============================================================
# 29. CONTRÔLE FINAL AVANT CLASSIFIEUR
# ============================================================

print("\n" + "=" * 70)
print("29. CONTRÔLE FINAL AVANT ENTRAÎNEMENT")
print("=" * 70)

if len(natures_manquantes) == 0:

    print(
        "✓ COUVERTURE COMPLÈTE"
    )

    print(
        f"✓ {len(natures_originales_set)} / "
        f"{len(natures_originales_set)} Natures présentes."
    )

else:

    print(
        "✗ COUVERTURE INCOMPLÈTE"
    )

    print(
        f"✗ {len(natures_manquantes)} "
        "Natures sont absentes."
    )

    print(
        "✗ NE PAS considérer le dataset comme "
        "complet pour l'entraînement."
    )


# ============================================================
# 30. RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 70)
print("30. RÉSUMÉ FINAL")
print("=" * 70)

print(
    f"Lignes originales                  : "
    f"{len(data):,}"
)

print(
    f"Produits uniques                   : "
    f"{len(unique_products):,}"
)

print(
    f"Natures originales                 : "
    f"{len(natures_originales_set):,}"
)

print(
    f"Lignes classifiées                 : "
    f"{len(results):,}"
)

print(
    f"Lignes dataset confiance           : "
    f"{len(dataset_confiance):,}"
)

print(
    f"Taux de conservation               : "
    f"{len(dataset_confiance) / len(results):.2%}"
)

print(
    f"Natures dans dataset confiance     : "
    f"{len(natures_confiance_set):,}"
)

print(
    f"Natures manquantes                 : "
    f"{len(natures_manquantes):,}"
)

print(
    f"Natures < 10 exemples              : "
    f"{len(natures_moins_10):,}"
)

print(
    f"Natures < 5 exemples               : "
    f"{len(natures_moins_5):,}"
)

print(
    "\nFichiers générés :"
)

print(
    "1.",
    PREDICTIONS_FILE
)

print(
    "2.",
    CONFIDENCE_FILE
)

print(
    "3.",
    MISSING_NATURES_FILE
)

print(
    "4.",
    DISTRIBUTION_FILE
)

print(
    "5.",
    PRODUCT_EMBEDDINGS_FILE
)

print("\nTraitement terminé.")