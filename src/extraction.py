import pandas as pd
import re
import os


# ============================================================
# ÉTAPE 1 — CHARGEMENT DES DONNÉES
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 1 — CHARGEMENT DES DONNÉES")
print("=" * 70)

input_file = "../data/input/20210614 Ecommerce sales.xlsb"

data = pd.read_excel(
    input_file,
    engine="pyxlsb"
)

print(f"Nombre total de lignes : {len(data)}")


# ============================================================
# ÉTAPE 2 — PRODUITS UNIQUES
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 2 — PRODUITS UNIQUES")
print("=" * 70)

data["Libellé produit"] = (
    data["Libellé produit"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"[\x00-\x1F\x7F]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

unique_products = data["Libellé produit"].drop_duplicates().tolist()

print(f"Nombre total de lignes : {len(data)}")
print(f"Nombre de descriptions uniques : {len(unique_products)}")

print("\nPremiers produits :")

for i, product in enumerate(unique_products[:10], start=1):
    print(f"{i}. {product}")


# ============================================================
# ÉTAPE 3 — DICTIONNAIRE DES COULEURS
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 3 — DICTIONNAIRE DES COULEURS")
print("=" * 70)


# ------------------------------------------------------------
# Couleurs composées
# ------------------------------------------------------------

COMPOUND_COLORS = [
    "bleu pétrole",
    "bleu canard",
    "bleu marine",
    "bleu nuit",
    "bleu ciel",

    "vert sauge",
    "vert olive",
    "vert bouteille",
    "vert kaki",

    "gris anthracite",
    "gris clair",
    "gris foncé",

    "rose poudré",
    "rose pâle",
    "rose pale",
]


# ------------------------------------------------------------
# Couleurs simples
# ------------------------------------------------------------

SIMPLE_COLORS = [
    "blanc",
    "noir",
    "gris",
    "beige",
    "marron",
    "brun",
    "rouge",
    "orange",
    "jaune",
    "vert",
    "bleu",
    "violet",
    "rose",
    "turquoise",
    "taupe",
    "écru",
    "ecru",
    "ivoire",
    "crème",
    "creme",
    "terracotta",
]


# ------------------------------------------------------------
# Variantes grammaticales
#
# Exemple :
# blanche / blanches -> blanc
# noire / noires    -> noir
# grise / grises    -> gris
# bleue / bleues    -> bleu
# verte / vertes    -> vert
# rouge / rouges    -> rouge
# ------------------------------------------------------------

COLOR_VARIANTS = {

    "blanc": [
        "blanc",
        "blanche",
        "blancs",
        "blanches"
    ],

    "noir": [
        "noir",
        "noire",
        "noirs",
        "noires"
    ],

    "gris": [
        "gris",
        "grise",
        "grises"
    ],

    "bleu": [
        "bleu",
        "bleue",
        "bleus",
        "bleues"
    ],

    "vert": [
        "vert",
        "verte",
        "verts",
        "vertes"
    ],

    "jaune": [
        "jaune",
        "jaunes"
    ],

    "rouge": [
        "rouge",
        "rouges"
    ],

    "rose": [
        "rose",
        "roses"
    ],

    "violet": [
        "violet",
        "violette",
        "violets",
        "violettes"
    ],

    "orange": [
        "orange",
        "oranges"
    ],

    "marron": [
        "marron",
        "marrons"
    ],

    "brun": [
        "brun",
        "brune",
        "bruns",
        "brunes"
    ],

    "beige": [
        "beige",
        "beiges"
    ],

    "turquoise": [
        "turquoise",
        "turquoises"
    ],

    "taupe": [
        "taupe",
        "taupes"
    ],

    "ivoire": [
        "ivoire",
        "ivoires"
    ],

    "écru": [
        "écru",
        "écrue",
        "écrus",
        "écrues"
    ],

    "ecru": [
        "ecru",
        "ecrue",
        "ecrus",
        "ecrues"
    ],

    "crème": [
        "crème",
        "crèmes"
    ],

    "creme": [
        "creme",
        "cremes"
    ],

    "terracotta": [
        "terracotta"
    ],
}


print(
    f"Nombre de couleurs composées : "
    f"{len(COMPOUND_COLORS)}"
)

print(
    f"Nombre de couleurs simples : "
    f"{len(SIMPLE_COLORS)}"
)

print(
    f"Nombre de couleurs au total : "
    f"{len(COMPOUND_COLORS) + len(SIMPLE_COLORS)}"
)


# ============================================================
# ÉTAPE 4 — FONCTION UTILITAIRE
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 4 — FONCTION D'EXTRACTION DES COULEURS")
print("=" * 70)


def contains_word(text, word):
    """
    Vérifie si un mot ou une expression apparaît
    comme terme complet dans le texte.
    """

    pattern = rf"(?<!\w){re.escape(word)}(?!\w)"

    return re.search(pattern, text) is not None


def extract_colors(text):

    """
    Extrait les couleurs présentes dans un produit.

    Priorité :
    1. couleurs composées
    2. couleurs simples

    Les variantes grammaticales sont normalisées.

    Exemples :
        "chaise noire" -> ["noir"]
        "chaises noires" -> ["noir"]
        "canapé bleu marine" -> ["bleu marine"]
        "chaise gris anthracite" -> ["gris anthracite"]
        "meuble blanc et noir" -> ["blanc", "noir"]
    """

    if not text:
        return []

    text = str(text).lower()

    detected_colors = []

    # --------------------------------------------------------
    # 1. COULEURS COMPOSÉES
    # --------------------------------------------------------

    for color in COMPOUND_COLORS:

        if contains_word(text, color):

            detected_colors.append(color)

    # --------------------------------------------------------
    # 2. COULEURS SIMPLES
    # --------------------------------------------------------

    for normalized_color, variants in COLOR_VARIANTS.items():

        found = False

        for variant in variants:

            if contains_word(text, variant):

                found = True
                break

        if not found:
            continue

        # ----------------------------------------------------
        # Vérifier si la couleur simple fait déjà partie
        # d'une couleur composée détectée.
        #
        # Exemple :
        # "bleu marine"
        #
        # On a déjà :
        # ["bleu marine"]
        #
        # Donc on ne rajoute pas :
        # "bleu"
        # ----------------------------------------------------

        covered_by_compound = False

        for compound_color in detected_colors:

            compound_words = compound_color.split()

            if normalized_color in compound_words:

                covered_by_compound = True
                break

        if not covered_by_compound:

            detected_colors.append(normalized_color)

    # --------------------------------------------------------
    # Suppression des doublons
    # --------------------------------------------------------

    detected_colors = list(
        dict.fromkeys(detected_colors)
    )

    return detected_colors


# ============================================================
# ÉTAPE 5 — TEST DE L'EXTRACTION DES COULEURS
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 5 — TEST DE L'EXTRACTION DES COULEURS")
print("=" * 70)


test_products = [

    "Matelas mousse 140x190 cm blanc",

    "Canapé bleu marine",

    "Chaise gris anthracite",

    "Table à manger chêne naturel",

    "Meuble TV noir et blanc",

    "Canapé bleu ciel et gris clair",

    "Fauteuil vert olive",

    "Chaise rose poudré",

    "Lot de 4 chaises mia noires",

    "Meuble à chaussures 3 portes blanches",

    "Lot de 6 chaises grises",

    "Table blanche et noire",

    "Canapé rouge",

    "Fauteuils vertes",

]


for product in test_products:

    colors = extract_colors(product)

    print(f"\nProduit : {product}")
    print(f"Couleurs détectées : {colors}")


# ============================================================
# ÉTAPE 6 — EXTRACTION DES DIMENSIONS
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 6 — EXTRACTION DES DIMENSIONS")
print("=" * 70)


# ------------------------------------------------------------
# Dimensions composées
#
# Exemples :
# 140x190
# 140 x 190
# 140×190
# 140*190
# 120x60x90
# ------------------------------------------------------------

DIMENSION_PATTERN = re.compile(
    r"""
    (?P<d1>\d+(?:[.,]\d+)?)
    \s*
    [x×*]
    \s*
    (?P<d2>\d+(?:[.,]\d+)?)

    (?:
        \s*
        [x×*]
        \s*
        (?P<d3>\d+(?:[.,]\d+)?)
    )?

    \s*
    (?P<unit>cm|mm|m)?
    """,
    re.IGNORECASE | re.VERBOSE
)


# ------------------------------------------------------------
# Dimension simple
#
# Exemples :
# 150 cm
# 150cm
# 1.5 m
# 140 mm
# ------------------------------------------------------------

SINGLE_DIMENSION_PATTERN = re.compile(
    r"""
    (?<![\dx×*])
    (?P<value>\d+(?:[.,]\d+)?)
    \s*
    (?P<unit>cm|mm|m)
    \b
    """,
    re.IGNORECASE | re.VERBOSE
)


def normalize_number(value):

    """
    Normalise un nombre.

    Exemples :
        140      -> 140
        140.0    -> 140
        140,5    -> 140.5
        120,50   -> 120.5
    """

    value = value.replace(",", ".")

    number = float(value)

    if number.is_integer():

        return str(int(number))

    return str(number)


def extract_dimensions(text):
    """
    Extrait et normalise les dimensions.

    Exemples :
        140x190 cm
        -> 140*190 cm

        140 x 190
        -> 140*190

        160X200cm
        -> 160*200 cm

        120 × 60 × 90 cm
        -> 120*60*90 cm

        150 cm
        -> 150 cm

    Une dimension simple appartenant déjà à une dimension
    composée n'est pas ajoutée une deuxième fois.
    """

    if not text:
        return []

    text = str(text).lower()

    dimensions = []

    # Positions occupées par les dimensions composées
    # Exemple :
    # "120 × 80 cm"
    #  ^^^^^^^^^^
    # Cette zone sera ignorée par la recherche
    # des dimensions simples.
    covered_spans = []

    # ========================================================
    # 1. DIMENSIONS COMPOSÉES
    # ========================================================

    for match in DIMENSION_PATTERN.finditer(text):

        d1 = normalize_number(
            match.group("d1")
        )

        d2 = normalize_number(
            match.group("d2")
        )

        d3 = match.group("d3")

        unit = match.group("unit")

        if d3:
            d3 = normalize_number(d3)

            dimension = (
                f"{d1}*{d2}*{d3}"
            )

        else:
            dimension = (
                f"{d1}*{d2}"
            )

        if unit:
            dimension = (
                f"{dimension} {unit}"
            )

        dimensions.append(dimension)

        # Mémoriser la zone correspondant à cette
        # dimension composée
        covered_spans.append(
            (
                match.start(),
                match.end()
            )
        )

    # ========================================================
    # 2. DIMENSIONS SIMPLES
    # ========================================================

    for match in SINGLE_DIMENSION_PATTERN.finditer(text):

        # Vérifier si cette dimension simple se trouve
        # déjà à l'intérieur d'une dimension composée.
        is_inside_compound = False

        for start, end in covered_spans:

            if (
                match.start() >= start
                and match.end() <= end
            ):
                is_inside_compound = True
                break

        # Si elle appartient déjà à une dimension composée,
        # on ne l'ajoute pas.
        if is_inside_compound:
            continue

        value = normalize_number(
            match.group("value")
        )

        unit = match.group("unit").lower()

        dimension = f"{value} {unit}"

        dimensions.append(dimension)

    # ========================================================
    # 3. SUPPRESSION DES DOUBLONS
    # ========================================================

    dimensions = list(
        dict.fromkeys(dimensions)
    )

    return dimensions


# ============================================================
# ÉTAPE 7 — TEST DES DIMENSIONS
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 7 — TEST DES DIMENSIONS")
print("=" * 70)


test_dimensions = [

    "Matelas mousse 140x190 cm",

    "Matelas 140 x 190",

    "Lit 160X200cm",

    "Table 120*80 cm",

    "Table 120 × 80 cm",

    "Meuble 120x60x90 cm",

    "Canapé 200 × 90 × 85 cm",

    "Bureau 120,5 x 60,5 cm",

    "Ours en peluche géant 150 cm",

    "Produit 100cm",

    "Table 140cm",

    "Produit sans dimension",

]


for product in test_dimensions:

    dimensions = extract_dimensions(product)

    print(f"\nProduit : {product}")
    print(f"Dimensions détectées : {dimensions}")


# ============================================================
# ÉTAPE 8 — EXTRACTION SUR LES PRODUITS UNIQUES
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 8 — EXTRACTION SUR LES PRODUITS UNIQUES")
print("=" * 70)


color_results = {}
dimension_results = {}

products_without_color = []
products_without_dimension = []

total_products = len(unique_products)


for i, product in enumerate(
    unique_products,
    start=1
):

    # --------------------------------------------------------
    # COULEURS
    # --------------------------------------------------------

    colors = extract_colors(product)

    color_results[product] = colors

    if not colors:

        products_without_color.append(
            product
        )

    # --------------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------------

    dimensions = extract_dimensions(product)

    dimension_results[product] = dimensions

    if not dimensions:

        products_without_dimension.append(
            product
        )

    # --------------------------------------------------------
    # PROGRESSION
    # --------------------------------------------------------

    if (
        i % 1000 == 0
        or i == total_products
    ):

        print(
            f"Progression : "
            f"{i}/{total_products}"
        )


# ============================================================
# ÉTAPE 9 — STATISTIQUES
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 9 — STATISTIQUES")
print("=" * 70)


products_with_color = sum(
    1
    for colors in color_results.values()
    if colors
)

products_with_dimension = sum(
    1
    for dimensions in dimension_results.values()
    if dimensions
)


products_without_color_count = (
    total_products
    - products_with_color
)

products_without_dimension_count = (
    total_products
    - products_with_dimension
)


print(
    f"Produits uniques : "
    f"{total_products}"
)

print(
    f"Produits avec couleur : "
    f"{products_with_color}"
)

print(
    f"Produits sans couleur : "
    f"{products_without_color_count}"
)

print(
    f"Taux de détection couleur : "
    f"{products_with_color / total_products * 100:.2f}%"
)


print()

print(
    f"Produits avec dimension : "
    f"{products_with_dimension}"
)

print(
    f"Produits sans dimension : "
    f"{products_without_dimension_count}"
)

print(
    f"Taux de détection dimension : "
    f"{products_with_dimension / total_products * 100:.2f}%"
)


# ============================================================
# ÉTAPE 10 — PRODUITS SANS COULEUR
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 10 — PRODUITS SANS COULEUR")
print("=" * 70)


print(
    f"Nombre de produits sans couleur : "
    f"{len(products_without_color)}"
)

print("\nPremiers produits sans couleur :")

for i, product in enumerate(
    products_without_color[:30],
    start=1
):

    print(f"{i}. {product}")


# ============================================================
# ÉTAPE 11 — PRODUITS SANS DIMENSION
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 11 — PRODUITS SANS DIMENSION")
print("=" * 70)


print(
    f"Nombre de produits sans dimension : "
    f"{len(products_without_dimension)}"
)

print("\nPremiers produits sans dimension :")

for i, product in enumerate(
    products_without_dimension[:30],
    start=1
):

    print(f"{i}. {product}")


# ============================================================
# ÉTAPE 12 — APERÇU DES RÉSULTATS
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 12 — APERÇU DES RÉSULTATS")
print("=" * 70)


for product in unique_products[:20]:

    print("\n----------------------------------------")

    print(
        f"Produit : {product}"
    )

    print(
        f"Couleurs : "
        f"{color_results[product]}"
    )

    print(
        f"Dimensions : "
        f"{dimension_results[product]}"
    )


# ============================================================
# ÉTAPE 13 — CRÉATION DU DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 13 — CRÉATION DU DATAFRAME")
print("=" * 70)


results_df = pd.DataFrame({

    "Libellé produit": unique_products,

    "Couleurs": [
        color_results[product]
        for product in unique_products
    ],

    "Dimensions": [
        dimension_results[product]
        for product in unique_products
    ]

})


print("\nDimensions du DataFrame :")

print(results_df.shape)


print("\nAperçu :")

print(
    results_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# ÉTAPE 14 — SAUVEGARDE
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 14 — SAUVEGARDE")
print("=" * 70)


output_file = (
    "../data/output/"
    "extraction_attributes.csv"
)


os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
)


results_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"Résultats sauvegardés dans : "
    f"{output_file}"
)


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("TERMINÉ")
print("=" * 70)