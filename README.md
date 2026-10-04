# 🛒 E-commerce Categorization & Product Attribute Extraction

## 📌 Présentation

Ce projet a pour objectif de **catégoriser automatiquement des produits e-commerce** à partir de leur libellé et d'extraire des **attributs produits** tels que les couleurs et les dimensions.

Le projet repose sur deux grandes briques :

1. **Catégorisation sémantique** des produits par embeddings et similarité cosinus ;
2. **Extraction d'attributs** à partir des descriptions produits.

Une troisième étape permet ensuite d'exploiter les résultats de la catégorisation sémantique pour construire un dataset suffisamment fiable et représentatif afin d'entraîner des **modèles de classification supervisée**.

L'approche globale est donc :

```text
Descriptions produits
        │
        ▼
Prétraitement
        │
        ▼
Définitions des catégories (Nature)
        │
        ▼
Embeddings des catégories
        │
        ▼
Embeddings des produits
        │
        ▼
Similarité sémantique
        │
        ▼
Nature prédite + score
        │
        ▼
Comparaison avec la Nature existante
        │
        ▼
Dataset de confiance
        │
        ▼
Enrichissement manuel des classes rares/absentes
        │
        ▼
Classification supervisée
        │
        ▼
Modèle final
```

En parallèle :

```text
Description produit
        │
        ├──► Extraction des couleurs
        │
        └──► Extraction des dimensions
```

Les deux pipelines sont ensuite fusionnés afin de produire un dataset enrichi.

---

# 📂 Structure du projet

```text
ecommerce_categorization_v3/
│
├── data/
│   ├── input/
│   │   └── 20210614 Ecommerce sales.xlsb
│   │
│   └── output/
│       ├── embeddings_natures.csv
│       ├── embeddings_produits_uniques.csv
│       ├── predictions_525034.csv
│       ├── dataset_confiance.csv
│       ├── dataset_confiance_enrichi.csv
│       ├── natures_manquantes.csv
│       ├── distribution_natures_confiance.csv
│       ├── extraction_attributes.csv
│       ├── dataset_enrichi.csv
│       └── classification_results/
│
├── src/
│   ├── categorisation.py
│   ├── extraction.py
│   └── classify.py
│
├── notebooks/
│   └── analyse_results.ipynb
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 📊 Données

Le dataset contient environ :

* **525 034 produits**
* **596 catégories `Nature`**
* **55 830 libellés produits uniques**

La colonne `Nature` représente la catégorie produit existante dans les données.

Exemple conceptuel :

| Libellé produit        | Nature         |
| ---------------------- | -------------- |
| meuble TV bois blanc   | Meuble TV      |
| chaise de bureau noire | Chaise bureau  |
| table à manger 140 cm  | Table à manger |

L'objectif est de retrouver automatiquement cette catégorie à partir du **sens du libellé produit**.

---

# 1. 🧠 Catégorisation sémantique

## 1.1 Pourquoi une catégorisation sémantique ?

Une simple recherche lexicale peut échouer lorsqu'un produit et sa catégorie utilisent des formulations différentes.

Par exemple :

```text
Produit :
"meuble télévision contemporain avec portes"

Catégorie :
"Meuble TV"
```

Les mots ne sont pas nécessairement identiques, mais les deux textes décrivent le même concept.

L'approche utilisée consiste donc à représenter les produits et les catégories sous forme de **vecteurs numériques appelés embeddings**.

Deux textes ayant un sens proche doivent produire des vecteurs proches dans l'espace vectoriel.

---

# 2. 🏷️ Préparation des catégories `Nature`

Le dataset contient **596 Natures différentes**.

Une première approche consistant uniquement à utiliser le nom de la catégorie peut être insuffisante.

Par exemple :

```text
Nature :
"Accessoire aspi"
```

est une information relativement courte et ambiguë pour un modèle d'embeddings.

Pour améliorer la représentation sémantique des catégories, une **définition textuelle est générée automatiquement pour chaque Nature**.

---

## 2.1 Génération des définitions

Un LLM local est utilisé pour générer une définition courte et adaptée à la classification e-commerce.

Modèle utilisé :

```text
llama3.2:3b
```

Pour chaque Nature, le modèle reçoit une instruction du type :

```text
La catégorie produit est :
"Accessoire aspi"

Donne une définition courte, précise et adaptée
à la classification e-commerce de cette catégorie.
```

Le résultat contient une description permettant de mieux caractériser la catégorie.

Exemple conceptuel :

```text
Nature :
"Accessoire aspi"

Définition :
"Accessoires destinés aux aspirateurs permettant
d'améliorer ou compléter leur utilisation, comme
les embouts, brosses ou tubes."
```

Cette définition apporte donc au modèle davantage d'informations sémantiques que le simple nom de la catégorie.

---

# 3. 🔢 Construction du texte représentant chaque Nature

Pour chaque catégorie, le texte utilisé pour calculer l'embedding est construit sous la forme :

```text
Catégorie : {Nature}
Définition : {Definition}
```

Exemple :

```text
Catégorie : Meuble TV
Définition : Meuble destiné à accueillir une télévision
et généralement à organiser les équipements multimédias.
```

Ce texte enrichi est ensuite transformé en embedding.

---

# 4. 🔤 Embeddings des Natures

Le modèle utilisé pour les embeddings est :

```text
bge-m3
```

Chaque Nature est transformée en un vecteur de dimension :

```text
1024
```

Ainsi, les 596 catégories sont représentées par une matrice :

```text
596 × 1024
```

Les embeddings sont sauvegardés dans :

```text
data/output/embeddings_natures.csv
```

Le fichier contient notamment :

```text
Nature
Definition
Embedding
```

Un cache est utilisé afin d'éviter de recalculer les embeddings déjà générés.

---

# 5. 🛍️ Prétraitement des descriptions produits

Avant de calculer les embeddings des produits, les libellés sont nettoyés.

Les principales opérations sont :

* remplacement des valeurs manquantes ;
* conversion en chaînes de caractères ;
* passage en minuscules ;
* suppression des caractères de contrôle ;
* normalisation des espaces ;
* suppression des espaces inutiles.


# 6. 🔢 Embeddings des produits

Chaque libellé produit unique est également transformé en embedding avec :

```text
bge-m3
```

Les descriptions identiques ne sont calculées qu'une seule fois.

Le dataset contient :

```text
525 034 lignes
```

mais seulement :

```text
55 830 descriptions uniques
```

Cela permet d'éviter de recalculer inutilement les mêmes embeddings.

Le nombre de calculs évités est donc d'environ :

```text
525 034 - 55 830 = 469 204
```

Les embeddings sont sauvegardés dans :

```text
data/output/embeddings_produits_uniques.csv
```

---

# 7. 📐 Similarité sémantique

Une fois les embeddings des produits et des Natures calculés, chaque produit est comparé aux **596 catégories**.

La mesure utilisée est la **similarité cosinus**.

Pour un produit \(p\) et une catégorie \(c\) :

$$
similarity(p,c)
=
\frac{p \cdot c}
{\|p\|\|c\|}
$$

Plus la similarité est élevée, plus le produit et la catégorie sont considérés comme sémantiquement proches.

---

# 8. 🎯 Prédiction de la Nature

Pour chaque produit, les similarités avec les 596 Natures sont calculées.

La catégorie ayant le score maximal est sélectionnée :

$$
\hat{y}
=
\arg\max_{c \in C}
similarity(p,c)
$$

où :

* \(p\) = embedding du produit ;
* \(C\) = ensemble des 596 Natures ;
* \(\hat{y}\) = Nature prédite.

Le résultat contient donc :

```text
Libellé produit
Prediction
Score
Nature
```

Exemple :

```text
Libellé produit :
"table à manger extensible blanche 140 cm"

Prediction :
"Table à manger"

Score :
0.78

Nature :
"Table à manger"
```

---

# 9. 💾 Gestion du calcul des similarités

Le calcul direct de toutes les similarités représenterait une matrice :

```text
525 034 × 596
```

soit plus de 300 millions de comparaisons.

Pour éviter une consommation excessive de mémoire, les produits sont traités **par blocs**.

Le calcul utilise notamment :

```text
SIMILARITY_BATCH_SIZE=1000
```

Les vecteurs étant normalisés, la similarité cosinus peut être calculée efficacement par produit matriciel.

---

# 10. 📈 Résultats de la catégorisation sémantique

La catégorisation a été réalisée sur :

```text
525 034 produits
```

Résultats :

| Résultat             |  Nombre | Pourcentage |
| -------------------- | ------: | ----------: |
| Prediction = Nature  | 244 539 |     46,58 % |
| Prediction != Nature | 280 495 |     53,42 % |
| Total                | 525 034 |       100 % |

La prédiction sémantique correspond donc à la Nature déjà présente dans le dataset pour :

```text
244 539 produits
```

soit :

```text
46,58 %
```

Les prédictions sont sauvegardées dans :

```text
data/output/predictions_525034.csv
```

---

# 11. ⚠️ Interprétation du taux de 46,58 %

Le taux de **46,58 % ne doit pas être interprété comme une accuracy classique**.

La colonne `Nature` utilisée dans le dataset est la catégorie existante dans les données, mais elle n'est pas considérée ici comme une vérité terrain entièrement validée.

Le résultat mesure plutôt :

> le taux d'accord entre la catégorie obtenue par similarité sémantique et la catégorie `Nature` existante.

Cette distinction est importante pour l'interprétation des résultats.

---

# 12. 🟢 Construction du dataset de confiance

Les produits pour lesquels :

```python
Prediction == Nature
```

sont conservés dans un dataset de confiance.

Ce dataset contient :

```text
244 539 produits
```

et est sauvegardé dans :

```text
data/output/dataset_confiance.csv
```

Ce dataset est appelé **dataset de confiance** car les produits sélectionnés présentent une cohérence entre :

* la description du produit ;
* la représentation sémantique du produit ;
* la catégorie prédite ;
* la catégorie `Nature` existante.

Cependant, il s'agit d'un **dataset pseudo-labellisé**, et non d'un dataset de vérité terrain entièrement annoté manuellement.

---

# 13. ⚖️ Problème de déséquilibre des classes

Les 244 539 produits du dataset de confiance ne sont pas répartis uniformément entre les 596 Natures.

Résultats observés :

```text
503 / 596 Natures
```

sont représentées dans le dataset de confiance.

Il reste donc :

```text
93 Natures
```

absentes du dataset de confiance.

De plus :

```text
149 Natures
```

possèdent moins de 10 exemples.

Et :

```text
96 Natures
```

possèdent moins de 5 exemples.

Cette distribution très déséquilibrée constitue une limitation importante pour l'apprentissage supervisé.

---

# 14. 📝 Enrichissement manuel du dataset

Les classes absentes ou très peu représentées doivent être enrichies manuellement.

L'objectif est de créer :

```text
dataset_confiance_enrichi.csv
```

qui combine :

```text
Dataset de confiance
        +
Exemples annotés manuellement
        ↓
Dataset d'apprentissage plus équilibré
```

Cette étape permet notamment d'ajouter des exemples pour les Natures absentes de la catégorisation sémantique.

L'annotation manuelle constitue donc une étape de validation et d'enrichissement avant la classification supervisée.

---

# 15. 🤖 Classification supervisée

La classification supervisée intervient **après la catégorisation sémantique**.

Elle ne remplace donc pas la première étape.

Son objectif est d'apprendre, à partir du dataset enrichi, une fonction permettant de prédire directement la Nature d'un nouveau produit.

Pipeline :

```text
Description produit
        │
        ▼
BGE-M3 embedding
        │
        ▼
Vecteur 1024 dimensions
        │
        ▼
Modèle supervisé
        │
        ▼
Nature prédite
```

Le script utilisé est :

```text
src/classify.py
```

---

# 16. 🧠 Modèles supervisés

Trois modèles sont évalués :

### Logistic Regression


### Linear SVM



### MLP

```text
MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    early_stopping=True
)
```

---

# 17. 🧪 Séparation train / test

Le dataset est séparé en :

```text
80 % → entraînement
20 % → test
```

La séparation est stratifiée afin de conserver autant que possible la distribution des classes.

Les classes possédant trop peu d'exemples sont traitées séparément afin d'éviter des problèmes lors de la séparation stratifiée.

---

# 18. 📊 Évaluation des modèles

L'accuracy seule n'est pas suffisante en raison du fort déséquilibre entre les Natures.

Les métriques utilisées sont :

* Accuracy ;
* Balanced Accuracy ;
* Precision Macro ;
* Recall Macro ;
* F1 Macro ;
* F1 Weighted.

La métrique principale utilisée pour comparer les modèles est :

```text
F1 Macro
```

car elle donne le même poids à chaque catégorie, y compris les catégories minoritaires.

Une **matrice de confusion** et un **classification report par classe** sont également générés.

---

# 19. 📁 Résultats de la classification supervisée

Les résultats sont sauvegardés dans :

```text
data/output/classification_results/
```


Le meilleur modèle est sélectionné principalement selon le :

```text
F1 Macro
```

---

# 20. 🎨 Extraction des attributs produits

En parallèle de la catégorisation, le projet extrait des attributs directement depuis les descriptions.

Les attributs actuellement étudiés sont :

* couleurs ;
* dimensions.

Le script utilisé est :

```text
src/extraction.py
```

---

## 20.1 Extraction des couleurs

L'extraction utilise des expressions régulières et un dictionnaire de couleurs.

Les couleurs composées sont traitées avant les couleurs simples.

Exemples :

```text
bleu marine
bleu pétrole
bleu canard
vert sauge
gris anthracite
rose poudré
```

ainsi que :

```text
blanc
noir
gris
beige
marron
rouge
bleu
vert
rose
...
```

Exemple :

```text
"meuble tv falko bois blanc et gris"
```

donne :

```text
["blanc", "gris"]
```

---

# 21. 📏 Extraction des dimensions

Les dimensions sont détectées avec des expressions régulières.

Formats pris en charge :

```text
120 x 80 cm
120 × 80 cm
200 x 90 x 85 cm
120,5 x 60,5 cm
150 cm
```

Exemples :

```text
120 × 80 cm
→ 120*80 cm

200 × 90 × 85 cm
→ 200*90*85 cm

120,5 x 60,5 cm
→ 120.5*60.5 cm

150 cm
→ 150 cm
```

---

# 22. 📊 Résultats de l'extraction

L'extraction a été réalisée sur :

```text
55 830 descriptions uniques
```

Résultats :

### Couleurs

```text
25 982 descriptions
```

contiennent au moins une couleur.

Soit :

```text
46,54 %
```

### Dimensions

```text
29 016 descriptions
```

contiennent au moins une dimension.

Soit :

```text
51,97 %
```

Les résultats sont sauvegardés dans :

```text
data/output/extraction_attributes.csv
```

---

# 23. 🔗 Fusion des résultats

Les résultats de catégorisation et d'extraction sont finalement fusionnés.

Le dataset final contient notamment :

```text
Libellé produit
Prediction
Score
Nature
Correct
Couleurs
Dimensions
```

Le fichier final est :

```text
data/output/dataset_enrichi.csv
```

Il permet donc d'associer à chaque produit :

* sa catégorie existante ;
* sa catégorie prédite ;
* son score sémantique ;
* l'indicateur d'accord ;
* ses couleurs détectées ;
* ses dimensions détectées.

---

# 24. 📓 Analyse dans le notebook

Le notebook permet de visualiser et analyser les résultats.

Les analyses comprennent notamment :

### Catégorisation

* distribution des Natures ;
* taux d'accord `Prediction == Nature` ;
* distribution des scores ;
* scores selon les prédictions correctes/incorrectes ;
* erreurs de catégorisation ;
* couples `Nature → Prediction` ;
* classes rares ;
* classes absentes.

### Extraction

* distribution des couleurs ;
* nombre de couleurs par produit ;
* distribution des dimensions ;
* taux d'extraction par Nature ;
* comparaison extraction / catégorisation.

---

# 25. ⚙️ Configuration

Les paramètres sont centralisés dans `.env` :

```env
OLLAMA_HOST=http://localhost:11434

OLLAMA_LLM_MODEL=llama3.2:3b

OLLAMA_EMBEDDING_MODEL=bge-m3

OLLAMA_BATCH_SIZE=32

SIMILARITY_BATCH_SIZE=1000

CLASSIFICATION_BATCH_SIZE=32

CLASSIFICATION_TEST_SIZE=0.20

CLASSIFICATION_RANDOM_STATE=42

CLASSIFICATION_MIN_CLASS_SAMPLES=2
```


---

# 26. 📦 Installation

Créer l'environnement virtuel :

```powershell
python -m venv .venv
```

Activer l'environnement :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
pip install -r requirements.txt
```

Ollama doit également être installé et lancé.

Vérifier les modèles :

```powershell
ollama list
```

Les modèles utilisés sont :

```text
llama3.2:3b
bge-m3
```

---

# 27. ▶️ Exécution du projet

## Étape 1 — Catégorisation sémantique

```powershell
python src/categorisation.py
```

Cette étape produit :

```text
embeddings_natures.csv
embeddings_produits_uniques.csv
predictions_525034.csv
dataset_confiance.csv
```

---

## Étape 2 — Analyse et enrichissement manuel

Analyser :

```text
dataset_confiance.csv
```

Identifier :

* Natures absentes ;
* Natures rares ;
* erreurs de catégorisation ;
* classes nécessitant davantage d'exemples.

Créer ensuite :

```text
dataset_confiance_enrichi.csv
```

---

## Étape 3 — Classification supervisée

Une fois le dataset enrichi disponible :

```powershell
python src/classify.py
```

Les différents modèles sont entraînés et comparés.

---

## Étape 4 — Extraction des attributs

```powershell
python src/extraction.py
```

Cette étape génère :

```text
extraction_attributes.csv
```

---

## Étape 5 — Fusion

Les résultats sont fusionnés dans le notebook afin de produire :

```text
dataset_enrichi.csv
```

---

# 28. 🧩 Architecture globale

Le projet peut être résumé comme suit :

```text
                         DATASET E-COMMERCE
                                │
                                ▼
                    ┌──────────────────────┐
                    │ Prétraitement        │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       CATÉGORISATION                 EXTRACTION
        SÉMANTIQUE                    ATTRIBUTS
                 │                           │
                 ▼                           ├── Couleurs
       Définitions Nature                    │
                 │                           └── Dimensions
                 ▼
          Embeddings
          des Natures
                 │
                 ▼
          Embeddings
          des produits
                 │
                 ▼
       Similarité cosinus
                 │
                 ▼
       Nature prédite + score
                 │
                 ▼
     Prediction == Nature ?
                 │
                 ▼
       Dataset de confiance
                 │
                 ▼
       Enrichissement manuel
                 │
                 ▼
    Dataset confiance enrichi
                 │
                 ▼
      CLASSIFICATION SUPERVISÉE
                 │
          ┌──────┼──────┐
          ▼      ▼      ▼
         LR     SVM     MLP
          │      │      │
          └──────┼──────┘
                 ▼
          Meilleur modèle
                 │
                 ▼
        Nature prédite finale

               
```

---


---


---

# 31. 📌 État actuel du projet

| Étape                              | État |
| ---------------------------------- | ---- |
| Chargement des données             | ✅    |
| Prétraitement                      | ✅    |
| Identification des 596 Natures     | ✅    |
| Génération des définitions         | ✅    |
| Embeddings des Natures             | ✅    |
| Embeddings des produits            | ✅    |
| Catégorisation sémantique          | ✅    |
| Similarité cosinus                 | ✅    |
| Dataset de confiance               | ✅    |
| Analyse des classes absentes/rares | ✅    |
| Enrichissement manuel              | 🔄   |
| Dataset confiance enrichi          | 🔄   |
| Classification supervisée          | 🔄   |
| Extraction des couleurs            | ✅    |
| Extraction des dimensions          | ✅    |
| Fusion des résultats               | ✅    |
| Analyse notebook                   | 🔄   |

---


