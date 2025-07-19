# 🎯 Module 3 : Classification et Analyse de Sentiments
## *Créer une IA qui détecte les émotions comme un psychologue*

---

## 📹 **Vidéo d'Introduction** *(5 minutes)*

> **🎬 Script de la vidéo :**
> 
> "Salut les futurs experts NLP ! Imaginez pouvoir analyser automatiquement des milliers d'avis clients en quelques secondes pour savoir s'ils sont contents ou furieux. C'est exactement ce qu'on va créer dans ce module !
> 
> Vous allez apprendre les secrets des algorithmes de classification - ces petites merveilles mathématiques qui transforment votre machine en détective émotionnel. De Naive Bayes (oui, il est vraiment 'naïf' mais diablement efficace !) aux forêts aléatoires, vous maîtriserez l'art de faire comprendre les nuances humaines à un ordinateur.
> 
> À la fin de ce module, vous aurez créé un analyseur de sentiments capable de traiter du texte en temps réel et de vous dire si vos utilisateurs sont aux anges ou sur le point d'exploser. Prêts à devenir des chuchoteurs d'émotions digitales ? C'est parti !"

**🎯 Ce que vous allez apprendre :**
- Transformer des textes en catégories (positif/négatif/neutre)
- Entraîner des modèles qui "comprennent" les émotions
- Évaluer la performance de vos IA avec des métriques pro
- Créer un analyseur de sentiments déployable

**⏱️ Durée estimée :** 4-5 heures  
**🎖️ Niveau :** Intermédiaire  
**🛠️ Prérequis :** Modules 1 & 2 terminés

---

## 📋 **Plan du Module**

| Section | Contenu | Durée | Type |
|---------|---------|-------|------|
| **Théorie** | 5 chapitres concepts | 2h | 📚 Lecture |
| **Exercices** | 4 exercices pratiques | 2h | 💻 Code |
| **Projet** | Analyseur complet | 1h | 🚀 Intégration |

---

## 📚 **1. Classification Supervisée - Les Fondations**

### 🎯 **Objectif**
Comprendre comment une machine peut apprendre à catégoriser automatiquement du texte en se basant sur des exemples.

### 📖 **Le Principe de Base**

L'apprentissage supervisé, c'est comme apprendre à reconnaître les races de chiens :

```
👨‍🏫 PHASE D'ENTRAÎNEMENT
Humain : "Ça c'est un Labrador" (montre photo + étiquette)
Humain : "Ça c'est un Bulldog" (montre photo + étiquette)
Machine : *analyse les patterns* 🤖

🎯 PHASE DE PRÉDICTION  
Humain : "C'est quelle race ?" (montre nouvelle photo)
Machine : "Je pense que c'est un Labrador à 85%" 🎲
```

**En NLP, on fait pareil avec le texte :**

```python
# Phase d'entraînement
textes_exemples = [
    "Ce produit est fantastique !",      # → positif
    "Très déçu de cet achat",           # → négatif  
    "Ça va, sans plus",                 # → neutre
]

# Phase de prédiction
nouveau_texte = "J'adore cette app !"
prediction = modele.predict(nouveau_texte)  # → positif (92%)
```

### 🔢 **Types de Classification**

#### **Classification Binaire**
2 catégories seulement (comme un interrupteur ON/OFF)
```
😊 Positif  |  😡 Négatif
```

#### **Classification Multi-Classe**
3+ catégories exclusives (comme choisir une couleur)
```
😊 Positif  |  😡 Négatif  |  😐 Neutre
```

#### **Classification Multi-Label**
Plusieurs étiquettes possibles (comme les tags d'un article)
```
😊 Positif + 😍 Enthousiaste + 🛒 Achat
```

### 📊 **Train/Validation/Test Split**

**Pourquoi diviser ses données ?**

Imaginez que vous préparez un examen :
- **📚 Train (70%)** : Vos cours pour apprendre
- **📝 Validation (15%)** : Vos exercices pour vous tester
- **🎓 Test (15%)** : L'examen final (jamais vu avant !)

```python
from sklearn.model_selection import train_test_split

# Division intelligente des données
X_train, X_temp, y_train, y_temp = train_test_split(
    textes, sentiments, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"📚 Train: {len(X_train)} exemples")
print(f"📝 Validation: {len(X_val)} exemples") 
print(f"🎓 Test: {len(X_test)} exemples")
```

### ⚠️ **Pièges Courants à Éviter**

**🚫 Data Leakage :** Utiliser les données de test pendant l'entraînement
```python
# ❌ MAUVAIS
modele.fit(X_train + X_test, y_train + y_test)  # Triche !

# ✅ BON  
modele.fit(X_train, y_train)  # Apprentissage honnête
```

**🚫 Overfitting :** Apprendre par cœur au lieu de comprendre
```python
# Symptôme : 99% sur train, 60% sur test
# Solution : régularisation, plus de données, validation croisée
```

---

## 📊 **2. Préparation des Données - La Cuisine Secrète**

### 🎯 **Objectif**
Transformer vos données brutes en festin pour algorithmes affamés.

### 📋 **Collecte et Annotation**

#### **Sources de Données Textuelles**
```python
sources_donnees = {
    "🛒 E-commerce": ["Amazon", "Fnac", "Cdiscount"],
    "🎬 Divertissement": ["AlloCiné", "Netflix", "Spotify"], 
    "🏨 Services": ["TripAdvisor", "Booking", "Uber"],
    "📱 Apps": ["App Store", "Google Play"],
    "🐦 Social": ["Twitter", "Facebook", "Reddit"]
}
```

#### **Annotation Manuelle - Les Règles d'Or**

**📏 Échelle de Sentiment :**
```
😡 Très Négatif (-2) : "Je déteste, c'est nul !"
😞 Négatif (-1)       : "Pas terrible, déçu"  
😐 Neutre (0)         : "Ça va, correct"
🙂 Positif (+1)       : "Bien, satisfait"
😍 Très Positif (+2) : "Fantastique, j'adore !"
```

**🎯 Critères d'Annotation :**
- **Intention** : Que veut exprimer la personne ?
- **Contexte** : Sarcasme, ironie, second degré ?
- **Intensité** : Légèrement vs extrêmement positif/négatif

#### **Cas Complexes**

```python
exemples_pieges = [
    {
        "texte": "Ce produit n'est pas mauvais du tout !",
        "piege": "Double négation = positif",
        "label": "positif"
    },
    {
        "texte": "Vraiment 'fantastique' ce service...",  
        "piege": "Guillemets = sarcasme probable",
        "label": "négatif"
    },
    {
        "texte": "Bon produit mais livraison horrible",
        "piege": "Sentiment mixte sur 2 aspects",
        "label": "neutre"  # ou séparer en 2 phrases
    }
]
```

### ⚖️ **Équilibrage des Classes**

#### **Le Problème du Déséquilibre**
```python
# ❌ Dataset déséquilibré  
distribution_naive = {
    "positif": 8000,   # 80% - Majorité écrasante
    "neutre": 1500,    # 15% 
    "négatif": 500     # 5% - Minorité négligée
}

# Problème : Le modèle va toujours prédire "positif" !
```

#### **Solutions d'Équilibrage**

**🎯 Under-sampling :** Réduire la majorité
```python
from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Résultat : 500 exemples de chaque classe
```

**🎯 Over-sampling :** Augmenter la minorité  
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Résultat : 8000 exemples de chaque classe (SMOTE génère des exemples synthétiques)
```

**🎯 Pondération des Classes :** Dire au modèle que certaines erreurs coûtent plus cher
```python
from sklearn.naive_bayes import MultinomialNB

# Le modèle pénalise plus les erreurs sur les classes rares
modele = MultinomialNB(class_weight='balanced')
```

### 🧹 **Nettoyage Spécialisé pour Classification**

```python
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocesser_pour_classification(texte):
    """
    Pipeline de nettoyage optimisé pour l'analyse de sentiments
    """
    # 1. Préserver les émotions importantes
    texte = re.sub(r'[!]{2,}', ' très_excite ', texte)  # !!! → très_excite
    texte = re.sub(r'[?]{2,}', ' très_confus ', texte)   # ??? → très_confus
    
    # 2. Gérer les négations (CRUCIAL pour sentiments!)
    texte = re.sub(r"n'|ne ", " ne_pas ", texte)
    texte = re.sub(r" pas ", " ne_pas ", texte)
    
    # 3. Normaliser les intensifiants
    texte = re.sub(r'très très', 'extrêmement', texte)
    texte = re.sub(r'super ', 'très ', texte)
    
    # 4. Nettoyer sans détruire le sens
    texte = re.sub(r'http\S+', '', texte)  # URLs
    texte = re.sub(r'@\w+', '', texte)     # Mentions
    texte = re.sub(r'#(\w+)', r'\1', texte)  # Hashtags → mots
    
    # 5. Normalisation finale
    texte = texte.lower()
    texte = re.sub(r'[^\w\s]', ' ', texte)
    texte = ' '.join(texte.split())  # Espaces multiples
    
    return texte

# Test
exemple = "Ce produit n'est pas terrible... vraiment pas !!! #déçu"
print(preprocesser_pour_classification(exemple))
# Output: "ce produit ne_pas est ne_pas terrible vraiment ne_pas très_excite déçu"
```

---

## 🤖 **3. Algorithmes de Classification - L'Arsenal**

### 🎯 **Objectif**  
Maîtriser les 3 algorithmes stars de la classification de texte et savoir quand les utiliser.

### 🧠 **Naive Bayes - Le Génie "Naïf"**

#### **Pourquoi "Naïf" ?**
Il assume que tous les mots sont indépendants (ce qui est faux, mais ça marche !)

```python
# Naive Bayes pense que dans "très bon produit" :
# - "très" n'influence pas "bon"  
# - "bon" n'influence pas "produit"
# C'est naïf, mais statistiquement efficace !
```

#### **Le Principe Mathématique (Simplifié)**
```
P(sentiment|texte) = P(texte|sentiment) × P(sentiment) / P(texte)

Traduction : 
"Probabilité que ce soit positif sachant ce texte"
= 
"Fréquence de ces mots dans les textes positifs" 
× "Proportion de textes positifs globalement"
/ "Fréquence de ces mots au total"
```

#### **Implémentation Pratique**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Préparation des features
vectorizer = TfidfVectorizer(
    max_features=10000,      # Top 10k mots les + fréquents
    ngram_range=(1, 2),      # Mots seuls + bigrammes  
    stop_words='english'     # Retire "le", "de", "et"...
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 2. Entraînement  
nb_model = MultinomialNB(alpha=1.0)  # alpha = lissage de Laplace
nb_model.fit(X_train_vec, y_train)

# 3. Prédiction
predictions = nb_model.predict(X_val_vec)
probabilities = nb_model.predict_proba(X_val_vec)

print(f"Exemple : {X_val[0]}")
print(f"Prédiction : {predictions[0]}")  
print(f"Confiance : {max(probabilities[0]):.2%}")
```

#### **Avantages/Inconvénients**
```python
avantages_nb = [
    "⚡ Très rapide à entraîner",
    "📊 Fonctionne bien avec peu de données", 
    "🎯 Excellent baseline pour commencer",
    "💡 Interprétable (on voit quels mots influencent)"
]

inconvenients_nb = [
    "🤔 Assume l'indépendance des mots (faux)",
    "📝 Ignore l'ordre des mots",
    "🎭 Mal avec sarcasme/ironie complexe"
]
```

### 🎯 **SVM - Les Frontières Optimales**

#### **L'Idée Géniale**
SVM cherche la "frontière" parfaite qui sépare les classes avec la plus grande marge possible.

```python
# Visualisation 2D (simplifié)
"""
    😊 Positifs        |        😡 Négatifs
         😊            |            😡
    😊       😊        |        😡     😡  
         😊            |            😡
                   FRONTIÈRE
                   (hyperplan)
"""
```

#### **Avantage Secret : Le Kernel Trick**
```python
from sklearn.svm import SVC

# SVM linéaire : frontière droite
svm_linear = SVC(kernel='linear', C=1.0)

# SVM polynomial : frontière courbe  
svm_poly = SVC(kernel='poly', degree=3, C=1.0)

# SVM RBF : frontière très flexible
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)
```

#### **Implémentation Complète**
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 1. Recherche des meilleurs hyperparamètres
param_grid = {
    'C': [0.1, 1, 10, 100],           # Régularisation
    'kernel': ['linear', 'rbf'],       # Type de kernel
    'gamma': ['scale', 'auto']         # Pour kernel RBF
}

svm_grid = GridSearchCV(
    SVC(probability=True),  # probability=True pour predict_proba
    param_grid, 
    cv=5,                   # Validation croisée 5-fold
    scoring='f1_weighted',  # Métrique d'optimisation
    n_jobs=-1              # Parallélisation
)

# 2. Entraînement avec recherche d'hyperparamètres
svm_grid.fit(X_train_vec, y_train)

# 3. Meilleur modèle
best_svm = svm_grid.best_estimator_
print(f"Meilleurs paramètres : {svm_grid.best_params_}")
print(f"Score CV : {svm_grid.best_score_:.3f}")

# 4. Évaluation finale
val_score = best_svm.score(X_val_vec, y_val)
print(f"Score validation : {val_score:.3f}")
```

### 🌳 **Random Forest - La Puissance Collective**

#### **Le Principe de la Sagesse des Foules**
```python
# Au lieu d'un seul arbre de décision :
decision_tree_1 = "Je pense que c'est positif"
decision_tree_2 = "Je pense que c'est négatif"  
decision_tree_3 = "Je pense que c'est positif"
# ... 100 arbres différents

# Random Forest fait voter :
vote_final = "Majorité dit positif → POSITIF !"
```

#### **Pourquoi c'est Magique ?**
- Chaque arbre voit des données légèrement différentes
- Chaque arbre utilise des features légèrement différentes  
- Les erreurs individuelles se compensent
- Résultat : plus stable et robuste

#### **Implémentation Avancée**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# 1. Grid search randomisé (plus efficace pour RF)
param_dist = {
    'n_estimators': [100, 200, 300, 500],        # Nombre d'arbres
    'max_depth': [10, 20, 30, None],             # Profondeur max
    'min_samples_split': [2, 5, 10],             # Split minimum  
    'min_samples_leaf': [1, 2, 4],               # Feuilles minimum
    'max_features': ['sqrt', 'log2', None],      # Features par split
    'bootstrap': [True, False]                   # Échantillonnage
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,              # 50 combinaisons testées
    cv=3,                   # 3-fold CV (plus rapide)
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

# 2. Entraînement
rf_random.fit(X_train_vec, y_train)
best_rf = rf_random.best_estimator_

# 3. Analyse de l'importance des features
feature_names = vectorizer.get_feature_names_out()
importances = best_rf.feature_importances_

# Top 10 mots les plus importants
top_features = sorted(
    zip(feature_names, importances), 
    key=lambda x: x[1], 
    reverse=True
)[:10]

print("🏆 Top 10 mots les plus discriminants :")
for mot, importance in top_features:
    print(f"{mot}: {importance:.4f}")
```

### 📊 **Comparaison des Algorithmes**

```python
comparaison_algos = {
    "Critère": ["Vitesse", "Précision", "Interprétabilité", "Peu de données", "Robustesse"],
    "Naive Bayes": ["🟢 Très rapide", "🟡 Correct", "🟢 Excellente", "🟢 Très bon", "🟡 Moyen"],
    "SVM": ["🟡 Moyen", "🟢 Excellent", "🔴 Faible", "🟡 Moyen", "🟢 Très bon"],
    "Random Forest": ["🔴 Lent", "🟢 Excellent", "🟡 Bonne", "🔴 Besoin de beaucoup", "🟢 Excellent"]
}
```

**🎯 Conseil du Pro :**
```python
strategie_choix = {
    "Prototype rapide": "Naive Bayes",
    "Performance maximale": "SVM + Grid Search", 
    "Équilibre perf/interprétabilité": "Random Forest",
    "Production avec gros volume": "Naive Bayes ou RF optimisé"
}
```

---

## 🔧 **4. Feature Engineering - L'Art du Détective**

### 🎯 **Objectif**
Créer des features qui capturent les nuances émotionnelles que les mots seuls ne révèlent pas.

### 📊 **Features Linguistiques Basiques**

#### **Statistiques de Base**
```python
def extraire_features_basiques(texte):
    """Features simples mais puissantes"""
    
    features = {}
    
    # Longueur et structure
    features['nb_mots'] = len(texte.split())
    features['nb_caracteres'] = len(texte)
    features['mots_par_phrase'] = features['nb_mots'] / max(1, texte.count('.'))
    
    # Ponctuation émotionnelle  
    features['nb_exclamations'] = texte.count('!')
    features['nb_questions'] = texte.count('?')
    features['ratio_majuscules'] = sum(1 for c in texte if c.isupper()) / max(1, len(texte))
    
    # Intensité
    features['mots_intensifiants'] = sum(1 for mot in ['très', 'super', 'extrêmement'] 
                                       if mot in texte.lower())
    
    return features

# Test
exemple = "Ce produit est VRAIMENT très décevant !!! Pourquoi ?"
print(extraire_features_basiques(exemple))
```

#### **Features Émotionnelles Avancées**

```python
# Lexiques de sentiments (à charger depuis fichiers)
MOTS_POSITIFS = {'génial', 'fantastique', 'parfait', 'excellent', 'satisfait'}
MOTS_NEGATIFS = {'nul', 'horrible', 'décevant', 'mauvais', 'insatisfait'}
MOTS_INTENSITE = {'très', 'super', 'extrêmement', 'vraiment', 'totalement'}

def extraire_features_sentiment(texte):
    """Features spécialisées sentiment"""
    
    mots = texte.lower().split()
    features = {}
    
    # Comptage direct
    features['mots_positifs'] = sum(1 for mot in mots if mot in MOTS_POSITIFS)
    features['mots_negatifs'] = sum(1 for mot in mots if mot in MOTS_NEGATIFS)
    features['mots_intensite'] = sum(1 for mot in mots if mot in MOTS_INTENSITE)
    
    # Ratios  
    total_mots = len(mots)
    features['ratio_positif'] = features['mots_positifs'] / max(1, total_mots)
    features['ratio_negatif'] = features['mots_negatifs'] / max(1, total_mots)
    
    # Score global
    features['score_brut'] = features['mots_positifs'] - features['mots_negatifs']
    
    return features
```

### 🔤 **N-grammes - Capturer le Contexte**

#### **Pourquoi les N-grammes ?**
```python
phrase = "Ce produit n'est pas terrible"

# Unigrams (1-gramme) : mots isolés
unigrams = ["ce", "produit", "n'est", "pas", "terrible"]
# Problème : "terrible" semble négatif, mais ici c'est positif !

# Bigrams (2-grammes) : paires de mots  
bigrams = ["ce produit", "produit n'est", "n'est pas", "pas terrible"]
# Mieux : "pas terrible" capture la négation !

# Trigrams (3-grammes) : triplets
trigrams = ["ce produit n'est", "produit n'est pas", "n'est pas terrible"] 
# Parfait : "n'est pas terrible" = contexte complet !
```

#### **Implémentation Optimisée**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration multi-niveau
vectorizer_ngrams = TfidfVectorizer(
    ngram_range=(1, 3),          # Unigrams + Bigrams + Trigrams
    max_features=20000,          # Garde les 20k plus importantes
    min_df=2,                    # Ignore mots qui apparaissent < 2 fois
    max_df=0.8,                  # Ignore mots qui apparaissent > 80% docs
    stop_words='french',         # Stop words français
    sublinear_tf=True           # log(tf) au lieu de tf (meilleure perf)
)

# Test de l'impact des n-grammes
def tester_ngrams():
    configurations = [
        (1, 1),    # Unigrams seulement
        (1, 2),    # Unigrams + Bigrams  
        (1, 3),    # Unigrams + Bigrams + Trigrams
        (2, 3)     # Bigrams + Trigrams seulement
    ]
    
    for ngram_range in configurations:
        vec = TfidfVectorizer(ngram_range=ngram_range, max_features=5000)
        X_transformed = vec.fit_transform(X_train)
        
        # Test rapide avec Naive Bayes
        nb = MultinomialNB()
        scores = cross_val_score(nb, X_transformed, y_train, cv=3)
        
        print(f"N-grams {ngram_range}: {scores.mean():.3f} ± {scores.std():.3f}")
```

### ⚡ **Gestion des Négations - Le Piège Fatal**

#### **Le Problème Classique**
```python
# Ces deux phrases ont des mots similaires mais sens opposés !
phrase1 = "Ce produit est bon"          # POSITIF
phrase2 = "Ce produit n'est pas bon"    # NÉGATIF

# Mais pour un modèle naïf : mots = ["ce", "produit", "est", "bon"]
# Il va prédire POSITIF pour les deux ! 😱
```

#### **Solution : Transformation des Négations**
```python
import re

def transformer_negations(texte):
    """
    Transforme les négations pour préserver le sens
    """
    # Patterns de négation français
    patterns_negation = [
        (r"n'est pas (\w+)", r"ne_pas_\1"),          # n'est pas bon → ne_pas_bon
        (r"ne (\w+) pas", r"ne_pas_\1"),             # ne fonctionne pas → ne_pas_fonctionne
        (r"pas du tout (\w+)", r"pas_du_tout_\1"),   # pas du tout satisfait → pas_du_tout_satisfait
        (r"jamais (\w+)", r"jamais_\1"),             # jamais content → jamais_content
        (r"aucun (\w+)", r"aucun_\1"),               # aucun intérêt → aucun_intérêt
    ]
    
    for pattern, replacement in patterns_negation:
        texte = re.sub(pattern, replacement, texte, flags=re.IGNORECASE)
    
    return texte

# Test
exemples = [
    "Ce produit n'est pas terrible du tout",
    "Je ne recommande pas cet achat", 
    "Jamais satisfait de ce service",
    "Aucun problème avec cette commande"
]

for ex in exemples:
    print(f"Avant : {ex}")
    print(f"Après : {transformer_negations(ex)}")
    print()
```

### 🎭 **Features Émotionnelles Avancées**

#### **Analyse des Émojis**
```python
import emoji

def analyser_emojis(texte):
    """Extrait et analyse les émojis"""
    
    # Dictionnaire sentiment des émojis populaires
    emoji_sentiments = {
        '😊': 1, '😃': 1, '😄': 1, '🙂': 1, '😍': 2, '🥰': 2, '🤩': 2,
        '😞': -1, '😢': -2, '😭': -2, '😡': -2, '🤬': -2, '😠': -2,
        '🤔': 0, '😐': 0, '😑': 0
    }
    
    # Extraction des émojis
    emojis_trouves = [c for c in texte if c in emoji.UNICODE_EMOJI['en']]
    
    features = {
        'nb_emojis': len(emojis_trouves),
        'score_emoji': sum(emoji_sentiments.get(em, 0) for em in emojis_trouves),
        'ratio_emojis_positifs': sum(1 for em in emojis_trouves 
                                   if emoji_sentiments.get(em, 0) > 0) / max(1, len(emojis_trouves))
    }
    
    return features

# Test
exemple_emoji = "J'adore ce produit ! 😍🤩 Vraiment top 😊"
print(analyser_emojis(exemple_emoji))
```

#### **Pipeline Complète de Feature Engineering**
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import pandas as pd

class ExtracteurFeaturesCustom(BaseEstimator, TransformerMixin):
    """Extracteur de features personnalisées pour analyse de sentiments"""
    
    def __init__(self):
        self.feature_names = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transforme une liste de textes en matrice de features"""
        
        features_list = []
        
        for texte in X:
            # Préprocessing
            texte_clean = transformer_negations(texte)
            
            # Extraction de toutes les features
            features = {}
            features.update(extraire_features_basiques(texte))
            features.update(extraire_features_sentiment(texte_clean))
            features.update(analyser_emojis(texte))
            
            features_list.append(features)
        
        # Conversion en DataFrame puis array numpy
        df_features = pd.DataFrame(features_list).fillna(0)
        self.feature_names = df_features.columns.tolist()
        
        return df_features.values
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names

# Utilisation dans un pipeline complet
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline_complet = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ('custom', ExtracteurFeaturesCustom())
    ])),
    ('classifier', MultinomialNB())
])

# Entraînement et test
pipeline_complet.fit(X_train, y_train)
score = pipeline_complet.score(X_val, y_val)
print(f"Score avec features custom: {score:.3f}")
```

---

## 📊 **5. Évaluation - Mesurer la Performance**

### 🎯 **Objectif**
Comprendre si votre modèle est vraiment bon ou s'il vous fait illusion.

### 🎲 **Métriques Essentielles**

#### **Accuracy - La Métrique Piège**
```python
# Accuracy = (Prédictions correctes) / (Total prédictions)

# ⚠️ PIÈGE CLASSIQUE
donnees_desequilibrees = {
    "positif": 950,    # 95%
    "négatif": 50      # 5%  
}

# Un modèle stupide qui dit toujours "positif" :
# Accuracy = 950/1000 = 95% ← IMPRESSIONNANT mais INUTILE !
```

#### **Precision, Recall, F1-Score - Les Vraies Métriques**

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluer_modele_complet(y_true, y_pred, labels=None):
    """Évaluation complète d'un modèle de classification"""
    
    # 1. Rapport de classification détaillé
    print("📊 RAPPORT DE CLASSIFICATION")
    print("=" * 50)
    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)
    
    # 2. Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Étiquettes')
    plt.show()
    
    # 3. Analyse détaillée par classe
    print("\n🔍 ANALYSE PAR CLASSE")
    print("=" * 30)
    
    for i, label in enumerate(labels):
        tp = cm[i, i]  # Vrais positifs
        fp = cm[:, i].sum() - tp  # Faux positifs  
        fn = cm[i, :].sum() - tp  # Faux négatifs
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label.upper()}:")
        print(f"  Precision: {precision:.3f} (Sur 100 prédictions '{label}', {precision*100:.1f}% sont correctes)")
        print(f"  Recall: {recall:.3f} (Sur 100 vrais '{label}', {recall*100:.1f}% sont détectés)")
        print(f"  F1-Score: {f1:.3f} (Moyenne harmonique des deux)")
        print()

# Utilisation
predictions = modele.predict(X_test)
evaluer_modele_complet(y_test, predictions, ['négatif', 'neutre', 'positif'])
```

#### **Validation Croisée - Le Test de Robustesse**

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

def validation_croisee_complete(modele, X, y, cv=5):
    """Validation croisée avec analyse statistique"""
    
    # Configuration du cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Scores multiples
    metriques = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    resultats = {}
    
    for metrique in metriques:
        scores = cross_val_score(modele, X, y, cv=skf, scoring=metrique)
        resultats[metrique] = {
            'scores': scores,
            'moyenne': scores.mean(),
            'std': scores.std(),
            'intervalle': (scores.mean() - 2*scores.std(), scores.mean() + 2*scores.std())
        }
    
    # Affichage des résultats
    print("🎯 VALIDATION CROISÉE (5-FOLD)")
    print("=" * 40)
    
    for metrique, stats in resultats.items():
        print(f"{metrique.upper()}:")
        print(f"  Moyenne: {stats['moyenne']:.3f} ± {stats['std']:.3f}")
        print(f"  Intervalle confiance 95%: [{stats['intervalle'][0]:.3f}, {stats['intervalle'][1]:.3f}]")
        print(f"  Scores individuels: {[f'{s:.3f}' for s in stats['scores']]}")
        print()
    
    return resultats

# Test de plusieurs modèles
modeles_test = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='rbf', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

for nom, modele in modeles_test.items():
    print(f"\n🤖 MODÈLE: {nom}")
    validation_croisee_complete(modele, X_train_vec, y_train)
```

### 📈 **Courbes d'Apprentissage**

```python
from sklearn.model_selection import learning_curve
import numpy as np

def tracer_courbes_apprentissage(modele, X, y, title="Courbes d'Apprentissage"):
    """Trace les courbes d'apprentissage pour détecter overfitting/underfitting"""
    
    train_sizes, train_scores, val_scores = learning_curve(
        modele, X, y, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1
    )
    
    # Calcul des moyennes et écarts-types
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Graphique
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Score Entraînement')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Score Validation')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.3, color='red')
    
    plt.xlabel('Taille du dataset d\'entraînement')
    plt.ylabel('Score F1')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Interprétation automatique
    gap_final = train_mean[-1] - val_mean[-1]
    
    print(f"📊 ANALYSE DES COURBES:")
    print(f"Score final entraînement: {train_mean[-1]:.3f}")
    print(f"Score final validation: {val_mean[-1]:.3f}")
    print(f"Gap train/val: {gap_final:.3f}")
    
    if gap_final > 0.1:
        print("⚠️  OVERFITTING détecté ! Le modèle apprend par cœur.")
        print("💡 Solutions: régularisation, plus de données, early stopping")
    elif val_mean[-1] < 0.7:
        print("⚠️  UNDERFITTING détecté ! Le modèle est trop simple.")
        print("💡 Solutions: features plus complexes, modèle plus puissant")
    else:
        print("✅ Modèle bien équilibré !")
    
    plt.show()

# Test sur nos modèles
for nom, modele in modeles_test.items():
    tracer_courbes_apprentissage(modele, X_train_vec, y_train, f"Courbes - {nom}")
```

### 🎯 **Détection des Erreurs Typiques**

```python
def analyser_erreurs(modele, X_test, y_test, X_test_texte):
    """Analyse détaillée des erreurs pour amélioration"""
    
    predictions = modele.predict(X_test)
    probas = modele.predict_proba(X_test)
    
    # 1. Erreurs avec faible confiance
    erreurs_faible_confiance = []
    
    for i, (vraie, pred) in enumerate(zip(y_test, predictions)):
        if vraie != pred:
            confiance = max(probas[i])
            erreurs_faible_confiance.append({
                'index': i,
                'texte': X_test_texte[i],
                'vraie_classe': vraie,
                'pred_classe': pred, 
                'confiance': confiance
            })
    
    # Tri par confiance croissante
    erreurs_faible_confiance.sort(key=lambda x: x['confiance'])
    
    print("🔍 TOP 10 ERREURS AVEC FAIBLE CONFIANCE")
    print("=" * 50)
    
    for i, erreur in enumerate(erreurs_faible_confiance[:10]):
        print(f"\n{i+1}. Confiance: {erreur['confiance']:.2%}")
        print(f"   Texte: {erreur['texte'][:100]}...")
        print(f"   Vraie classe: {erreur['vraie_classe']}")
        print(f"   Prédiction: {erreur['pred_classe']}")
    
    # 2. Analyse des confusions fréquentes
    cm = confusion_matrix(y_test, predictions)
    classes = ['négatif', 'neutre', 'positif']
    
    print(f"\n❌ CONFUSIONS LES PLUS FRÉQUENTES")
    print("=" * 35)
    
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                confusions.append((classes[i], classes[j], cm[i][j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    for vraie, pred, nb in confusions[:5]:
        print(f"{vraie} → {pred}: {nb} erreurs")
    
    return erreurs_faible_confiance

# Analyse des erreurs
erreurs = analyser_erreurs(best_modele, X_test_vec, y_test, X_test)
```

---

## 💻 **Exercices Pratiques**

### 📝 **Exercice 9 : Classification Naive Bayes**
**Objectif :** Maîtriser l'algorithme de base de la classification de texte  
**Difficulté :** ⭐⭐⭐☆☆  
**Temps estimé :** 45 minutes

#### **Énoncé**
Créez un classificateur de sentiments binaire (positif/négatif) en utilisant Naive Bayes sur un dataset d'avis clients.

#### **Dataset Fourni**
- **avis_clients_binaire.csv** : 2000 avis (1000 positifs, 1000 négatifs)
- Colonnes : `texte`, `sentiment`

#### **Tâches à Réaliser**
1. **Exploration** : Analysez la distribution et la longueur des textes
2. **Preprocessing** : Nettoyez les données avec la pipeline du module
3. **Vectorisation** : Testez différentes configurations TF-IDF
4. **Entraînement** : Naive Bayes avec optimisation des hyperparamètres
5. **Évaluation** : Métriques complètes + analyse d'erreurs

#### **Code Template**
```python
# Votre code ici - template fourni dans /exercices/exercice-09/
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Chargement des données
df = pd.read_csv('datasets/avis_clients_binaire.csv')

# 2. TODO: Exploration des données
# Votre code ici

# 3. TODO: Preprocessing
# Votre code ici

# 4. TODO: Split et vectorisation  
# Votre code ici

# 5. TODO: Entraînement Naive Bayes
# Votre code ici

# 6. TODO: Évaluation complète
# Votre code ici
```

#### **Critères de Réussite**
- [ ] F1-Score > 0.85 sur le test set
- [ ] Analyse d'au moins 3 configurations TF-IDF différentes
- [ ] Identification des 10 mots les plus discriminants
- [ ] Code propre et commenté

---

### 📝 **Exercice 10 : Feature Engineering pour Sentiments**
**Objectif :** Créer des features custom qui améliorent la performance  
**Difficulté :** ⭐⭐⭐⭐☆  
**Temps estimé :** 60 minutes

#### **Énoncé**
Développez un système de features personnalisées qui capture les nuances émotionnelles et testez leur impact sur la performance.

#### **Dataset**
- **tweets_sentiments.csv** : 5000 tweets annotés (positif/négatif/neutre)
- Défi : Textes courts avec emojis, argot, négations

#### **Features à Implémenter**
1. **Émotionnelles** : Score des mots positifs/négatifs, ratio d'émojis
2. **Linguistiques** : Longueur, ponctuation, majuscules  
3. **Négations** : Transformation des constructions négatives
4. **N-grammes** : Bigrammes et trigrammes contextuels
5. **Custom** : Une feature innovante de votre invention !

#### **Analyse Requise**
```python
# Comparaison obligatoire
configurations = [
    "TF-IDF seul",
    "TF-IDF + Features basiques", 
    "TF-IDF + Features émotionnelles",
    "TF-IDF + Toutes features custom",
    "Features custom seulement"
]

# Pour chaque config : F1-score + temps d'entraînement
```

#### **Bonus Challenge**
Créez une feature qui détecte le sarcasme/ironie (indices : guillemets, patterns linguistiques)

#### **Critères de Réussite**
- [ ] Amélioration de +5% minimum avec features custom
- [ ] Pipeline réutilisable et modulaire
- [ ] Analyse de l'importance des features
- [ ] Documentation des choix de design

---

### 📝 **Exercice 11 : Validation Croisée et Optimisation**
**Objectif :** Maîtriser l'évaluation rigoureuse et l'optimisation d'hyperparamètres  
**Difficulté :** ⭐⭐⭐⭐☆  
**Temps estimé :** 45 minutes

#### **Énoncé**
Comparez scientifiquement 3 algorithmes de classification et optimisez le meilleur avec Grid Search et validation croisée.

#### **Algorithmes à Tester**
1. **Naive Bayes** : MultinomialNB avec lissage
2. **SVM** : Kernel RBF avec régularisation
3. **Random Forest** : Ensemble avec profondeur variable

#### **Protocole d'Évaluation**
```python
# Validation croisée stratifiée 5-fold
# Métriques : Accuracy, Precision, Recall, F1 (weighted)
# Analyse statistique : moyenne ± écart-type
# Test de significativité entre modèles
```

#### **Grid Search Requis**
```python
param_grids = {
    'nb': {'alpha': [0.1, 0.5, 1.0, 2.0]},
    'svm': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'rf': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
}
```

#### **Livrables**
- Tableau comparatif avec intervalles de confiance
- Courbes d'apprentissage pour le meilleur modèle
- Analyse du temps de calcul vs performance
- Recommandation justifiée pour la production

#### **Critères de Réussite**
- [ ] Validation croisée correctement implémentée
- [ ] Grid search exhaustif sur les 3 modèles
- [ ] Analyse statistique des différences
- [ ] Visualisations professionnelles

---

### 📝 **Exercice 12 : Optimisation Avancée et Diagnostic**
**Objectif :** Diagnostiquer et corriger les problèmes de performance  
**Difficulté :** ⭐⭐⭐⭐⭐  
**Temps estimé :** 75 minutes

#### **Énoncé**
Vous recevez un modèle "cassé" avec de mauvaises performances. Diagnostiquez les problèmes et proposez des solutions.

#### **Scénario**
```python
# Modèle fourni avec problèmes volontaires :
modele_casse = {
    "f1_score": 0.45,  # Très faible !
    "train_accuracy": 0.95,  # Suspect...
    "val_accuracy": 0.50,   # Overfitting évident
    "problemes": ["données déséquilibrées", "features inadaptées", "hyperparamètres sous-optimaux"]
}
```

#### **Mission de Diagnostic**
1. **Analyse des Données**
   - Distribution des classes
   - Qualité des annotations
   - Présence de doublons/bruit

2. **Diagnostic du Modèle**
   - Courbes d'apprentissage
   - Matrice de confusion détaillée
   - Analyse des erreurs par classe

3. **Solutions d'Amélioration**
   - Rééquilibrage des données
   - Feature engineering ciblé
   - Régularisation appropriée
   - Ensemble methods

#### **Optimisations à Tester**
```python
strategies_amelioration = [
    "SMOTE pour rééquilibrage",
    "Feature selection avec chi2",
    "Regularisation L1/L2",
    "Ensemble Voting/Bagging",
    "Threshold tuning",
    "Calibration des probabilités"
]
```

#### **Rapport Final Requis**
- Diagnostic des problèmes identifiés
- Impact quantifié de chaque amélioration
- Recommandations pour éviter ces problèmes
- Code optimisé final avec documentation

#### **Critères de Réussite**
- [ ] Amélioration du F1-score à >0.80
- [ ] Élimination de l'overfitting (gap <0.05)
- [ ] Documentation complète du processus
- [ ] Propositions innovantes d'amélioration

---

## 🚀 **Projet Final : Analyseur de Sentiments Multi-Classes**

### 🎯 **Objectif Global**
Créer un analyseur de sentiments professionnel capable de traiter du texte en temps réel et de fournir des insights business.

### 📋 **Cahier des Charges**

#### **Fonctionnalités Obligatoires**
1. **Classification Multi-Classes** : Positif/Négatif/Neutre avec scores de confiance
2. **API REST** : Endpoint pour analyse en temps réel
3. **Interface Web** : Upload de fichiers CSV + analyse en batch
4. **Visualisations** : Graphiques de distribution des sentiments
5. **Export** : Résultats en JSON/CSV avec métriques détaillées

#### **Spécifications Techniques**
```python
# Architecture imposée
projet_structure = {
    "modele/": "Pipeline d'entraînement + modèle sauvegardé",
    "api/": "FastAPI avec endpoints documentés",
    "interface/": "Streamlit pour demo interactive", 
    "tests/": "Tests unitaires + tests d'intégration",
    "data/": "Datasets + preprocessing pipeline",
    "docs/": "Documentation technique complète"
}
```

#### **Performance Minimale Requise**
- **F1-Score** : >0.75 sur test set multi-classes
- **Latence API** : <200ms par requête
- **Robustesse** : Gestion des erreurs et cas limites
- **Scalabilité** : Traitement de 1000+ textes en batch

### 🛠️ **Template de Démarrage**

#### **Structure du Projet**
```
analyseur-sentiments/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── config.yaml
│   └── logging.conf
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── datasets.py
│   ├── models/
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── prediction.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── interface/
│       ├── app.py
│       └── components.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
├── notebooks/
│   ├── exploration.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
└── models/
    ├── vectorizer.pkl
    ├── classifier.pkl
    └── metadata.json
```

#### **API Template (FastAPI)**
```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI(title="Analyseur de Sentiments", version="1.0.0")

# Modèles chargés au démarrage
vectorizer = joblib.load("models/vectorizer.pkl")
classifier = joblib.load("models/classifier.pkl")

class TexteInput(BaseModel):
    texte: str

class SentimentOutput(BaseModel):
    sentiment: str
    confiance: float
    scores: dict

@app.post("/analyser", response_model=SentimentOutput)
async def analyser_sentiment(input_data: TexteInput):
    """Analyse le sentiment d'un texte"""
    try:
        # TODO: Implémenter la logique d'analyse
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyser_batch", response_model=List[SentimentOutput])
async def analyser_batch(textes: List[TexteInput]):
    """Analyse multiple textes en une fois"""
    # TODO: Implémenter l'analyse en lot
    pass
```

#### **Interface Streamlit Template**
```python
# src/interface/app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.title("🎯 Analyseur de Sentiments Pro")

# Sidebar pour configuration
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("URL API", "http://localhost:8000")

# Onglets pour différents modes
tab1, tab2, tab3 = st.tabs(["📝 Texte Simple", "📊 Analyse Batch", "📈 Dashboard"])

with tab1:
    st.header("Analyse d'un texte")
    texte_input = st.text_area("Entrez votre texte à analyser:")
    
    if st.button("Analyser"):
        if texte_input:
            # TODO: Appel API + affichage résultats
            pass

with tab2:
    st.header("Analyse en lot")
    fichier = st.file_uploader("Upload CSV", type=['csv'])
    
    if fichier:
        # TODO: Traitement batch + visualisations
        pass

with tab3:
    st.header("Dashboard Analytics")
    # TODO: Métriques globales + graphiques
    pass
```

### 📊 **Critères d'Évaluation**

| Critère | Poids | Détail |
|---------|-------|--------|
| **Performance Technique** | 30% | F1-score, robustesse, optimisation |
| **Architecture** | 25% | Code modulaire, tests, documentation |
| **Interface Utilisateur** | 20% | UX/UI, facilité d'usage |
| **Innovation** | 15% | Features originales, amélirations |
| **Documentation** | 10% | README, API docs, guide utilisateur |

### 🏆 **Bonus Points**

- **🚀 Déploiement** : Application déployée sur Heroku/Streamlit Cloud (+5 pts)
- **🧪 A/B Testing** : Comparaison de modèles avec métriques (+3 pts)
- **🎨 Design** : Interface particulièrement soignée (+2 pts)
- **⚡ Performance** : Optimisations avancées (caching, async) (+3 pts)
- **📱 Mobile** : Interface responsive/mobile-friendly (+2 pts)

---

## 📚 **Ressources et Outils**

### 📦 **Packages Essentiels**
```python
# requirements.txt du module
scikit-learn==1.3.0      # Algorithmes ML
pandas==2.0.3            # Manipulation données  
numpy==1.24.3            # Calculs numériques
matplotlib==3.7.1        # Visualisations de base
seaborn==0.12.2          # Visualisations avancées
plotly==5.15.0           # Graphiques interactifs

# NLP spécialisé
nltk==3.8.1              # Outils linguistiques
spacy==3.6.1             # NLP moderne
textblob==0.17.1         # Sentiment analysis simple

# API et interface
fastapi==0.100.1         # API REST moderne
streamlit==1.25.0        # Interface web rapide
uvicorn==0.23.2          # Serveur ASGI

# Utilitaires
joblib==1.3.1            # Sérialisation modèles
tqdm==4.65.0             # Barres de progression
jupyter==1.0.0           # Notebooks
```

### 📊 **Datasets Fournis**

#### **Dataset Principal : Avis Multi-Domaines**
```
avis_multidomaines.csv (10,000 entrées)
├── colonnes: texte, sentiment, domaine, longueur
├── sentiments: positif (40%), neutre (20%), négatif (40%)  
├── domaines: e-commerce, restaurants, films, tech, voyage
└── challenge: variabilité de vocabulaire entre domaines
```

#### **Dataset Challenge : Tweets en Temps Réel**
```
tweets_realtime.csv (5,000 entrées)
├── défis: emojis, argot, fautes, sarcasme
├── annotation: crowd-sourcing avec accord inter-annotateurs
└── métadonnées: timestamp, nb_retweets, nb_likes
```

#### **Dataset Validation : Avis Produits Amazon**
```
amazon_reviews_fr.csv (3,000 entrées)
├── structure: review_text, rating (1-5 étoiles)
├── conversion: 1-2★