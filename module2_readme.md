# 🔢 Module 2 : Vectorisation - Transformer les Mots en Nombres

> **"Comment expliquer à un ordinateur que 'roi' et 'reine' sont similaires ?"**
> 
> Bienvenue dans l'univers fascinant de la vectorisation ! Ici, nous transformons des mots en coordonnées mathématiques pour que les machines comprennent enfin le **sens** derrière les mots.

---

## 📹 **Vidéo d'Introduction** *(5 minutes)*

🎬 **[Lien vers la vidéo : "De Shakespeare aux Mathématiques"]**

**Résumé de la vidéo :**
> Dans cette vidéo, vous découvrirez pourquoi un ordinateur ne peut pas naturellement comprendre que "excellent" et "fantastique" veulent dire la même chose. Je vous montrerai comment les techniques de vectorisation permettent de créer un "GPS du langage" où chaque mot a ses coordonnées dans un espace de sens. Vous verrez des démonstrations live de word embeddings qui résolvent des analogies comme "roi - homme + femme = reine" et comprendrez pourquoi Google Translate s'améliore sans cesse. À la fin de ce module, vous aurez créé votre propre détecteur de plagiat plus malin que celui de votre université !

**Points clés abordés :**
- ✨ Démonstration visuelle : mots dans l'espace 3D
- 🧮 Du comptage simple aux embeddings sophistiqués
- 🎯 Applications concrètes : détection de similarité, recommendation
- 🚀 Teasing du projet final : détecteur de plagiat

---

## 🎯 **Objectifs d'Apprentissage**

À la fin de ce module, vous serez capable de :

- [ ] **Expliquer** pourquoi les machines ont besoin de transformer les mots en nombres
- [ ] **Implémenter** les techniques Bag of Words et TF-IDF from scratch
- [ ] **Utiliser** des word embeddings pré-entraînés pour des tâches pratiques
- [ ] **Calculer** la similarité entre textes avec différentes méthodes
- [ ] **Créer** un système de détection de similarité fonctionnel
- [ ] **Analyser** les avantages/inconvénients de chaque approche

---

## 📚 **Plan du Module**

| Section | Contenu | Durée | Difficulté |
|---------|---------|-------|------------|
| **1** | Le Problème de Représentation | 45 min | ⭐⭐☆☆☆ |
| **2** | Bag of Words (BoW) | 60 min | ⭐⭐☆☆☆ |
| **3** | TF-IDF | 75 min | ⭐⭐⭐☆☆ |
| **4** | Word Embeddings | 90 min | ⭐⭐⭐⭐☆ |
| **5** | Calcul de Similarité | 60 min | ⭐⭐⭐☆☆ |
| **Exercices** | 4 exercices pratiques | 120 min | Variable |
| **Projet** | Détecteur de similarité | 90 min | ⭐⭐⭐⭐☆ |

**Total estimé :** 4-5 heures

---

## 📖 **Section 1 : Le Problème de Représentation**

### 🤔 **Pourquoi les Mots ne Sont pas des Nombres ?**

Imaginez que vous essayez d'expliquer à un alien (votre ordinateur) les relations entre les mots humains. Comment lui faire comprendre que :

- "Chat" et "Félin" sont proches
- "Roi" et "Reine" partagent un concept de royauté
- "Courir" et "Sprint" sont des variations d'intensité

**Le défi fondamental :** Les ordinateurs ne manipulent que des nombres (0 et 1), mais le langage humain est fait de symboles abstraits chargés de sens.

### 🗺️ **L'Espace Vectoriel du Langage**

**Analogie GPS :** Tout comme chaque lieu sur Terre a des coordonnées (latitude, longitude), nous pouvons donner des "coordonnées de sens" à chaque mot.

```python
# Exemple conceptuel
mots_coordonnees = {
    "roi": [0.8, 0.2, 0.9],      # [pouvoir, genre_masculin, noblesse]
    "reine": [0.8, 0.8, 0.9],    # [pouvoir, genre_feminin, noblesse]
    "chat": [0.1, 0.5, 0.2],     # [pouvoir, genre_neutre, noblesse]
}
```

### 🎯 **Applications Concrètes**

**Où utilisez-vous déjà la vectorisation sans le savoir ?**

- 🔍 **Moteurs de recherche** : Google trouve des documents similaires à votre requête
- 🎵 **Spotify** : Recommandations basées sur la similarité des descriptions musicales
- 🛒 **E-commerce** : "Les clients qui ont aimé X ont aussi aimé Y"
- 🌐 **Traduction** : Aligner les concepts entre langues différentes

### 💡 **Les Défis à Relever**

1. **Synonymes** : "Voiture" et "automobile" doivent être proches
2. **Polysémie** : "Avocat" (fruit) vs "avocat" (métier)
3. **Contexte** : "Pomme" dans "pomme de terre" vs "pomme rouge"
4. **Négation** : "Pas bon" ≠ "bon"

---

## 📖 **Section 2 : Bag of Words (BoW) - L'Approche Naive**

### 🎒 **Le Principe du Sac de Mots**

**Métaphore :** Imaginez que vous videz un livre dans un sac et que vous comptez chaque mot, **en ignorant l'ordre**.

```python
# Exemple simple
phrases = [
    "Le chat mange",
    "Le chien mange aussi",
    "Chat et chien sont amis"
]

# Vocabulaire global
vocabulaire = ["le", "chat", "mange", "chien", "aussi", "et", "sont", "amis"]

# Représentation BoW
bow_representations = [
    [1, 1, 1, 0, 0, 0, 0, 0],  # "Le chat mange"
    [1, 0, 1, 1, 1, 0, 0, 0],  # "Le chien mange aussi"
    [0, 1, 0, 1, 0, 1, 1, 1]   # "Chat et chien sont amis"
]
```

### 🛠️ **Implémentation Manuelle**

```python
def create_bow_manual(documents):
    """
    Crée une représentation Bag of Words from scratch
    """
    # Étape 1: Construire le vocabulaire
    vocabulaire = set()
    for doc in documents:
        vocabulaire.update(doc.lower().split())
    
    vocab_list = sorted(list(vocabulaire))
    
    # Étape 2: Vectoriser chaque document
    bow_matrix = []
    for doc in documents:
        words = doc.lower().split()
        vector = [words.count(word) for word in vocab_list]
        bow_matrix.append(vector)
    
    return bow_matrix, vocab_list

# Test
documents = [
    "Python est génial",
    "J'adore programmer en Python",
    "Le machine learning avec Python"
]

bow_matrix, vocabulaire = create_bow_manual(documents)
print("Vocabulaire:", vocabulaire)
print("Matrice BoW:", bow_matrix)
```

### ⚖️ **Avantages et Limitations**

| ✅ **Avantages** | ❌ **Limitations** |
|------------------|-------------------|
| Simple à comprendre | Perte de l'ordre des mots |
| Rapide à calculer | Pas de contexte sémantique |
| Fonctionne sur tous les langages | Problème de dimensionnalité |
| Base solide pour d'autres techniques | Sensible aux mots fréquents |

### 🧪 **Exemple Pratique : Classification de Textes**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Données d'exemple
textes = [
    "Ce film est fantastique",
    "J'ai adoré ce film",
    "Film décevant et ennuyeux",
    "Très mauvais film",
    "Excellent divertissement",
    "Perte de temps totale"
]
labels = ["positif", "positif", "négatif", "négatif", "positif", "négatif"]

# Vectorisation BoW
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textes)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3)
model = MultinomialNB()
model.fit(X_train, y_train)

# Test
nouveau_texte = ["Ce film est incroyable"]
prediction = model.predict(vectorizer.transform(nouveau_texte))
print(f"Sentiment prédit: {prediction[0]}")
```

---

## 📖 **Section 3 : TF-IDF - L'Intelligence du Comptage**

### 🧠 **Le Problème avec BoW Simple**

Considérez ces deux documents :
- Doc 1: "Le chat mange le poisson"
- Doc 2: "Le machine learning transforme le monde"

Avec BoW, "le" a le même poids que "machine learning". Mais "le" n'apporte aucune information discriminante !

### 📊 **TF-IDF : Term Frequency × Inverse Document Frequency**

**Philosophie :** Un mot est important s'il apparaît souvent dans un document (TF) mais rarement dans la collection globale (IDF).

#### **TF (Term Frequency)**
```
TF(terme, document) = Nombre d'occurrences du terme / Nombre total de mots
```

#### **IDF (Inverse Document Frequency)**
```
IDF(terme) = log(Nombre total de documents / Nombre de documents contenant le terme)
```

#### **TF-IDF Final**
```
TF-IDF(terme, document) = TF(terme, document) × IDF(terme)
```

### 🧮 **Calcul Manuel Détaillé**

```python
import math
from collections import Counter

def calculate_tf_idf_manual(documents):
    """
    Calcul TF-IDF from scratch avec explications détaillées
    """
    
    # Préparation des documents
    docs_words = [doc.lower().split() for doc in documents]
    
    # Construction du vocabulaire
    all_words = set()
    for words in docs_words:
        all_words.update(words)
    vocab = sorted(list(all_words))
    
    # Calcul TF pour chaque document
    tf_matrix = []
    for words in docs_words:
        word_count = Counter(words)
        total_words = len(words)
        
        tf_vector = []
        for word in vocab:
            tf = word_count[word] / total_words
            tf_vector.append(tf)
        tf_matrix.append(tf_vector)
    
    # Calcul IDF pour chaque terme
    idf_vector = []
    total_docs = len(documents)
    
    for word in vocab:
        docs_with_word = sum(1 for words in docs_words if word in words)
        idf = math.log(total_docs / docs_with_word)
        idf_vector.append(idf)
    
    # Calcul TF-IDF final
    tfidf_matrix = []
    for tf_vector in tf_matrix:
        tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vector)]
        tfidf_matrix.append(tfidf_vector)
    
    return tfidf_matrix, vocab, tf_matrix, idf_vector

# Exemple d'utilisation
documents = [
    "Le chat mange du poisson",
    "Le chien mange des croquettes",
    "Machine learning et intelligence artificielle",
    "Python pour le machine learning"
]

tfidf_matrix, vocab, tf_matrix, idf_vector = calculate_tf_idf_manual(documents)

# Affichage des résultats
print("Vocabulaire:", vocab)
print("\nScores IDF (mots rares = scores élevés):")
for word, idf in zip(vocab, idf_vector):
    print(f"{word}: {idf:.3f}")
```

### 📈 **Visualisation des Scores TF-IDF**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_tfidf(tfidf_matrix, vocab, documents):
    """
    Visualise les scores TF-IDF sous forme de heatmap
    """
    # Création du DataFrame
    df = pd.DataFrame(tfidf_matrix, 
                     columns=vocab,
                     index=[f"Doc {i+1}" for i in range(len(documents))])
    
    # Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title("Scores TF-IDF par Document et Terme")
    plt.xlabel("Termes")
    plt.ylabel("Documents")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Top mots par document
    for i, (doc, scores) in enumerate(zip(documents, tfidf_matrix)):
        word_scores = list(zip(vocab, scores))
        top_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:3]
        print(f"\nDocument {i+1}: '{doc[:30]}...'")
        print("Mots les plus importants:")
        for word, score in top_words:
            if score > 0:
                print(f"  - {word}: {score:.3f}")
```

### 🚀 **TF-IDF avec sklearn**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Version professionnelle
vectorizer = TfidfVectorizer(
    max_features=1000,        # Limite le vocabulaire
    stop_words='english',     # Supprime les mots vides
    ngram_range=(1, 2),       # Uni + bigrammes
    min_df=2,                 # Ignore les mots très rares
    max_df=0.95              # Ignore les mots très fréquents
)

# Application sur corpus
corpus = [
    "Python est parfait pour le machine learning",
    "Scikit-learn simplifie le machine learning",
    "TensorFlow pour le deep learning avancé",
    "Les réseaux de neurones transforment l'IA"
]

tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

print("Forme de la matrice:", tfidf_matrix.shape)
print("Premiers termes:", feature_names[:10])
```

---

## 📖 **Section 4 : Word Embeddings - La Révolution Sémantique**

### 🌟 **Au-delà du Comptage : Comprendre le Sens**

**Le problème avec TF-IDF :** Il ne sait pas que "voiture" et "automobile" sont synonymes.

**La solution Word Embeddings :** Représenter chaque mot par un vecteur dense qui capture son sens et ses relations avec d'autres mots.

### 🧬 **Le Miracle de Word2Vec**

Word2Vec apprend les représentations en analysant les **contextes** où apparaissent les mots.

**Principe :** "Les mots qui apparaissent dans des contextes similaires ont des sens similaires"

```python
# Exemples de contextes
contexts = [
    "Le [roi] règne sur son royaume",
    "La [reine] gouverne avec sagesse", 
    "Le [chat] dort sur le canapé",
    "Le [chaton] joue dans le jardin"
]

# Word2Vec va apprendre que:
# roi ≈ reine (contexte de pouvoir)
# chat ≈ chaton (contexte animal domestique)
```

### 🎯 **Les Analogies Magiques**

```python
import gensim.downloader as api

# Chargement d'un modèle pré-entraîné français
# Note: En pratique, vous devrez télécharger un modèle français
# model = api.load('word2vec-google-news-300')

# Exemples d'analogies possibles:
# roi - homme + femme ≈ reine
# Paris - France + Italie ≈ Rome
# grand - plus_grand + intelligent ≈ plus_intelligent

def test_analogies(model):
    """
    Teste des analogies avec Word2Vec
    """
    analogies = [
        ("roi", "homme", "femme"),  # → reine
        ("Paris", "France", "Italie"),  # → Rome
        ("grand", "petit", "haut")  # → bas
    ]
    
    for a, b, c in analogies:
        try:
            # Calcul: a - b + c = ?
            result = model.most_similar(positive=[a, c], negative=[b], topn=1)
            print(f"{a} - {b} + {c} = {result[0][0]} (score: {result[0][1]:.3f})")
        except KeyError as e:
            print(f"Mot non trouvé dans le vocabulaire: {e}")
```

### 🛠️ **Utilisation Pratique avec spaCy**

```python
import spacy

# Chargement du modèle français avec embeddings
nlp = spacy.load("fr_core_news_md")  # Modèle moyen avec vecteurs

def explore_word_vectors():
    """
    Exploration des embeddings avec spaCy
    """
    
    # Mots à analyser
    mots = ["chat", "chien", "voiture", "automobile", "heureux", "joyeux"]
    
    # Extraction des vecteurs
    vecteurs = {}
    for mot in mots:
        doc = nlp(mot)
        if doc[0].has_vector:
            vecteurs[mot] = doc[0].vector
            print(f"{mot}: vecteur de dimension {len(doc[0].vector)}")
    
    # Calcul de similarités
    print("\n=== Similarités ===")
    from sklearn.metrics.pairwise import cosine_similarity
    
    for i, mot1 in enumerate(mots):
        for mot2 in mots[i+1:]:
            if mot1 in vecteurs and mot2 in vecteurs:
                sim = cosine_similarity([vecteurs[mot1]], [vecteurs[mot2]])[0][0]
                print(f"{mot1} ↔ {mot2}: {sim:.3f}")

explore_word_vectors()
```

### 🔍 **Clustering Sémantique**

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def semantic_clustering(words, n_clusters=3):
    """
    Regroupe des mots par similarité sémantique
    """
    nlp = spacy.load("fr_core_news_md")
    
    # Extraction des vecteurs
    vectors = []
    valid_words = []
    
    for word in words:
        doc = nlp(word)
        if doc[0].has_vector:
            vectors.append(doc[0].vector)
            valid_words.append(word)
    
    if len(vectors) == 0:
        print("Aucun vecteur trouvé!")
        return
    
    vectors = np.array(vectors)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(vectors)
    
    # Visualisation 2D avec PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i in range(n_clusters):
        cluster_points = vectors_2d[clusters == i]
        cluster_words = [valid_words[j] for j in range(len(valid_words)) if clusters[j] == i]
        
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i], label=f'Cluster {i+1}', s=100)
        
        # Annotations
        for j, word in enumerate(cluster_words):
            plt.annotate(word, 
                        (cluster_points[j, 0], cluster_points[j, 1]),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title("Clustering Sémantique des Mots")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Affichage des clusters
    for i in range(n_clusters):
        cluster_words = [valid_words[j] for j in range(len(valid_words)) if clusters[j] == i]
        print(f"Cluster {i+1}: {', '.join(cluster_words)}")

# Test avec des mots variés
mots_test = [
    "chat", "chien", "animal", "oiseau",
    "voiture", "automobile", "transport", "train",
    "heureux", "joyeux", "triste", "mélancolique",
    "ordinateur", "machine", "robot", "intelligence"
]

semantic_clustering(mots_test, n_clusters=4)
```

---

## 📖 **Section 5 : Calcul de Similarité**

### 📐 **Métriques de Distance et Similarité**

Une fois que nous avons transformé nos textes en vecteurs, comment mesurer leur proximité ?

#### **1. Similarité Cosinus** *(La plus populaire)*

```python
import numpy as np

def cosine_similarity_manual(vec1, vec2):
    """
    Calcule la similarité cosinus entre deux vecteurs
    """
    # Produit scalaire
    dot_product = np.dot(vec1, vec2)
    
    # Normes des vecteurs
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Similarité cosinus
    if norm1 == 0 or norm2 == 0:
        return 0
    
    similarity = dot_product / (norm1 * norm2)
    return similarity

# Exemple pratique
texte1_vec = np.array([1, 2, 0, 1])  # Représentation TF-IDF du texte 1
texte2_vec = np.array([2, 1, 1, 0])  # Représentation TF-IDF du texte 2

similarity = cosine_similarity_manual(texte1_vec, texte2_vec)
print(f"Similarité cosinus: {similarity:.3f}")

# Interprétation:
# 1.0 = Identiques
# 0.0 = Orthogonaux (aucune relation)
# -1.0 = Opposés
```

#### **2. Distance Euclidienne**

```python
def euclidean_distance(vec1, vec2):
    """
    Calcule la distance euclidienne entre deux vecteurs
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# Note: Plus la distance est petite, plus les textes sont similaires
distance = euclidean_distance(texte1_vec, texte2_vec)
print(f"Distance euclidienne: {distance:.3f}")
```

#### **3. Distance de Manhattan**

```python
def manhattan_distance(vec1, vec2):
    """
    Calcule la distance de Manhattan entre deux vecteurs
    """
    return np.sum(np.abs(vec1 - vec2))

distance_manhattan = manhattan_distance(texte1_vec, texte2_vec)
print(f"Distance de Manhattan: {distance_manhattan:.3f}")
```

### 🔬 **Comparaison Pratique des Métriques**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

def compare_similarity_metrics(texts):
    """
    Compare différentes métriques de similarité sur un corpus
    """
    # Vectorisation
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calcul des similarités/distances
    cos_sim = cosine_similarity(tfidf_matrix)
    eucl_dist = euclidean_distances(tfidf_matrix)
    manh_dist = manhattan_distances(tfidf_matrix)
    
    # Affichage comparatif
    n_texts = len(texts)
    
    print("=== COMPARAISON DES MÉTRIQUES ===\n")
    
    for i in range(n_texts):
        for j in range(i+1, n_texts):
            print(f"Texte {i+1} vs Texte {j+1}:")
            print(f"  Cosinus Similarité: {cos_sim[i,j]:.3f}")
            print(f"  Distance Euclidienne: {eucl_dist[i,j]:.3f}")
            print(f"  Distance Manhattan: {manh_dist[i,j]:.3f}")
            print()

# Test avec exemples concrets
textes_test = [
    "Python est un excellent langage de programmation",
    "J'adore programmer en Python, c'est fantastique",
    "Java est utilisé pour le développement d'applications",
    "Le machine learning révolutionne la technologie"
]

compare_similarity_metrics(textes_test)
```

### 🎯 **Applications Pratiques**

#### **Système de Recommandation Simple**

```python
def recommend_similar_articles(query, articles, top_k=3):
    """
    Recommande des articles similaires à une requête
    """
    # Préparation des données
    all_texts = [query] + articles
    
    # Vectorisation
    vectorizer = TfidfVectorizer(stop_words='french')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calcul similarité avec la requête (premier élément)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Tri par similarité décroissante
    similar_indices = similarities.argsort()[::-1][:top_k]
    
    # Retour des résultats
    recommendations = []
    for idx in similar_indices:
        recommendations.append({
            'article': articles[idx],
            'score': similarities[idx],
            'rank': len(recommendations) + 1
        })
    
    return recommendations

# Test du système de recommandation
query = "intelligence artificielle et machine learning"

articles_db = [
    "Les réseaux de neurones transforment l'IA moderne",
    "Recette de cuisine: tarte aux pommes traditionnelle",
    "Deep learning pour la reconnaissance d'images",
    "Guide de voyage: visiter Paris en 3 jours",
    "Algorithmes de classification en machine learning",
    "Histoire de l'art contemporain français"
]

recommendations = recommend_similar_articles(query, articles_db, top_k=3)

print(f"Requête: '{query}'\n")
print("Recommandations:")
for rec in recommendations:
    print(f"{rec['rank']}. {rec['article']}")
    print(f"   Score: {rec['score']:.3f}\n")
```

---

## 🛠️ **Exercices Pratiques**

### 📝 **Exercice 5 : Bag of Words Manuel**
**Objectif :** Implémenter BoW from scratch et comparer avec sklearn
**Difficulté :** ⭐⭐☆☆☆ | **Points :** 7

#### Instructions
1. Créez une fonction `create_bow_manual()` qui transforme une liste de textes en matrice BoW
2. Testez sur 3 phrases de votre choix
3. Comparez les résultats avec `CountVectorizer` de sklearn
4. Analysez les différences et expliquez pourquoi elles existent

#### Code de départ
```python
def create_bow_manual(documents):
    """
    Votre implémentation ici
    Retourne: (matrice_bow, vocabulaire)
    """
    pass

# Tests à effectuer
test_documents = [
    "Python est génial pour programmer",
    "J'adore programmer en Python",
    "Le machine learning avec Python est fascinant"
]
```

#### Critères d'évaluation
- [ ] Fonction correctement implémentée
- [ ] Gestion de la casse et ponctuation
- [ ] Comparaison détaillée avec sklearn
- [ ] Analyse des différences

---

### 📝 **Exercice 6 : TF-IDF from Scratch**
**Objectif :** Comprendre TF-IDF en l'implémentant manuellement
**Difficulté :** ⭐⭐⭐☆☆ | **Points :** 7

#### Instructions
1. Implémentez les fonctions `calculate_tf()`, `calculate_idf()` et `calculate_tfidf()`
2. Testez sur un corpus de 4-5 documents
3. Créez une visualisation des scores TF-IDF
4. Identifiez les mots les plus discriminants pour chaque document

#### Code de départ
```python
import math
from collections import Counter

def calculate_tf(document):
    """Calcule la fréquence des termes"""
    pass

def calculate_idf(documents, vocabulary):
    """Calcule l'inverse document frequency"""
    pass

def calculate_tfidf(documents):
    """Fonction principale qui combine TF et IDF"""
    pass

# Corpus de test
corpus_test = [
    "Le chat mange du poisson frais",
    "Le chien mange des croquettes",
    "Machine learning et intelligence artificielle",
    "Python pour le machine learning avancé",
    "Les algorithmes de deep learning"
]
```

#### Critères d'évaluation
- [ ] Calculs TF et IDF corrects
- [ ] Implémentation complète de TF-IDF
- [ ] Visualisation claire des résultats
- [ ] Analyse des mots discriminants

---

### 📝 **Exercice 7 : Similarité Cosinus**
**Objectif :** Maîtriser le calcul de similarité entre textes
**Difficulté :** ⭐⭐⭐☆☆ | **Points :** 7

#### Instructions
1. Créez une fonction qui compare 10 paires de phrases
2. Implémentez la similarité cosinus from scratch
3. Créez un ranking humain vs machine des similarités
4. Analysez les cas où l'humain et la machine divergent

#### Code de départ
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_similarity_manual(vec1, vec2):
    """Votre implémentation de la similarité cosinus"""
    pass

def compare_similarities(phrases_pairs):
    """Compare les similarités humaines vs machine"""
    pass

# Paires de phrases à tester
test_pairs = [
    ("Python est génial", "J'adore Python"),
    ("Il fait beau", "Le soleil brille"),
    ("Chat noir", "Chien blanc"),
    ("Machine learning", "Intelligence artificielle"),
    # Ajoutez 6 autres paires...
]
```

#### Critères d'évaluation
- [ ] Fonction cosinus correcte
- [ ] Comparaison humain/machine documentée
- [ ] Analyse des divergences
- [ ] Insights sur les limites

---

### 📝 **Exercice 8 : Exploration Word2Vec**
**Objectif :** Découvrir la puissance des word embeddings
**Difficulté :** ⭐⭐⭐⭐☆ | **Points :** 7

#### Instructions
1. Chargez un modèle Word2Vec pré-entraîné français
2. Trouvez 10 analogies qui fonctionnent bien
3. Créez un quiz "devine le mot manquant"
4. Analysez les limitations et biais du modèle

#### Code de départ
```python
import spacy

# Chargement du modèle
nlp = spacy.load("fr_core_news_md")

def test_analogies(word_a, word_b, word_c):
    """Teste l'analogie: a est à b ce que c est à ?"""
    pass

def create_word_quiz():
    """Crée un quiz interactif"""
    pass

def analyze_word_clusters(words_list):
    """Analyse les clusters de mots similaires"""
    pass

# Mots à analyser
test_words = [
    "roi", "reine", "homme", "femme",
    "Paris", "France", "Londres", "Angleterre",
    # Ajoutez d'autres mots...
]
```

#### Critères d'évaluation
- [ ] 10 analogies fonctionnelles trouvées
- [ ] Quiz interactif créé
- [ ] Analyse critique des biais
- [ ] Visualisation des clusters

---

## 🎯 **Projet Final : Détecteur de Plagiat/Similarité**

### 🏆 **Cahier des Charges**

Créez un système complet de détection de similarité entre textes qui :

#### **Fonctionnalités Principales**
1. **Interface utilisateur** (CLI ou web avec Streamlit)
2. **Plusieurs méthodes** de vectorisation (BoW, TF-IDF, embeddings)
3. **Calcul de similarité** avec différentes métriques
4. **Visualisations** des résultats
5. **Base de données** de documents de référence

#### **Spécifications Techniques**

```python
class DetecteurSimilarite:
    def __init__(self, method='tfidf'):
        """
        Initialise le détecteur
        method: 'bow', 'tfidf', ou 'embeddings'
        """
        pass
    
    def add_reference_document(self, text, title):
        """Ajoute un document à la base de référence"""
        pass
    
    def check_similarity(self, query_text, threshold=0.7):
        """
        Vérifie la similarité avec tous les documents de référence
        Retourne: [(doc_title, similarity_score), ...]
        """
        pass
    
    def visualize_results(self, results):
        """Crée des graphiques des résultats"""
        pass
    
    def generate_report(self, query_text, results):
        """Génère un rapport détaillé"""
        pass
```

#### **Interface Streamlit Suggérée**

```python
import streamlit as st

def main():
    st.title("🔍 Détecteur de Similarité de Textes")
    
    # Sidebar pour configuration
    st.sidebar.header("Configuration")
    method = st.sidebar.selectbox(
        "Méthode de vectorisation",
        ["TF-IDF", "Bag of Words", "Word Embeddings"]
    )
    
    threshold = st.sidebar.slider(
        "Seuil de similarité",
        0.0, 1.0, 0.7
    )
    
    # Interface principale
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Texte à analyser")
        query_text = st.text_area("Entrez votre texte:", height=200)
        
        if st.button("Analyser"):
            # Logique d'analyse
            pass
    
    with col2:
        st.header("Résultats")
        # Affichage des résultats
        pass
```

#### **Dataset de Test Fourni**

- **50 articles Wikipedia** français (sciences, histoire, littérature)
- **30 essais d'étudiants** avec versions originales et plagiat détecté
- **100 tweets** sur des sujets variés
- **20 articles de presse** sur l'actualité technologique

#### **Critères d'Évaluation**

| Critère | Points | Description |
|---------|--------|-------------|
| **Fonctionnalité** | 30 | Toutes les features demandées |
| **Qualité du Code** | 20 | Code propre, commenté, structuré |
| **Interface** | 20 | UX intuitive et attractive |
| **Visualisations** | 15 | Graphiques informatifs |
| **Documentation** | 10 | README détaillé |
| **Tests** | 5 | Cas de test variés |

#### **Bonus Possibles** *(+10 points)*
- [ ] Support multilingue
- [ ] API REST
- [ ] Déploiement en ligne
- [ ] Détection de paraphrase
- [ ] Mode batch pour plusieurs fichiers

---

## 📊 **Auto-Évaluation**

### ✅ **Checklist de Compréhension**

Avant de passer au module suivant, assurez-vous de pouvoir :

- [ ] **Expliquer** la différence entre BoW, TF-IDF et embeddings
- [ ] **Calculer** manuellement un score TF-IDF
- [ ] **Interpréter** une similarité cosinus
- [ ] **Choisir** la bonne métrique selon le contexte
- [ ] **Identifier** les limitations de chaque approche
- [ ] **Implémenter** un système de comparaison de textes
- [ ] **Visualiser** des résultats de vectorisation
- [ ] **Débugger** des problèmes de preprocessing

### 🎯 **Quiz d'Auto-Évaluation**

#### **Question 1 :** Vrai ou Faux ?
"TF-IDF donne plus d'importance aux mots rares qu'aux mots fréquents"

<details>
<summary>Voir la réponse</summary>

**Vrai.** IDF (Inverse Document Frequency) pénalise les mots qui apparaissent dans beaucoup de documents, donnant plus de poids aux mots rares et discriminants.
</details>

#### **Question 2 :** Calcul pratique
Soit le vocabulaire ["chat", "mange", "poisson"] et les documents :
- Doc1: "Le chat mange"
- Doc2: "Le poisson nage"

Quelle est la représentation BoW de Doc1 ?

<details>
<summary>Voir la réponse</summary>

**[1, 1, 0]** - "chat":1, "mange":1, "poisson":0
</details>

#### **Question 3 :** Analogie Word2Vec
Complétez : "roi - homme + femme = ___"

<details>
<summary>Voir la réponse</summary>

**reine** - C'est l'analogie classique qui démontre que Word2Vec capture les relations sémantiques.
</details>

---

## 🔗 **Ressources et Liens Utiles**

### 📚 **Documentation Officielle**
- [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [spaCy Word Vectors](https://spacy.io/usage/vectors-similarity)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)

### 🎓 **Ressources Pédagogiques**
- [Visualisation interactive TF-IDF](http://www.tfidf.com/)
- [Word2Vec expliqué visuellement](https://ronxin.github.io/wevi/)
- [Cours Stanford CS224N](http://web.stanford.edu/class/cs224n/)

### 🔧 **Outils Pratiques**
- [Datasets français pour NLP](https://github.com/clu-ling/french-nlp-datasets)
- [Modèles spaCy français](https://spacy.io/models/fr)
- [Word2Vec français pré-entraîné](https://fauconnier.github.io/)

### 📖 **Lectures Approfondies**
- Paper original TF-IDF (Salton & McGill, 1983)
- Word2Vec original papers (Mikolov et al., 2013)
- "Speech and Language Processing" (Jurafsky & Martin)

---

## 🚀 **Transition vers le Module 3**

### 🎯 **Ce que Vous Avez Acquis**

Félicitations ! Vous maîtrisez maintenant :
- ✅ La transformation de texte en représentations numériques
- ✅ Les techniques de vectorisation (BoW, TF-IDF, embeddings)
- ✅ Le calcul de similarité entre documents
- ✅ L'utilisation d'outils professionnels (sklearn, spaCy)

### 🔮 **Ce qui Vous Attend**

Dans le **Module 3 : Classification et Analyse de Sentiments**, vous apprendrez à :
- 🎯 Entraîner des modèles de classification sur vos vecteurs
- 😊😡 Détecter automatiquement les émotions dans les textes
- 📊 Évaluer et optimiser les performances de vos modèles
- 🏭 Créer un pipeline de production robuste

### 🌉 **Préparation**

Assurez-vous d'avoir :
- [ ] Validé tous les exercices de ce module
- [ ] Terminé le projet détecteur de similarité
- [ ] Compris les concepts de vectorisation
- [ ] Installé les dépendances pour la classification

---

## 📞 **Support et Communauté**

### 🆘 **En Cas de Problème**

1. **FAQ Module 2** : Consultez les questions fréquentes
2. **Debugging Guide** : Solutions aux erreurs communes
3. **Forum Communauté** : Posez vos questions
4. **Office Hours** : Sessions Q&R hebdomadaires

### 🤝 **Contribuer**

- Proposez des améliorations via GitHub Issues
- Partagez vos projets créatifs
- Aidez les autres étudiants
- Suggérez de nouveaux datasets

---

**🎉 Bravo ! Vous avez terminé le Module 2 : Vectorisation !**

*Prochain arrêt : Module 3 - Classification et Analyse de Sentiments* 🎯