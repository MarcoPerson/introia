# 📝 Cours NLP - Structure README GitHub
## *Plan détaillé avec exercices et vidéos introductives*

---

## 📁 **Structure du Repository**

```
nlp-course/
├── README.md
├── module-1-anatomie-texte/
│   ├── README.md
│   ├── video-intro.md (lien vidéo + résumé)
│   ├── 01-introduction/
│   ├── 02-tokenisation/
│   ├── 03-preprocessing/
│   ├── 04-outils/
│   ├── exercices/
│   │   ├── exercice-01-tokenisation-naive.md
│   │   ├── exercice-02-comparaison-outils.md
│   │   ├── exercice-03-nettoyage-tweets.md
│   │   └── exercice-04-debug-tokenisation.md
│   ├── datasets/
│   ├── solutions/
│   └── projet-final/
├── module-2-vectorisation/
│   ├── README.md
│   ├── video-intro.md
│   ├── 01-probleme-representation/
│   ├── 02-bag-of-words/
│   ├── 03-tfidf/
│   ├── 04-word-embeddings/
│   ├── 05-similarite/
│   ├── exercices/
│   │   ├── exercice-05-bow-manuel.md
│   │   ├── exercice-06-tfidf-scratch.md
│   │   ├── exercice-07-similarite-cosinus.md
│   │   └── exercice-08-word2vec-exploration.md
│   ├── datasets/
│   ├── solutions/
│   └── projet-final/
├── module-3-classification/
│   ├── README.md
│   ├── video-intro.md
│   ├── 01-classification-supervisee/
│   ├── 02-preparation-donnees/
│   ├── 03-algorithmes/
│   ├── 04-feature-engineering/
│   ├── 05-evaluation/
│   ├── exercices/
│   │   ├── exercice-09-naive-bayes.md
│   │   ├── exercice-10-features-sentiments.md
│   │   ├── exercice-11-validation-croisee.md
│   │   └── exercice-12-optimisation-modele.md
│   ├── datasets/
│   ├── solutions/
│   └── projet-final/
├── module-4-chatbot/
│   ├── README.md
│   ├── video-intro.md
│   ├── 01-architecture-pipeline/
│   ├── 02-classification-intentions/
│   ├── 03-generation-reponses/
│   ├── 04-gestion-erreurs/
│   ├── exercices/
│   │   ├── exercice-13-intentions-basiques.md
│   │   ├── exercice-14-reponses-contextuelles.md
│   │   └── exercice-15-chatbot-complet.md
│   ├── datasets/
│   ├── solutions/
│   └── projet-final/
├── datasets-globaux/
├── utils/
├── requirements.txt
└── setup-guide.md
```

---

## 📋 **Plan Détaillé par Module**

### 🎬 **MODULE 1 : Anatomie du Texte**

#### 📹 **Vidéo d'Introduction** *(5 minutes)*
**Résumé de la vidéo :**
> "Découvrez pourquoi Siri vous comprend parfois si mal ! Dans ce module, nous allons décortiquer le texte comme un chirurgien pour comprendre comment transformer des mots en quelque chose que les machines peuvent digérer. Vous allez apprendre à 'découper' intelligemment le langage humain et nettoyer vos données textuelles comme un pro. À la fin, vous aurez créé votre propre explorateur de texte qui révèle tous les secrets cachés dans n'importe quel document !"

#### 📚 **Contenu Théorique**
1. **Introduction au Problème** (`01-introduction/`)
   - Pourquoi les machines détestent le langage humain
   - Le fossé entre texte et nombres
   - Exemples concrets d'échecs NLP

2. **Tokenisation** (`02-tokenisation/`)
   - Découpage naïf vs intelligent
   - Gestion de la ponctuation et cas spéciaux
   - Tokenisation par mots, phrases, caractères

3. **Preprocessing** (`03-preprocessing/`)
   - Normalisation du texte
   - Gestion des majuscules/minuscules
   - Suppression des stop words
   - Lemmatisation vs stemming

4. **Outils Essentiels** (`04-outils/`)
   - spaCy : installation et premiers pas
   - NLTK : les classiques qui marchent
   - Comparaison et choix d'outil

#### 🛠️ **Exercices Pratiques**

**📝 Exercice 1 : Tokenisation Naïve**
- Implémenter un tokenizer avec `split()`
- Identifier 5 problèmes majeurs
- Comparer avec spaCy sur des tweets réels

**📝 Exercice 2 : Comparaison d'Outils**
- Tokeniser le même texte avec spaCy et NLTK
- Mesurer les performances (temps + qualité)
- Créer un tableau comparatif

**📝 Exercice 3 : Nettoyage de Tweets**
- Dataset : 100 tweets avec emojis, URLs, mentions
- Créer une pipeline de nettoyage complète
- Avant/après avec statistiques

**📝 Exercice 4 : Debug de Tokenisation**
- 5 textes "problématiques" fournis
- Identifier pourquoi la tokenisation échoue
- Proposer des solutions

#### 🎯 **Projet Final Module 1**
**Explorateur de Texte Interactif**
- Interface simple (CLI ou Streamlit)
- Upload de fichier texte
- Statistiques complètes : mots, phrases, entités
- Export des résultats en JSON

---

### 🎬 **MODULE 2 : Vectorisation - Transformer les Mots en Nombres**

#### 📹 **Vidéo d'Introduction** *(5 minutes)*
**Résumé de la vidéo :**
> "Comment expliquer à un ordinateur que 'roi' et 'reine' sont similaires ? C'est le défi de la vectorisation ! Dans ce module, vous allez apprendre les techniques magiques pour transformer n'importe quel texte en coordonnées mathématiques. Du simple comptage de mots jusqu'aux mystérieux word embeddings, vous maîtriserez l'art de faire comprendre le SENS aux machines. Votre projet final ? Un détecteur de plagiat qui impressionnera vos professeurs !"

#### 📚 **Contenu Théorique**
1. **Le Problème de Représentation** (`01-probleme-representation/`)
   - Pourquoi les mots ne sont pas des nombres
   - L'espace vectoriel du langage
   - Notion de distance sémantique

2. **Bag of Words** (`02-bag-of-words/`)
   - Principe du sac de mots
   - Matrice terme-document
   - Avantages et limitations

3. **TF-IDF** (`03-tfidf/`)
   - Term Frequency : compter intelligemment
   - Inverse Document Frequency : détecter la rareté
   - Implémentation et optimisation

4. **Word Embeddings** (`04-word-embeddings/`)
   - Introduction à Word2Vec
   - Analogies vectorielles (roi - homme + femme = reine)
   - Utilisation pratique avec spaCy

5. **Calcul de Similarité** (`05-similarite/`)
   - Distance cosinus
   - Similarité euclidienne
   - Applications pratiques

#### 🛠️ **Exercices Pratiques**

**📝 Exercice 5 : Bag of Words Manuel**
- Implémenter BoW from scratch (sans sklearn)
- Tester sur 3 phrases simples
- Comparer avec CountVectorizer

**📝 Exercice 6 : TF-IDF from Scratch**
- Calculer TF-IDF manuellement
- Vérifier avec TfidfVectorizer
- Analyser les scores sur corpus d'actualités

**📝 Exercice 7 : Similarité Cosinus**
- Comparer 10 paires de phrases
- Ranking de similarité humain vs machine
- Analyser les divergences

**📝 Exercice 8 : Exploration Word2Vec**
- Charger un modèle pré-entraîné français
- Trouver 10 analogies qui marchent
- Créer un quiz "devine le mot manquant"

#### 🎯 **Projet Final Module 2**
**Détecteur de Plagiat/Similarité**
- Interface pour comparer des textes
- Plusieurs méthodes (BoW, TF-IDF, embeddings)
- Score de similarité avec visualisation
- Test sur des cas réels d'étudiants

---

### 🎬 **MODULE 3 : Classification et Analyse de Sentiments**

#### 📹 **Vidéo d'Introduction** *(5 minutes)*
**Résumé de la vidéo :**
> "Votre mission : créer une IA qui devine si un client est content ou furieux juste en lisant son commentaire ! Dans ce module, vous allez maîtriser les algorithmes de classification pour transformer votre machine en détective émotionnel. De Naive Bayes aux forêts aléatoires, vous apprendrez à entraîner des modèles qui comprennent les nuances humaines. Votre récompense ? Un analyseur de sentiments qui peut traiter des milliers d'avis en quelques secondes !"

#### 📚 **Contenu Théorique**
1. **Classification Supervisée** (`01-classification-supervisee/`)
   - Principe de l'apprentissage supervisé
   - Types de classification (binaire, multi-classe)
   - Train/validation/test split

2. **Préparation des Données** (`02-preparation-donnees/`)
   - Collecte et annotation des données
   - Équilibrage des classes
   - Gestion des données manquantes

3. **Algorithmes** (`03-algorithmes/`)
   - Naive Bayes : simple et efficace
   - SVM : frontières optimales
   - Random Forest : la puissance collective
   - Comparaison de performances

4. **Feature Engineering** (`04-feature-engineering/`)
   - Features linguistiques (longueur, ponctuation)
   - N-grammes et contexte
   - Gestion des négations
   - Features émotionnelles

5. **Évaluation** (`05-evaluation/`)
   - Métriques : accuracy, precision, recall, F1
   - Matrice de confusion
   - Validation croisée
   - Détection de l'overfitting

#### 🛠️ **Exercices Pratiques**

**📝 Exercice 9 : Naive Bayes Simple**
- Classification binaire positif/négatif
- Dataset : 200 commentaires étiquetés
- Évaluation avec métriques complètes

**📝 Exercice 10 : Features pour Sentiments**
- Créer 10 features custom (émojis, majuscules, etc.)
- Tester impact sur performance
- Feature importance analysis

**📝 Exercice 11 : Validation Croisée**
- Implémenter k-fold cross-validation
- Comparer 3 algorithmes
- Analyse statistique des résultats

**📝 Exercice 12 : Optimisation de Modèle**
- Grid search sur hyperparamètres
- Feature selection automatique
- Courbes d'apprentissage

#### 🎯 **Projet Final Module 3**
**Analyseur de Sentiments Multi-Classes**
- Classifications : positif/négatif/neutre
- Interface web simple (Streamlit)
- Évaluation sur données réelles
- Export du modèle entraîné

---

### 🎬 **MODULE 4 : Chatbot Intelligent - Assemblage Final**

#### 📹 **Vidéo d'Introduction** *(5 minutes)*
**Résumé de la vidéo :**
> "Le moment final est arrivé ! Vous allez assembler tout ce que vous avez appris pour créer un chatbot qui ne dit pas n'importe quoi. Plus qu'un simple générateur de réponses automatiques, votre bot comprendra les intentions, détectera les émotions et répondra de manière contextuellement appropriée. C'est le projet qui fera la différence sur votre CV : un vrai système NLP de bout en bout que vous pouvez déployer et montrer au monde entier !"

#### 📚 **Contenu Théorique**
1. **Architecture Pipeline** (`01-architecture-pipeline/`)
   - Design pattern pour NLP
   - Pipeline de traitement modulaire
   - Gestion des erreurs et fallbacks

2. **Classification d'Intentions** (`02-classification-intentions/`)
   - Définition des intentions métier
   - Collecte et préparation des données d'intention
   - Entraînement du classificateur d'intention

3. **Génération de Réponses** (`03-generation-reponses/`)
   - Templates de réponses contextuelles
   - Personnalisation selon sentiment + intention
   - Gestion de l'historique de conversation

4. **Gestion d'Erreurs** (`04-gestion-erreurs/`)
   - Détection de cas non couverts
   - Réponses de fallback intelligentes
   - Logging et amélioration continue

#### 🛠️ **Exercices Pratiques**

**📝 Exercice 13 : Intentions Basiques**
- Définir 5 intentions (salutation, question, problème, etc.)
- Créer dataset d'entraînement (50 exemples/intention)
- Entraîner classificateur avec validation

**📝 Exercice 14 : Réponses Contextuelles**
- Créer matrice intentions × sentiments
- Templates de réponses variées
- Test A/B sur qualité perçue

**📝 Exercice 15 : Chatbot Complet**
- Intégration de tous les composants
- Interface utilisateur (CLI + web)
- Tests end-to-end avec scénarios réels

#### 🎯 **Projet Final Module 4**
**Chatbot Support Client Complet**
- Pipeline NLP intégré (tokenisation → intention → sentiment → réponse)
- Interface web déployable
- Logging et analytics
- Documentation technique complète

---

## 📊 **Système d'Évaluation par Exercice**

### ✅ **Critères de Validation**

| Exercice | Type | Critères | Points |
|----------|------|----------|--------|
| 1-4 | Fondamentaux | Code fonctionnel + compréhension | 5 pts |
| 5-8 | Implémentation | Algorithme correct + optimisation | 7 pts |
| 9-12 | Classification | Métriques atteintes + analyse | 10 pts |
| 13-15 | Intégration | Fonctionnalité + documentation | 15 pts |

### 🏆 **Seuils de Réussite**
- **Bronze** : 60% des points (bases acquises)
- **Argent** : 75% des points (maîtrise solide)
- **Or** : 90% des points (expertise confirmée)

---

## 🎁 **Ressources Techniques**

### 📦 **Setup Initial**
```bash
# requirements.txt
spacy>=3.4.0
nltk>=3.7
scikit-learn>=1.1.0
pandas>=1.4.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
streamlit>=1.10.0
jupyter>=1.0.0
```

### 📁 **Datasets Fournis**
- **tweets_sentiments_fr.csv** (10k tweets annotés)
- **avis_clients_ecommerce.csv** (5k avis produits)
- **conversations_support.json** (500 dialogues support)
- **textes_wikipedia_fr.txt** (corpus pour vectorisation)

### 🛠️ **Utilitaires Communs**
```python
# utils/nlp_helpers.py
def nettoyer_texte(texte):
    """Fonction de nettoyage standardisée"""
    pass

def evaluer_modele(y_true, y_pred):
    """Métriques d'évaluation complètes"""
    pass

def visualiser_confusion_matrix(y_true, y_pred):
    """Graphiques d'évaluation"""
    pass
```

---

## 🚀 **Progression et Certification**

### 📈 **Suivi de Progression**
- [ ] Module 1 : Anatomie du Texte (4 exercices)
- [ ] Module 2 : Vectorisation (4 exercices)
- [ ] Module 3 : Classification (4 exercices)
- [ ] Module 4 : Chatbot (3 exercices)
- [ ] Portfolio Final : Documentation + déploiement

### 🎓 **Livrables Finaux**
1. **Repository GitHub** avec tous les projets
2. **Analyseur de Sentiments** déployé (Streamlit Cloud)
3. **Chatbot Fonctionnel** avec API
4. **Documentation Technique** complète
5. **Présentation** du parcours (README portfolio)

---

*Ce plan garantit 15 exercices progressifs avec support vidéo et structure GitHub professionnelle pour un apprentissage autonome optimal !*