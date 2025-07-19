# 📝 Module 1 : Anatomie du Texte
## *Apprendre à découper le langage humain pour les machines*

---

## 🎬 **Vidéo d'Introduction** *(5 minutes)*

### 📺 **[Regarder la Vidéo Intro](video-intro.md)**

**💡 Résumé de la vidéo :**
> "Découvrez pourquoi Siri vous comprend parfois si mal ! Dans ce module, nous allons décortiquer le texte comme un chirurgien pour comprendre comment transformer des mots en quelque chose que les machines peuvent digérer. Vous allez apprendre à 'découper' intelligemment le langage humain et nettoyer vos données textuelles comme un pro. À la fin, vous aurez créé votre propre explorateur de texte qui révèle tous les secrets cachés dans n'importe quel document !"

---

## 🎯 **Objectifs d'Apprentissage**

À la fin de ce module, vous serez capable de :

- ✅ **Comprendre** pourquoi les machines ont du mal avec le langage humain
- ✅ **Maîtriser** la tokenisation et ses défis
- ✅ **Implémenter** des pipelines de preprocessing robustes
- ✅ **Utiliser** spaCy et NLTK efficacement
- ✅ **Déboguer** les problèmes courants de traitement de texte
- ✅ **Créer** un explorateur de texte interactif

---

## 📚 **Plan du Module**

| Section | Contenu | Durée | Difficulté |
|---------|---------|-------|------------|
| [01](#01-introduction) | Introduction au Problème | 45 min | ⭐☆☆ |
| [02](#02-tokenisation) | Tokenisation Intelligente | 60 min | ⭐⭐☆ |
| [03](#03-preprocessing) | Preprocessing et Nettoyage | 60 min | ⭐⭐☆ |
| [04](#04-outils) | Outils spaCy et NLTK | 45 min | ⭐⭐☆ |
| [Exercices](#exercices) | 4 Exercices Pratiques | 120 min | ⭐⭐⭐ |
| [Projet](#projet-final) | Explorateur de Texte | 90 min | ⭐⭐⭐ |

**⏱️ Durée totale estimée : 6-7 heures**

---

## 📖 **01. Introduction au Problème**

### 🤔 **Pourquoi les Machines Détestent le Langage Humain ?**

Imaginez que vous devez expliquer à un extraterrestre ce que signifie "Il fait un temps de chien" quand il pleut. Pas évident, non ? C'est exactement le problème des machines avec notre langage !

#### **Le Fossé Texte ↔ Nombres**

```python
# Ce que nous voyons
texte_humain = "Salut ! Comment ça va ? 😊"

# Ce que voit l'ordinateur (représentation ASCII)
[83, 97, 108, 117, 116, 32, 33, 32, 67, 111, 109, 109, 101, 110, 116, ...]
```

**🎯 Problèmes majeurs :**

1. **Ambiguïté** : "La poule du pot" (repas vs récipient ?)
2. **Contexte** : "Il est cool" vs "Il fait cool"
3. **Variations** : "super", "génial", "fantastique" = même sens
4. **Erreurs** : "bjr", "slt", "cc" = variations informelles
5. **Multilingue** : mélanges français/anglais/argot

#### **🧪 Expérience : Tester Google Translate**

Essayez de traduire ces phrases et observez les erreurs :

```
1. "Je suis dans le rouge ce mois-ci"
2. "Il a pris la mouche"
3. "C'est du chinois pour moi"
4. "Elle a un poil dans la main"
```

**💡 Analyse :** Google Translate traduit littéralement car il ne comprend pas les expressions idiomatiques !

### 🎯 **Notre Mission**

Transformer ce chaos linguistique en données exploitables par les machines. C'est le préalable OBLIGATOIRE à toute application NLP !

---

## 🔪 **02. Tokenisation Intelligente**

### 📚 **Qu'est-ce que la Tokenisation ?**

**Définition :** Découper un texte en unités plus petites (tokens) : mots, phrases, caractères, etc.

**Analogie :** Comme découper les ingrédients avant de cuisiner ! 👨‍🍳

### 🚫 **L'Approche Naïve (qui ne marche pas)**

```python
# Méthode débutant (MAUVAISE)
texte = "Bonjour, comment allez-vous ? J'espère que ça va !"
tokens_naifs = texte.split(" ")
print(tokens_naifs)
# Résultat : ['Bonjour,', 'comment', 'allez-vous', '?', "J'espère", ...]
```

**🔥 Problèmes identifiés :**
- Ponctuation collée aux mots
- Contractions mal gérées
- Majuscules conservées
- Espaces multiples ignorés

### ✅ **L'Approche Intelligente**

```python
import spacy

# Chargement du modèle français
nlp = spacy.load("fr_core_news_sm")

def tokeniser_intelligemment(texte):
    """Tokenise un texte avec spaCy"""
    doc = nlp(texte)
    
    tokens = []
    for token in doc:
        if not token.is_space:  # Ignorer les espaces
            tokens.append({
                'texte': token.text,
                'lemme': token.lemma_,
                'pos': token.pos_,
                'est_ponctuation': token.is_punct,
                'est_stop_word': token.is_stop
            })
    
    return tokens

# Test
texte = "Bonjour, comment allez-vous ? J'espère que ça va !"
tokens = tokeniser_intelligemment(texte)

for token in tokens:
    print(f"{token['texte']:15} | {token['lemme']:15} | {token['pos']}")
```

**📊 Résultat attendu :**
```
Bonjour         | bonjour         | INTJ
,               | ,               | PUNCT
comment         | comment         | ADV
allez           | aller           | VERB
-               | -               | PUNCT
vous            | vous            | PRON
?               | ?               | PUNCT
J'              | je              | PRON
espère          | espérer         | VERB
que             | que             | SCONJ
ça              | ça              | PRON
va              | aller           | VERB
!               | !               | PUNCT
```

### 🎯 **Types de Tokenisation**

#### **1. Tokenisation par Mots**
```python
# Standard pour la plupart des applications
doc = nlp("Python est génial pour le NLP !")
mots = [token.text for token in doc if not token.is_punct]
# Résultat : ['Python', 'est', 'génial', 'pour', 'le', 'NLP']
```

#### **2. Tokenisation par Phrases**
```python
# Utile pour l'analyse de documents longs
texte = "Python est super. J'adore ce langage ! Et vous ?"
doc = nlp(texte)
phrases = [sent.text for sent in doc.sents]
# Résultat : ['Python est super.', "J'adore ce langage !", 'Et vous ?']
```

#### **3. Tokenisation par Caractères**
```python
# Pour les langues sans espaces (chinois) ou l'analyse fine
mot = "génial"
caracteres = list(mot)
# Résultat : ['g', 'é', 'n', 'i', 'a', 'l']
```

### 🔧 **Gestion des Cas Spéciaux**

#### **URLs et Mentions**
```python
import re

def nettoyer_urls_mentions(texte):
    """Remplace URLs et mentions par des tokens spéciaux"""
    # URLs
    texte = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                   '<URL>', texte)
    
    # Mentions Twitter
    texte = re.sub(r'@\w+', '<MENTION>', texte)
    
    # Hashtags
    texte = re.sub(r'#\w+', '<HASHTAG>', texte)
    
    return texte

# Test
tweet = "Regardez cette vidéo https://youtube.com/watch?v=123 @elonmusk #IA #cool"
tweet_propre = nettoyer_urls_mentions(tweet)
print(tweet_propre)
# Résultat : "Regardez cette vidéo <URL> <MENTION> <HASHTAG> <HASHTAG>"
```

#### **Émojis et Caractères Spéciaux**
```python
import emoji

def gerer_emojis(texte):
    """Convertit les émojis en texte descriptif"""
    return emoji.demojize(texte, language='fr')

# Test
texte_emoji = "J'adore Python ! 😍🐍"
texte_sans_emoji = gerer_emojis(texte_emoji)
print(texte_sans_emoji)
# Résultat : "J'adore Python ! :visage_souriant_avec_des_yeux_en_forme_de_cœur::serpent:"
```

---

## 🧹 **03. Preprocessing et Nettoyage**

### 🎯 **Objectif du Preprocessing**

Transformer un texte "sale" en texte "propre" et standardisé pour l'analyse.

**Principe :** Plus vos données sont propres, meilleurs seront vos résultats !

### 🔧 **Pipeline de Nettoyage Standard**

```python
import re
import string
from unidecode import unidecode

class NettoyeurTexte:
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_sm")
    
    def nettoyer_complet(self, texte):
        """Pipeline complet de nettoyage"""
        # Étape 1 : Normalisation de base
        texte = self.normaliser_base(texte)
        
        # Étape 2 : Gestion des entités spéciales
        texte = self.gerer_entites_speciales(texte)
        
        # Étape 3 : Tokenisation et lemmatisation
        tokens = self.tokeniser_et_lemmatiser(texte)
        
        # Étape 4 : Filtrage
        tokens = self.filtrer_tokens(tokens)
        
        return tokens
    
    def normaliser_base(self, texte):
        """Normalisation basique du texte"""
        # Conversion en minuscules
        texte = texte.lower()
        
        # Suppression des accents (optionnel)
        # texte = unidecode(texte)
        
        # Suppression des caractères de contrôle
        texte = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', texte)
        
        # Normalisation des espaces
        texte = re.sub(r'\s+', ' ', texte)
        
        return texte.strip()
    
    def gerer_entites_speciales(self, texte):
        """Gestion des URLs, emails, etc."""
        # URLs
        texte = re.sub(r'http[s]?://\S+', '<URL>', texte)
        
        # Emails
        texte = re.sub(r'\S+@\S+', '<EMAIL>', texte)
        
        # Numéros de téléphone français
        texte = re.sub(r'0[1-9](?:[0-9]{8})', '<TELEPHONE>', texte)
        
        # Dates (format basique)
        texte = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', '<DATE>', texte)
        
        return texte
    
    def tokeniser_et_lemmatiser(self, texte):
        """Tokenisation avec lemmatisation"""
        doc = self.nlp(texte)
        
        tokens = []
        for token in doc:
            if not token.is_space:
                tokens.append({
                    'original': token.text,
                    'lemme': token.lemma_,
                    'pos': token.pos_,
                    'est_alpha': token.is_alpha,
                    'est_stop': token.is_stop,
                    'est_punct': token.is_punct
                })
        
        return tokens
    
    def filtrer_tokens(self, tokens):
        """Filtrage des tokens selon des critères"""
        tokens_filtres = []
        
        for token in tokens:
            # Garder seulement les mots alphabétiques
            if not token['est_alpha']:
                continue
            
            # Supprimer les stop words (optionnel)
            if token['est_stop']:
                continue
            
            # Supprimer les mots trop courts
            if len(token['lemme']) < 2:
                continue
            
            tokens_filtres.append(token['lemme'])
        
        return tokens_filtres

# Utilisation
nettoyeur = NettoyeurTexte()

texte_sale = """
Salut !!! Comment ça va ??? 😊 
Mon email: test@exemple.com
Site web: https://monsite.fr
Tél: 0123456789
On se voit le 15/03/2024 ?
"""

tokens_propres = nettoyeur.nettoyer_complet(texte_sale)
print("Tokens nettoyés :", tokens_propres)
# Résultat attendu : ['salut', 'aller', 'email', 'site', 'web', 'voir']
```

### 🎛️ **Options de Preprocessing Avancées**

#### **1. Gestion des Négations**
```python
def gerer_negations(texte):
    """Transforme 'ne ... pas' en 'ne_pas'"""
    # Négations françaises courantes
    negations = [
        (r'\bne\s+(\w+)\s+pas\b', r'ne_\1_pas'),
        (r'\bn\'(\w+)\s+pas\b', r'ne_\1_pas'),
        (r'\bne\s+(\w+)\s+jamais\b', r'ne_\1_jamais'),
        (r'\bne\s+(\w+)\s+plus\b', r'ne_\1_plus'),
    ]
    
    for pattern, replacement in negations:
        texte = re.sub(pattern, replacement, texte, flags=re.IGNORECASE)
    
    return texte

# Test
phrase = "Je ne suis pas content"
phrase_neg = gerer_negations(phrase)
print(phrase_neg)  # "Je ne_suis_pas content"
```

#### **2. Expansion des Contractions**
```python
def expandre_contractions(texte):
    """Expanse les contractions françaises"""
    contractions = {
        "j'ai": "je ai",
        "j'étais": "je étais",
        "c'est": "ce est",
        "c'était": "ce était",
        "l'ai": "le ai",
        "n'ai": "ne ai",
        "n'est": "ne est",
        "qu'il": "que il",
        "qu'elle": "que elle",
    }
    
    for contraction, expansion in contractions.items():
        texte = texte.replace(contraction, expansion)
    
    return texte
```

#### **3. Correction Orthographique Basique**
```python
def corriger_erreurs_courantes(texte):
    """Corrige les erreurs d'orthographe courantes"""
    corrections = {
        "bjr": "bonjour",
        "bsr": "bonsoir",
        "slt": "salut",
        "cc": "coucou",
        "pk": "pourquoi",
        "pr": "pour",
        "ds": "dans",
        "vs": "vous",
        "ts": "tous",
    }
    
    for erreur, correction in corrections.items():
        texte = re.sub(r'\b' + erreur + r'\b', correction, texte, flags=re.IGNORECASE)
    
    return texte
```

---

## 🛠️ **04. Outils spaCy et NLTK**

### 🥊 **spaCy vs NLTK : Le Match du Siècle**

| Critère | spaCy | NLTK |
|---------|-------|------|
| **Performance** | ⚡ Très rapide | 🐌 Plus lent |
| **Facilité d'usage** | 😊 Simple | 🤔 Plus complexe |
| **Modèles pré-entraînés** | ✅ Excellents | ⚠️ Basiques |
| **Production** | ✅ Prêt pour prod | ⚠️ Plutôt recherche |
| **Communauté** | 🔥 Très active | 📚 Académique |

**🎯 Verdict :** spaCy pour la production, NLTK pour l'expérimentation !

### 🚀 **Installation et Setup spaCy**

```bash
# Installation
pip install spacy

# Téléchargement du modèle français
python -m spacy download fr_core_news_sm

# Pour plus de précision (modèle plus lourd)
python -m spacy download fr_core_news_lg
```

### 📋 **spaCy : Guide de Démarrage**

```python
import spacy

# Chargement du modèle
nlp = spacy.load("fr_core_news_sm")

def analyser_texte_complet(texte):
    """Analyse complète avec spaCy"""
    doc = nlp(texte)
    
    print("🔍 ANALYSE COMPLÈTE")
    print("=" * 50)
    
    # 1. Tokenisation de base
    print("\n📝 TOKENS :")
    for token in doc:
        print(f"{token.text:15} | {token.lemma_:15} | {token.pos_:8} | {token.tag_:8}")
    
    # 2. Entités nommées
    print("\n🏷️ ENTITÉS NOMMÉES :")
    for ent in doc.ents:
        print(f"{ent.text:20} | {ent.label_:10} | {spacy.explain(ent.label_)}")
    
    # 3. Phrases
    print("\n📖 PHRASES :")
    for i, sent in enumerate(doc.sents, 1):
        print(f"Phrase {i}: {sent.text}")
    
    # 4. Dépendances syntaxiques (aperçu)
    print("\n🌳 DÉPENDANCES (échantillon) :")
    for token in doc[:5]:  # Premiers 5 tokens seulement
        print(f"{token.text} ← {token.dep_} ← {token.head.text}")

# Test complet
texte_test = """
Salut ! Je m'appelle Marie Dupont et je travaille chez Google France.
J'habite à Paris depuis 2020. Mon email est marie@google.com.
"""

analyser_texte_complet(texte_test)
```

### 📚 **NLTK : Les Incontournables**

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Téléchargements nécessaires (à faire une fois)
nltk.download('punkt')
nltk.download('stopwords')

def analyser_avec_nltk(texte):
    """Analyse basique avec NLTK"""
    
    # Tokenisation par phrases
    phrases = sent_tokenize(texte, language='french')
    print(f"📖 Nombre de phrases : {len(phrases)}")
    
    # Tokenisation par mots
    mots = word_tokenize(texte, language='french')
    print(f"📝 Nombre de mots : {len(mots)}")
    
    # Stop words français
    stop_words_fr = set(stopwords.words('french'))
    mots_filtres = [mot for mot in mots if mot.lower() not in stop_words_fr and mot.isalpha()]
    print(f"🔍 Mots significatifs : {len(mots_filtres)}")
    
    # Stemming (racines des mots)
    stemmer = SnowballStemmer('french')
    mots_racines = [stemmer.stem(mot) for mot in mots_filtres]
    
    print("\n📊 ÉCHANTILLON D'ANALYSE :")
    for original, racine in zip(mots_filtres[:10], mots_racines[:10]):
        print(f"{original:15} → {racine}")

# Test
texte_nltk = "Les développeurs adorent programmer en Python car c'est un langage fantastique !"
analyser_avec_nltk(texte_nltk)
```

### 🎯 **Comparaison Pratique**

```python
import time

def comparer_performances(texte, nb_iterations=100):
    """Compare les performances spaCy vs NLTK"""
    
    # Test spaCy
    start_spacy = time.time()
    for _ in range(nb_iterations):
        doc = nlp(texte)
        tokens_spacy = [token.lemma_ for token in doc if token.is_alpha]
    temps_spacy = time.time() - start_spacy
    
    # Test NLTK
    stemmer = SnowballStemmer('french')
    stop_words = set(stopwords.words('french'))
    
    start_nltk = time.time()
    for _ in range(nb_iterations):
        tokens_nltk = word_tokenize(texte, language='french')
        tokens_nltk = [stemmer.stem(token) for token in tokens_nltk 
                      if token.lower() not in stop_words and token.isalpha()]
    temps_nltk = time.time() - start_nltk
    
    print(f"⚡ spaCy  : {temps_spacy:.3f}s | {len(tokens_spacy)} tokens")
    print(f"🐌 NLTK   : {temps_nltk:.3f}s | {len(tokens_nltk)} tokens")
    print(f"📊 Ratio  : spaCy est {temps_nltk/temps_spacy:.1f}x plus rapide")

# Test de performance
texte_perf = "Python est un langage de programmation fantastique pour le machine learning."
comparer_performances(texte_perf)
```

---

## 🏋️ **Exercices Pratiques**

### 📝 **Exercice 1 : Tokenisation Naïve vs Intelligente**
**🎯 Objectif :** Comprendre les limites de la tokenisation simple

**📋 Énoncé :**
1. Implémentez un tokenizer naïf avec `split()`
2. Testez sur 5 phrases problématiques fournies
3. Identifiez et listez tous les problèmes
4. Comparez avec spaCy
5. Rédigez un rapport de 200 mots sur vos observations

**🏅 Critères de réussite :**
- [ ] Code fonctionnel pour les deux approches
- [ ] Au moins 5 problèmes identifiés
- [ ] Comparaison quantitative (nombre de tokens)
- [ ] Analyse qualitative dans le rapport

---

### 📝 **Exercice 2 : Comparaison d'Outils**
**🎯 Objectif :** Maîtriser spaCy et NLTK

**📋 Énoncé :**
1. Tokenisez le même corpus avec spaCy et NLTK
2. Mesurez les performances (temps d'exécution)
3. Comparez la qualité des résultats
4. Créez un tableau comparatif détaillé
5. Recommandez un outil selon le contexte

**🏅 Critères de réussite :**
- [ ] Benchmarks de performance réalisés
- [ ] Tableau comparatif complet
- [ ] Recommandations justifiées
- [ ] Code optimisé et commenté

---

### 📝 **Exercice 3 : Nettoyage de Tweets**
**🎯 Objectif :** Créer un pipeline de preprocessing robuste

**📋 Énoncé :**
1. Dataset fourni : 100 tweets avec URLs, mentions, émojis
2. Créez une classe `NettoyeurTweets`
3. Implémentez 5 étapes de nettoyage minimum
4. Générez un rapport avant/après avec statistiques
5. Testez sur des cas edge cases

**🏅 Critères de réussite :**
- [ ] Pipeline modulaire et réutilisable
- [ ] Gestion des cas spéciaux (émojis, URLs, etc.)
- [ ] Rapport statistique détaillé
- [ ] Tests sur cas difficiles validés

---

### 📝 **Exercice 4 : Debug de Tokenisation**
**🎯 Objectif :** Développer des compétences de debugging

**📋 Énoncé :**
1. 5 textes "cassés" sont fournis avec des erreurs de tokenisation
2. Identifiez la source de chaque problème
3. Proposez une solution pour chaque cas
4. Implémentez les corrections
5. Documentez votre approche de debugging

**🏅 Critères de réussite :**
- [ ] Tous les bugs identifiés correctement
- [ ] Solutions élégantes implémentées
- [ ] Code défensif ajouté
- [ ] Documentation du processus de debug

---

## 🎯 **Projet Final : Explorateur de Texte Interactif**

### 🚀 **Description du Projet**

Créez une application qui analyse n'importe quel texte et révèle ses "secrets linguistiques" !

### 📋 **Cahier des Charges**

#### **Fonctionnalités Obligatoires :**

1. **📤 Upload de Fichier**
   - Support : .txt, .pdf, .docx
   - Limite : 10 MB maximum
   - Encodage automatique détecté

2. **🔍 Analyse Complète**
   - Statistiques de base (mots, phrases, caractères)
   - Distribution des types de mots (noms, verbes, etc.)
   - Entités nommées détectées
   - Mots les plus fréquents (top 10)
   - Complexité du texte (longueur moyenne des phrases)

3. **🧹 Pipeline de Nettoyage**
   - Texte original vs texte nettoyé
   - Options configurables de preprocessing
   - Visualisation avant/après

4. **📊 Visualisations**
   - Nuage de mots
   - Graphique de fréquence des mots
   - Distribution des longueurs de phrases

5. **💾 Export des Résultats**
   - JSON avec toutes les analyses
   - CSV des mots avec leurs propriétés
   - Rapport PDF généré automatiquement

#### **Interface Utilisateur :**

```python
# Structure de l'application Streamlit
import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.title("🔍 Explorateur de Texte Intelligent")

# Sidebar pour les options
st.sidebar.header("⚙️ Options de Traitement")
supprimer_stopwords = st.sidebar.checkbox("Supprimer les stop words")
lemmatiser = st.sidebar.checkbox("Lemmatiser les mots", value=True)
min_longueur = st.sidebar.slider("Longueur minimale des mots", 1, 10, 2)

# Zone d'upload
uploaded_file = st.file_uploader("📤 Choisissez votre fichier", 
                                 type=['txt', 'pdf', 'docx'])

if uploaded_file:
    # Traitement et affichage des résultats
    pass
```

### 🏆 **Critères d'Évaluation**

| Critère | Points | Description |
|---------|--------|-------------|
| **Fonctionnalités** | 40 pts | Toutes les fonctions obligatoires |
| **Code Quality** | 20 pts | Propre, commenté, modulaire |
| **UX/UI** | 20 pts | Interface intuitive et jolie |
| **Innovation** | 10 pts | Fonctionnalités bonus créatives |
| **Documentation** | 10 pts | README.md du projet détaillé |

### 🎁 **Bonus Possibles (+20 points)**

- **🌐 Support multilingue** : Détection automatique de la langue
- **📈 Analyse de sentiment** : Polarité générale du texte
- **🔗 Extraction d'entités** : Personnes, lieux, organisations
- **⚡ Cache intelligent** : Éviter de reprocesser les mêmes textes
- **🎨 Thèmes personnalisables** : Interface customisable

### 📚 **Ressources Fournies**

#### **Code Template**
```python
# explorateur_texte.py - Structure de base
class ExplorateurTexte:
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_sm")
        
    def charger_fichier(self, fichier):
        """Charge et lit différents formats de fichiers"""
        pass
    
    def analyser_texte(self, texte, options):
        """Analyse complète du texte"""
        pass
    
    def generer_statistiques(self, doc):
        """Génère les statistiques descriptives"""
        stats = {
            'nb_mots': len([t for t in doc if t.is_alpha]),
            'nb_phrases': len(list(doc.sents)),
            'nb_caracteres': len(doc.text),
            'mots_uniques': len(set([t.lemma_ for t in doc if t.is_alpha])),
            'longueur_moyenne_phrase': None  # À calculer
        }
        return stats
    
    def extraire_entites(self, doc):
        """Extrait les entités nommées"""
        pass
    
    def generer_nuage_mots(self, tokens):
        """Crée un nuage de mots"""
        pass
```

#### **Datasets d'Exemple**
- `exemple_article.txt` : Article de journal français (500 mots)
- `exemple_roman.txt` : Extrait de roman (1000 mots)  
- `exemple_tweets.txt` : Collection de tweets (200 tweets)
- `exemple_technique.txt` : Documentation technique (800 mots)

#### **Utilitaires Fournis**
```python
# utils/file_readers.py
def lire_pdf(fichier):
    """Lecture de fichiers PDF"""
    pass

def lire_docx(fichier):
    """Lecture de fichiers Word"""
    pass

def detecter_encodage(fichier):
    """Détection automatique de l'encodage"""
    pass

# utils/visualizations.py
def creer_graphique_frequence(mots, frequences):
    """Graphique en barres des mots fréquents"""
    pass

def creer_nuage_mots(texte):
    """Génération de nuage de mots stylé"""
    pass

def creer_distribution_longueurs(phrases):
    """Histogramme des longueurs de phrases"""
    pass
```

---

## 📈 **Progression et Validation**

### ✅ **Checklist de Progression**

#### **Niveau Débutant (Bronze)** 🥉
- [ ] Comprendre pourquoi la tokenisation naïve ne suffit pas
- [ ] Installer et utiliser spaCy correctement
- [ ] Implémenter un pipeline de nettoyage basique
- [ ] Réaliser tous les exercices avec aide
- [ ] Créer un explorateur de texte minimal

#### **Niveau Intermédiaire (Argent)** 🥈
- [ ] Expliquer les différences spaCy vs NLTK avec exemples
- [ ] Gérer les cas spéciaux (URLs, émojis, négations)
- [ ] Optimiser les performances de traitement
- [ ] Résoudre les exercices de façon autonome
- [ ] Ajouter des fonctionnalités bonus au projet

#### **Niveau Avancé (Or)** 🥇
- [ ] Créer ses propres fonctions de preprocessing
- [ ] Déboguer et corriger des pipelines cassés
- [ ] Proposer des améliorations aux outils existants
- [ ] Aider d'autres étudiants sur les exercices
- [ ] Projet final avec innovations significatives

### 🎯 **Auto-Évaluation**

#### **Questions de Compréhension**

1. **Conceptuel** : Expliquez pourquoi `"n'est-ce pas".split()` pose problème
2. **Pratique** : Quand utiliser la lemmatisation vs le stemming ?
3. **Performance** : Pourquoi spaCy est-il plus rapide que NLTK ?
4. **Architecture** : Comment structurer un pipeline de preprocessing réutilisable ?

#### **Défis de Code**

```python
# Défi 1 : Tokenisation robuste
def tokeniser_robuste(texte):
    """
    Créez un tokenizer qui gère :
    - Les contractions françaises
    - Les URLs et emails
    - Les émojis
    - Les négations
    """
    pass

# Défi 2 : Détection d'anomalies
def detecter_anomalies_texte(texte):
    """
    Identifiez automatiquement :
    - Encodage incorrect
    - Texte généré par IA
    - Langue incorrecte
    - Formatting cassé
    """
    pass
```

---

## 🔗 **Liens et Ressources Complémentaires**

### 📖 **Documentation Officielle**
- [spaCy Documentation](https://spacy.io/usage/spacy-101) - Guide complet
- [NLTK Book](https://www.nltk.org/book/) - Référence académique
- [Regex101](https://regex101.com/) - Testeur d'expressions régulières

### 🎥 **Vidéos Recommandées**
- "spaCy IRL" (YouTube) - Cas d'usage réels
- "Text Preprocessing Explained" - Concepts visuels
- "French NLP Challenges" - Spécificités du français

### 📚 **Articles Avancés**
- "Why Tokenization Matters" - Importance en NLP
- "French Language Processing" - Défis spécifiques
- "Production NLP Pipelines" - Bonnes pratiques

### 🛠️ **Outils Complémentaires**
- **Stanza** : Alternative à spaCy (Stanford)
- **TextBlob** : Simplicité maximale
- **Polyglot** : Support multilingue étendu

---

## 🚀 **Préparation au Module 2**

### 🎯 **Ce que vous avez acquis**
- ✅ Maîtrise de la tokenisation intelligente
- ✅ Pipelines de preprocessing robustes  
- ✅ Utilisation experte de spaCy et NLTK
- ✅ Debugging de problèmes textuels
- ✅ Application complète fonctionnelle

### 🔮 **Ce qui vous attend**
- 🔢 **Vectorisation** : Transformer vos tokens en nombres
- 📊 **TF-IDF** : Mesurer l'importance des mots
- 🧠 **Word Embeddings** : Capturer le sens sémantique
- 🎯 **Similarité** : Comparer des textes automatiquement

### 💡 **Conseil pour la Suite**
> "Maintenant que vous savez 'découper' le langage, vous allez apprendre à le 'mesurer' ! Les tokens que vous créez ici vont devenir les coordonnées GPS de vos mots dans l'espace mathématique. Gardez vos pipelines de preprocessing : vous allez en avoir besoin !"

---

## 📞 **Support et Communauté**

### 🤝 **Aide Entre Étudiants**
- **GitHub Discussions** : Posez vos questions techniques
- **Discord NLP-France** : Chat en temps réel  
- **Peer Review** : Échangez vos codes et solutions

### 🆘 **En Cas de Blocage**
1. **Consultez la FAQ** des erreurs courantes
2. **Utilisez le debugger** intégré des notebooks
3. **Postez votre code** avec le message d'erreur exact
4. **Demandez une review** de votre approche

### 🏆 **Partager ses Réussites**
- **Portfolio GitHub** : Montrez vos projets
- **LinkedIn** : Partagez vos accomplissements
- **Blog technique** : Expliquez vos apprentissages

---

## 📝 **Notes Finales**

**🎉 Félicitations !** Si vous êtes arrivé jusqu'ici, vous maîtrisez maintenant les fondations du NLP. Vous savez transformer du texte "sale" en données exploitables par les machines.

**🔥 Point Clé :** 80% du travail en NLP, c'est le preprocessing ! Vous venez d'acquérir une compétence absolument cruciale.

**🚀 Next Level :** Dans le module 2, nous allons transformer vos beaux tokens en vecteurs mathématiques. C'est là que la vraie magie commence !

---

*Dernière mise à jour : [Date] | Version 1.0*
*Contributeurs : [Votre nom] | Retours : [email/discord]*