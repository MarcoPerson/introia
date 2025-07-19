# 🤖 Module 4 : Chatbot Intelligent - L'Assemblage Final

> **"Créez un chatbot qui ne dit pas n'importe quoi !"**

## 🎬 **Vidéo d'Introduction** *(5 minutes)*

### 📺 **Script de la Vidéo**
```
🎯 ACCROCHE (30s) :
"Vous avez déjà pesté contre un chatbot qui vous répondait complètement à côté ? 
Aujourd'hui, on va créer un chatbot intelligent qui COMPREND vraiment ce que vous lui dites !"

🧠 ENJEUX (1min) :
"Dans ce module final, vous allez assembler TOUT ce qu'on a appris :
- La tokenisation pour découper le texte
- La vectorisation pour comprendre le sens  
- La classification pour détecter l'intention ET l'émotion
- Et les réponses contextuelles pour paraître humain !"

🚀 TEASING PROJET (1min30) :
"Votre mission ? Créer un chatbot de support client qui :
- Comprend si vous dites bonjour ou si vous êtes en colère
- Détecte si vous voulez de l'aide ou faire une réclamation
- Répond différemment selon votre état d'esprit
- Garde la conversation fluide et naturelle"

💡 MOTIVATION (1min30) :
"C'est LE projet portfolio qui fait la différence ! Un vrai système NLP 
de bout en bout que vous pouvez déployer et montrer au monde entier.
Plus qu'un simple bot, c'est une architecture complète que vous maîtriserez !"

🎯 PLAN MODULE (30s) :
"Au programme : Architecture pipeline, classification d'intentions, 
génération de réponses contextuelles, et gestion d'erreurs.
C'est parti pour l'aventure finale !"
```

---

## 🎯 **Objectifs du Module**

À la fin de ce module, vous serez capable de :
- ✅ Concevoir l'architecture complète d'un système NLP
- ✅ Classifier automatiquement les intentions utilisateur
- ✅ Générer des réponses adaptées au contexte émotionnel
- ✅ Gérer les cas d'erreur et les situations non prévues
- ✅ Déployer un chatbot fonctionnel avec interface web
- ✅ Documenter et maintenir un système NLP en production

---

## 🏗️ **1. Architecture Pipeline NLP**

### 🧠 **Concept Central : Le Pipeline Modulaire**

Un chatbot intelligent n'est pas une seule fonction magique, mais un **pipeline de composants** qui travaillent ensemble :

```
Entrée Utilisateur
    ↓
Preprocessing (nettoyage, tokenisation)
    ↓
Classification d'Intention (que veut-il ?)
    ↓
Analyse de Sentiment (dans quel état d'esprit ?)
    ↓
Génération de Réponse (comment répondre ?)
    ↓
Post-processing (personnalisation finale)
    ↓
Sortie Chatbot
```

### 📚 **Théorie : Design Patterns pour NLP**

#### **1.1 Le Pattern Pipeline**
```python
class NLPPipeline:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.intent_classifier = IntentClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.response_generator = ResponseGenerator()
    
    def process(self, user_input):
        # Chaque étape transforme et enrichit les données
        cleaned_text = self.preprocessor.clean(user_input)
        intent = self.intent_classifier.predict(cleaned_text)
        sentiment = self.sentiment_analyzer.predict(cleaned_text)
        response = self.response_generator.generate(intent, sentiment, cleaned_text)
        return response
```

#### **1.2 Gestion des États et Contexte**
Un bon chatbot se souvient de la conversation :
```python
class ConversationContext:
    def __init__(self):
        self.history = []
        self.user_profile = {}
        self.current_intent = None
        self.confidence_threshold = 0.7
    
    def update(self, user_input, intent, sentiment, response):
        self.history.append({
            'timestamp': datetime.now(),
            'user_input': user_input,
            'intent': intent,
            'sentiment': sentiment,
            'bot_response': response
        })
```

#### **1.3 Fallback et Gestion d'Erreurs**
```python
def handle_low_confidence(intent, confidence):
    if confidence < 0.5:
        return "clarification_needed"
    elif confidence < 0.7:
        return "confirmation_needed"
    else:
        return intent
```

---

## 🎯 **2. Classification d'Intentions**

### 🧠 **Qu'est-ce qu'une Intention ?**

L'intention = **ce que l'utilisateur veut vraiment accomplir**

| Intention | Exemples | Réponse Attendue |
|-----------|----------|------------------|
| `salutation` | "Bonjour", "Salut", "Hey" | Accueil chaleureux |
| `question_produit` | "Quel est le prix ?", "Caractéristiques ?" | Info technique |
| `probleme_technique` | "Ça marche pas", "Bug", "Erreur" | Support technique |
| `reclamation` | "Je veux un remboursement", "C'est nul" | Escalade service client |
| `compliment` | "Merci", "Parfait", "Génial" | Renforcement positif |
| `au_revoir` | "Bye", "À bientôt", "Ciao" | Clôture polie |

### 📊 **Préparation des Données d'Intention**

#### **2.1 Collecte et Annotation**
```python
# Structure des données d'entraînement
intentions_data = {
    'salutation': [
        "Bonjour",
        "Salut tout le monde",
        "Hey, comment ça va ?",
        "Coucou !",
        "Hello",
        # ... au moins 20-30 exemples par intention
    ],
    'question_produit': [
        "Quel est le prix de ce produit ?",
        "Quelles sont les caractéristiques techniques ?",
        "Est-ce que c'est compatible avec mon système ?",
        "Avez-vous ce modèle en stock ?",
        # ...
    ]
}
```

#### **2.2 Augmentation de Données**
Techniques pour enrichir votre dataset :
```python
def augment_intent_data(original_texts):
    augmented = []
    for text in original_texts:
        # Synonymes
        augmented.append(replace_with_synonyms(text))
        # Variations de forme
        augmented.append(add_typos(text))
        # Variations de longueur
        augmented.append(add_context(text))
    return augmented
```

### 🛠️ **Implémentation du Classificateur d'Intentions**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),  # Unigrams et bigrams
                max_features=5000,
                stop_words=self.get_french_stopwords()
            )),
            ('classifier', LogisticRegression(random_state=42))
        ])
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, texts, intentions):
        """Entraîne le classificateur d'intentions"""
        y_encoded = self.label_encoder.fit_transform(intentions)
        self.pipeline.fit(texts, y_encoded)
        self.is_trained = True
        print(f"✅ Modèle entraîné sur {len(texts)} exemples")
    
    def predict(self, text):
        """Prédit l'intention avec score de confiance"""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné d'abord!")
        
        # Prédiction
        pred_encoded = self.pipeline.predict([text])[0]
        intention = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        # Score de confiance
        probas = self.pipeline.predict_proba([text])[0]
        confidence = max(probas)
        
        return intention, confidence
    
    def save_model(self, filepath):
        """Sauvegarde le modèle entraîné"""
        joblib.dump({
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder
        }, filepath)
```

---

## 😊 **3. Génération de Réponses Contextuelles**

### 🧠 **Concept : Réponses Dynamiques**

Une réponse ne dépend pas que de l'intention, mais aussi :
- **Sentiment** de l'utilisateur (content vs énervé)
- **Contexte** de la conversation (première fois vs récurrent)
- **Moment** de la journée/semaine
- **Historique** des interactions

### 📋 **Matrice Intention × Sentiment**

| Intention | Sentiment Positif | Sentiment Neutre | Sentiment Négatif |
|-----------|------------------|------------------|------------------|
| `salutation` | "Hello ! 😊 Super de vous voir !" | "Bonjour ! Comment puis-je vous aider ?" | "Bonjour ! Je sens que ça ne va pas, dites-moi tout 🤗" |
| `question_produit` | "Avec plaisir ! Voici les infos..." | "Bien sûr, voici ce que vous cherchez..." | "Je comprends votre préoccupation, laissez-moi vous expliquer..." |
| `probleme_technique` | "Pas de souci, on va arranger ça ! 💪" | "Décrivez-moi le problème, je vais vous aider" | "Je comprends votre frustration 😔 On va résoudre ça ensemble" |
| `reclamation` | "Merci de nous faire part de votre retour !" | "Décrivez-moi la situation s'il vous plaît" | "Je suis vraiment désolé ! Voyons comment réparer ça 🙏" |

### 🛠️ **Implémentation du Générateur de Réponses**

```python
import random
from datetime import datetime

class ResponseGenerator:
    def __init__(self):
        self.response_templates = {
            ('salutation', 'positif'): [
                "Hello ! 😊 Ravi de vous voir ! Comment puis-je vous aider aujourd'hui ?",
                "Salut ! Vous avez l'air de bonne humeur, c'est contagieux ! 🌟",
                "Bonjour ! Super énergie ! Que puis-je faire pour vous ?"
            ],
            ('salutation', 'neutre'): [
                "Bonjour ! Comment puis-je vous aider ?",
                "Salut ! Dites-moi ce que vous cherchez !",
                "Hello ! Je suis là pour vous assister !"
            ],
            ('salutation', 'negatif'): [
                "Bonjour ! Je sens que quelque chose vous tracasse... Dites-moi tout ! 🤗",
                "Salut ! Mauvaise journée ? Je suis là pour arranger ça ! 💪",
                "Hello ! Vous semblez préoccupé, comment puis-je vous aider ?"
            ],
            # ... autres combinaisons
        }
        
        self.fallback_responses = [
            "Hmm, je ne suis pas sûr de comprendre... Pouvez-vous reformuler ?",
            "C'est intéressant ! Pouvez-vous m'en dire plus ?",
            "Je veux vous aider au mieux ! Pouvez-vous être plus précis ?"
        ]
    
    def generate(self, intent, sentiment, original_text, context=None):
        """Génère une réponse contextuelle"""
        
        # Clé pour la matrice intention × sentiment
        response_key = (intent, sentiment)
        
        if response_key in self.response_templates:
            # Sélection aléatoire dans les templates disponibles
            response = random.choice(self.response_templates[response_key])
            
            # Personnalisation contextuelle
            response = self._personalize_response(response, context)
            
        else:
            # Fallback si combinaison non prévue
            response = random.choice(self.fallback_responses)
        
        return response
    
    def _personalize_response(self, response, context):
        """Ajoute des éléments contextuels à la réponse"""
        if context:
            # Exemple : ajouter le prénom si disponible
            if 'user_name' in context:
                response = f"{context['user_name']}, {response.lower()}"
            
            # Adapter selon l'heure
            hour = datetime.now().hour
            if hour < 12 and "Bonjour" in response:
                response = response.replace("Bonjour", "Bon matin")
            elif hour > 18 and "Bonjour" in response:
                response = response.replace("Bonjour", "Bonsoir")
        
        return response
```

### 🎨 **Techniques Avancées de Génération**

#### **3.1 Variables Dynamiques**
```python
def inject_dynamic_content(template, user_input, context):
    """Injecte du contenu dynamique dans les templates"""
    
    # Variables disponibles
    variables = {
        '{user_name}': context.get('user_name', 'cher utilisateur'),
        '{current_time}': datetime.now().strftime("%H:%M"),
        '{day_part}': get_day_part(),
        '{product_mentioned}': extract_product_from_text(user_input)
    }
    
    # Remplacement des variables
    for var, value in variables.items():
        template = template.replace(var, str(value))
    
    return template
```

#### **3.2 Réponses Progressives**
```python
def get_progressive_response(intent, attempt_count):
    """Adapte la réponse selon le nombre de tentatives"""
    
    if attempt_count == 1:
        return "Je ne suis pas sûr de comprendre..."
    elif attempt_count == 2:
        return "Désolé, pouvez-vous reformuler différemment ?"
    else:
        return "Je vais vous transférer vers un humain qui pourra mieux vous aider !"
```

---

## ⚠️ **4. Gestion d'Erreurs et Cas Limites**

### 🧠 **Types d'Erreurs à Gérer**

#### **4.1 Erreurs de Compréhension**
- **Confiance faible** : < 50% de certitude sur l'intention
- **Intentions multiples** : "Bonjour, j'ai un problème avec ma commande"
- **Intentions contradictoires** : "Merci... mais c'est nul"

#### **4.2 Erreurs Techniques**
- **Texte vide** ou uniquement des espaces
- **Caractères spéciaux** non supportés
- **Langue non reconnue**
- **Spam** ou contenu inapproprié

#### **4.3 Erreurs Contextuelles**
- **Référence à conversation précédente** non disponible
- **Demande hors périmètre** du chatbot
- **Boucle conversationnelle** (utilisateur répète la même chose)

### 🛠️ **Implémentation de la Gestion d'Erreurs**

```python
class ErrorHandler:
    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.3
        }
        self.max_retries = 3
        self.inappropriate_words = self.load_inappropriate_words()
    
    def handle_low_confidence(self, intent, confidence, retry_count=0):
        """Gère les cas de faible confiance"""
        
        if confidence < self.confidence_thresholds['low']:
            if retry_count < self.max_retries:
                return {
                    'response': "Je n'ai pas bien saisi... Pouvez-vous reformuler ?",
                    'action': 'ask_clarification',
                    'retry_count': retry_count + 1
                }
            else:
                return {
                    'response': "Je vais vous mettre en relation avec un conseiller humain.",
                    'action': 'escalate_to_human',
                    'retry_count': 0
                }
        
        elif confidence < self.confidence_thresholds['medium']:
            return {
                'response': f"Vous voulez parler de '{intent}' ? (Oui/Non)",
                'action': 'confirm_intent',
                'suggested_intent': intent
            }
        
        else:
            return {
                'response': None,  # Procéder normalement
                'action': 'proceed',
                'confidence': 'sufficient'
            }
    
    def validate_input(self, user_input):
        """Valide l'entrée utilisateur"""
        
        # Vérifications basiques
        if not user_input or user_input.strip() == "":
            return False, "empty_input"
        
        if len(user_input) > 1000:
            return False, "too_long"
        
        # Détection de contenu inapproprié
        if self.contains_inappropriate_content(user_input):
            return False, "inappropriate_content"
        
        return True, "valid"
    
    def contains_inappropriate_content(self, text):
        """Détecte le contenu inapproprié"""
        text_lower = text.lower()
        return any(word in text_lower for word in self.inappropriate_words)
```

### 🔄 **Pattern de Fallback en Cascade**

```python
class FallbackManager:
    def __init__(self):
        self.fallback_chain = [
            self.try_intent_clarification,
            self.try_keyword_matching,
            self.try_similarity_search,
            self.try_generic_response,
            self.escalate_to_human
        ]
    
    def handle_failed_intent(self, user_input, context):
        """Essaie plusieurs stratégies de fallback"""
        
        for fallback_method in self.fallback_chain:
            result = fallback_method(user_input, context)
            if result['success']:
                return result
        
        # Si tout échoue
        return {
            'success': False,
            'response': "Je suis désolé, je ne peux pas vous aider avec ça.",
            'action': 'end_conversation'
        }
```

---

## 📋 **Exercices Pratiques**

### 📝 **Exercice 13 : Intentions Basiques** *(15 points)*

#### 🎯 **Objectif**
Créer un classificateur d'intentions robuste avec validation croisée.

#### 📋 **Énoncé**
1. **Définir 5 intentions métier** pour un chatbot de support e-commerce
2. **Créer un dataset** de 50 exemples par intention (250 au total)
3. **Entraîner un classificateur** avec validation croisée
4. **Évaluer les performances** avec métriques complètes
5. **Tester sur des phrases ambiguës** et analyser les erreurs

#### 🛠️ **Template de Code**
```python
# exercices/exercice_13_intentions_basiques.py

class IntentClassifierExercise:
    def __init__(self):
        self.intentions = {
            'salutation': [],
            'question_produit': [],
            'probleme_technique': [],
            'reclamation': [],
            'compliment': []
        }
    
    def create_dataset(self):
        """TODO: Créer le dataset d'intentions"""
        pass
    
    def train_classifier(self):
        """TODO: Entraîner le classificateur"""
        pass
    
    def evaluate_performance(self):
        """TODO: Évaluer avec validation croisée"""
        pass

# Tests à réussir
if __name__ == "__main__":
    classifier = IntentClassifierExercise()
    classifier.create_dataset()
    classifier.train_classifier()
    accuracy = classifier.evaluate_performance()
    
    assert accuracy > 0.85, "Accuracy doit être > 85%"
    print("✅ Exercice 13 réussi !")
```

#### ✅ **Critères de Validation**
- [ ] 5 intentions bien définies et distinctes
- [ ] 50 exemples variés par intention
- [ ] Accuracy > 85% en validation croisée
- [ ] Matrice de confusion analysée
- [ ] Gestion des cas ambigus documentée

---

### 📝 **Exercice 14 : Réponses Contextuelles** *(15 points)*

#### 🎯 **Objectif**
Implémenter un système de génération de réponses adapté au contexte émotionnel.

#### 📋 **Énoncé**
1. **Créer une matrice** intention × sentiment (5×3 = 15 combinaisons)
2. **Rédiger 3 templates** de réponse par combinaison
3. **Implémenter la personnalisation** avec variables dynamiques
4. **Tester la cohérence** des réponses générées
5. **Mesurer la satisfaction** avec un panel de testeurs

#### 🛠️ **Template de Code**
```python
# exercices/exercice_14_reponses_contextuelles.py

class ContextualResponseGenerator:
    def __init__(self):
        self.response_matrix = {}
        self.dynamic_variables = {}
    
    def build_response_matrix(self):
        """TODO: Construire la matrice intention × sentiment"""
        pass
    
    def generate_contextual_response(self, intent, sentiment, context):
        """TODO: Générer réponse avec personnalisation"""
        pass
    
    def test_response_consistency(self):
        """TODO: Tester la cohérence des réponses"""
        pass

# Tests à réussir
if __name__ == "__main__":
    generator = ContextualResponseGenerator()
    generator.build_response_matrix()
    
    # Test de cohérence
    response1 = generator.generate_contextual_response('salutation', 'positif', {})
    response2 = generator.generate_contextual_response('salutation', 'negatif', {})
    
    assert response1 != response2, "Réponses doivent différer selon sentiment"
    print("✅ Exercice 14 réussi !")
```

#### ✅ **Critères de Validation**
- [ ] Matrice 5×3 complète avec 3 templates par case
- [ ] Variables dynamiques fonctionnelles
- [ ] Réponses cohérentes avec le sentiment
- [ ] Système de fallback implémenté
- [ ] Test utilisateur avec score > 7/10

---

### 📝 **Exercice 15 : Chatbot Complet** *(15 points)*

#### 🎯 **Objectif**
Intégrer tous les composants en un chatbot fonctionnel avec interface web.

#### 📋 **Énoncé**
1. **Assembler le pipeline** complet (preprocessing → intent → sentiment → response)
2. **Créer une interface** Streamlit interactive
3. **Implémenter la gestion** d'historique de conversation
4. **Ajouter le logging** et analytics basiques
5. **Déployer** sur Streamlit Cloud ou Heroku

#### 🛠️ **Template de Code**
```python
# exercices/exercice_15_chatbot_complet.py

import streamlit as st
from datetime import datetime

class CompleteChatbot:
    def __init__(self):
        self.pipeline = self.build_pipeline()
        self.conversation_history = []
    
    def build_pipeline(self):
        """TODO: Assembler tous les composants"""
        pass
    
    def process_user_input(self, user_input):
        """TODO: Pipeline complet de traitement"""
        pass
    
    def create_streamlit_interface(self):
        """TODO: Interface web interactive"""
        pass
    
    def log_conversation(self, user_input, bot_response):
        """TODO: Logging et analytics"""
        pass

# Interface Streamlit
def main():
    st.title("🤖 Mon Chatbot NLP Intelligent")
    
    chatbot = CompleteChatbot()
    chatbot.create_streamlit_interface()

if __name__ == "__main__":
    main()
```

#### ✅ **Critères de Validation**
- [ ] Pipeline complet fonctionnel
- [ ] Interface web déployée et accessible
- [ ] Historique de conversation persistant
- [ ] Gestion d'erreurs robuste
- [ ] Logging des interactions
- [ ] Documentation utilisateur fournie

---

## 🎯 **Projet Final : Chatbot Support Client Complet**

### 🏆 **Objectif Global**
Créer un chatbot de support client professionnel qui peut être réellement déployé.

### 📋 **Cahier des Charges**

#### **Fonctionnalités Obligatoires**
- ✅ **Classification d'intentions** (6 intentions minimum)
- ✅ **Analyse de sentiment** intégrée
- ✅ **Réponses contextuelles** personnalisées
- ✅ **Interface web** professionnelle
- ✅ **Gestion d'erreurs** complète
- ✅ **Historique** de conversation
- ✅ **Logging** des interactions

#### **Fonctionnalités Bonus** *(pour aller plus loin)*
- 🌟 **Support multilingue** (français + anglais)
- 🌟 **Intégration API** externe (météo, actualités)
- 🌟 **Base de connaissances** interrogeable
- 🌟 **Analytics** avec graphiques
- 🌟 **Mode vocal** (speech-to-text)

### 🛠️ **Architecture Technique**

```
chatbot-support-client/
├── README.md
├── requirements.txt
├── app.py                    # Interface Streamlit principale
├── src/
│   ├── __init__.py
│   ├── preprocessor.py       # Nettoyage et tokenisation
│   ├── intent_classifier.py  # Classification d'intentions
│   ├── sentiment_analyzer.py # Analyse de sentiments
│   ├── response_generator.py # Génération de réponses
│   ├── conversation_manager.py # Gestion du contexte
│   └── error_handler.py      # Gestion d'erreurs
├── models/
│   ├── intent_model.pkl      # Modèle d'intentions entraîné
│   └── sentiment_model.pkl   # Modèle de sentiments entraîné
├── data/
│   ├── intentions_training.csv
│   ├── sentiment_training.csv
│   └── response_templates.json
├── logs/
│   └── conversations.log
├── tests/
│   ├── test_intent_classifier.py
│   ├── test_sentiment_analyzer.py
│   └── test_integration.py
└── deployment/
    ├── Dockerfile
    ├── heroku.yml
    └── streamlit_config.toml
```

### 📊 **Évaluation du Projet Final**

| Critère | Points | Description |
|---------|--------|-------------|
| **Fonctionnalité** | 40% | Pipeline complet qui fonctionne |
| **Interface** | 20% | UI/UX professionnelle et intuitive |
| **Code Quality** | 20% | Code propre, commenté, testé |
| **Documentation** | 10% | README complet, démo vidéo |
| **Innovation** | 10% | Fonctionnalités créatives ajoutées |

### 🎯 **Seuils de Réussite**
- **🥉 Bronze (60%)** : Fonctionnalités de base + interface simple
- **🥈 Argent (75%)** : + Gestion d'erreurs + documentation
- **🥇 Or (90%)** : + Tests + déploiement + fonctionnalités bonus

---

## 📚 **Ressources et Outils**

### 📦 **Dépendances Recommandées**
```python
# requirements.txt pour le module 4
spacy>=3.4.0
nltk>=3.7
scikit-learn>=1.1.0
pandas>=1.4.0
numpy>=1.21.0
streamlit>=1.15.0
plotly>=5.10.0
joblib>=1.2.0
python-dotenv>=0.19.0
requests>=2.28.0
```

### 🗃️ **Datasets Fournis**

#### **intentions_support_client.csv**
```csv
text,intent
"Bonjour, comment allez-vous ?",salutation
"J'ai un problème avec ma commande",probleme_technique
"Quel est le prix de ce produit ?",question_produit
"Je veux un remboursement",reclamation
"Merci beaucoup pour votre aide",compliment
"Au revoir et bonne journée",au_revoir
```

#### **conversations_exemples.json**
```json
{
  "conversations": [
    {
      "id": "conv_001",
      "messages": [
        {"role": "user", "text": "Bonjour"},
        {"role": "bot", "text": "Bonjour ! Comment puis-je vous aider ?"},
        {"role": "user", "text": "J'ai un problème avec ma commande"},
        {"role": "bot", "text": "Je comprends votre préoccupation. Pouvez-vous me donner votre numéro de commande ?"}
      ]
    }
  ]
}
```

### 🛠️ **Utilitaires Fournis**

#### **utils/chatbot_helpers.py**
```python
def load_conversation_history(user_id):
    """Charge l'historique d'un utilisateur"""
    try:
        with open(f"logs/user_{user_id}_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_conversation_turn(user_id, user_input, bot_response, metadata):
    """Sauvegarde un tour de conversation"""
    conversation_turn = {
        'timestamp': datetime.now().isoformat(),
        'user_input': user_input,
        'bot_response': bot_response,
        'metadata': metadata
    }
    
    history = load_conversation_history(user_id)
    history.append(conversation_turn)
    
    with open(f"logs/user_{user_id}_history.json", "w") as f:
        json.dump(history, f, indent=2)

def calculate_conversation_metrics(conversation_history):
    """Calcule des métriques sur une conversation"""
    if not conversation_history:
        return {}
    
    total_turns = len(conversation_history)
    avg_confidence = np.mean([turn.get('metadata', {}).get('confidence', 0) 
                             for turn in conversation_history])
    
    intent_distribution = {}
    for turn in conversation_history:
        intent = turn.get('metadata', {}).get('intent', 'unknown')
        intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
    
    return {
        'total_turns': total_turns,
        'avg_confidence': avg_confidence,
        'intent_distribution': intent_distribution,
        'conversation_length_minutes': calculate_duration(conversation_history)
    }

def detect_conversation_loops(conversation_history, window_size=3):
    """Détecte les boucles conversationnelles"""
    if len(conversation_history) < window_size * 2:
        return False
    
    recent_inputs = [turn['user_input'].lower().strip() 
                    for turn in conversation_history[-window_size:]]
    
    # Vérifier si l'utilisateur répète les mêmes choses
    unique_inputs = set(recent_inputs)
    if len(unique_inputs) <= 1:
        return True
    
    return False

def generate_conversation_summary(conversation_history):
    """Génère un résumé de la conversation"""
    if not conversation_history:
        return "Aucune conversation"
    
    metrics = calculate_conversation_metrics(conversation_history)
    main_intent = max(metrics['intent_distribution'].items(), 
                     key=lambda x: x[1])[0] if metrics['intent_distribution'] else 'unknown'
    
    summary = f"""
    📊 Résumé de Conversation:
    - Nombre de tours: {metrics['total_turns']}
    - Intention principale: {main_intent}
    - Confiance moyenne: {metrics['avg_confidence']:.1%}
    - Durée: {metrics['conversation_length_minutes']:.1f} minutes
    """
    
    return summary
```

#### **utils/evaluation_metrics.py**
```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_intent_classifier(y_true, y_pred, intent_labels):
    """Évaluation complète du classificateur d'intentions"""
    
    # Rapport de classification
    report = classification_report(y_true, y_pred, 
                                 target_names=intent_labels, 
                                 output_dict=True)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualisation
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=intent_labels, 
                yticklabels=intent_labels,
                cmap='Blues')
    plt.title('Matrice de Confusion - Classification d\'Intentions')
    plt.ylabel('Vraie Intention')
    plt.xlabel('Intention Prédite')
    plt.tight_layout()
    plt.savefig('confusion_matrix_intentions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return report, cm

def evaluate_sentiment_analyzer(y_true, y_pred):
    """Évaluation de l'analyseur de sentiments"""
    sentiment_labels = ['négatif', 'neutre', 'positif']
    return evaluate_intent_classifier(y_true, y_pred, sentiment_labels)

def calculate_response_quality_score(responses, human_ratings):
    """Calcule un score de qualité des réponses basé sur des évaluations humaines"""
    if len(responses) != len(human_ratings):
        raise ValueError("Le nombre de réponses et d'évaluations doit être identique")
    
    avg_rating = np.mean(human_ratings)
    response_lengths = [len(response.split()) for response in responses]
    avg_length = np.mean(response_lengths)
    
    # Score composite (exemple simple)
    quality_score = (avg_rating / 10) * 0.7 + (min(avg_length / 20, 1)) * 0.3
    
    return {
        'average_human_rating': avg_rating,
        'average_response_length': avg_length,
        'composite_quality_score': quality_score,
        'rating_distribution': {
            'excellent (9-10)': sum(1 for r in human_ratings if r >= 9),
            'good (7-8)': sum(1 for r in human_ratings if 7 <= r < 9),
            'average (5-6)': sum(1 for r in human_ratings if 5 <= r < 7),
            'poor (1-4)': sum(1 for r in human_ratings if r < 5)
        }
    }
```

---

## 🚀 **Guide de Déploiement**

### 🌐 **Déploiement sur Streamlit Cloud**

#### **1. Préparation du Repository**
```bash
# Structure pour déploiement
chatbot-nlp/
├── streamlit_app.py          # Point d'entrée principal
├── requirements.txt          # Dépendances
├── .streamlit/
│   └── config.toml          # Configuration Streamlit
├── src/                     # Code source
├── models/                  # Modèles pré-entraînés
└── README.md               # Documentation
```

#### **2. Configuration Streamlit**
```toml
# .streamlit/config.toml
[theme]
primaryColor = "#4285f4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#333333"

[server]
headless = true
enableCORS = false
port = 8501
```

#### **3. Script de Déploiement**
```python
# streamlit_app.py - Point d'entrée optimisé
import streamlit as st
import sys
import os

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot_main import CompleteChatbot

# Configuration de la page
st.set_page_config(
    page_title="Chatbot NLP Intelligent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache pour le chatbot (éviter de recharger à chaque interaction)
@st.cache_resource
def load_chatbot():
    return CompleteChatbot()

def main():
    st.title("🤖 Chatbot NLP Intelligent")
    st.markdown("*Créé avec les techniques de traitement du langage naturel*")
    
    # Initialisation du chatbot
    chatbot = load_chatbot()
    
    # Interface utilisateur
    chatbot.create_streamlit_interface()
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("📊 Statistiques")
        if 'conversation_history' in st.session_state:
            st.metric("Tours de conversation", 
                     len(st.session_state.conversation_history))
        
        st.header("🛠️ Techniques Utilisées")
        st.markdown("""
        - **Tokenisation** avec spaCy
        - **Classification d'intentions** (Logistic Regression)
        - **Analyse de sentiments** (Naive Bayes)
        - **Génération contextuelle** de réponses
        """)

if __name__ == "__main__":
    main()
```

### 🐳 **Déploiement avec Docker** *(Bonus)*

#### **Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install -r requirements.txt

# Téléchargement du modèle spaCy français
RUN python -m spacy download fr_core_news_sm

# Copie du code source
COPY . .

# Exposition du port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Commande de démarrage
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **docker-compose.yml**
```yaml
version: '3.8'

services:
  chatbot-nlp:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
```

---

## 🧪 **Tests et Validation**

### 🔬 **Tests Unitaires**

#### **tests/test_intent_classifier.py**
```python
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from intent_classifier import IntentClassifier

class TestIntentClassifier(unittest.TestCase):
    
    def setUp(self):
        self.classifier = IntentClassifier()
        # Données de test minimales
        self.test_texts = [
            "Bonjour comment allez-vous",
            "J'ai un problème avec ma commande",
            "Quel est le prix de ce produit",
            "Merci beaucoup pour votre aide"
        ]
        self.test_intents = [
            "salutation",
            "probleme_technique", 
            "question_produit",
            "compliment"
        ]
    
    def test_training(self):
        """Test de l'entraînement du modèle"""
        self.classifier.train(self.test_texts, self.test_intents)
        self.assertTrue(self.classifier.is_trained)
    
    def test_prediction(self):
        """Test de prédiction"""
        self.classifier.train(self.test_texts, self.test_intents)
        intent, confidence = self.classifier.predict("Salut !")
        
        self.assertIsInstance(intent, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_confidence_threshold(self):
        """Test du seuil de confiance"""
        self.classifier.train(self.test_texts, self.test_intents)
        intent, confidence = self.classifier.predict("texte complètement aléatoire xyz123")
        
        # Pour un texte sans rapport, la confiance devrait être faible
        self.assertLess(confidence, 0.8)

if __name__ == '__main__':
    unittest.main()
```

#### **tests/test_integration.py**
```python
import unittest
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chatbot_main import CompleteChatbot

class TestChatbotIntegration(unittest.TestCase):
    
    def setUp(self):
        self.chatbot = CompleteChatbot()
    
    def test_pipeline_complete(self):
        """Test du pipeline complet"""
        user_input = "Bonjour, j'ai un problème"
        response = self.chatbot.process_user_input(user_input)
        
        self.assertIsInstance(response, dict)
        self.assertIn('text', response)
        self.assertIn('intent', response)
        self.assertIn('sentiment', response)
        self.assertIn('confidence', response)
    
    def test_error_handling(self):
        """Test de la gestion d'erreurs"""
        # Test avec entrée vide
        response = self.chatbot.process_user_input("")
        self.assertIn('error', response)
        
        # Test avec texte très long
        long_text = "a" * 2000
        response = self.chatbot.process_user_input(long_text)
        self.assertIn('error', response)
    
    def test_conversation_context(self):
        """Test du contexte conversationnel"""
        # Premier message
        response1 = self.chatbot.process_user_input("Bonjour")
        
        # Deuxième message - doit tenir compte du contexte
        response2 = self.chatbot.process_user_input("J'ai un problème")
        
        # Vérifier que le contexte est maintenu
        self.assertTrue(len(self.chatbot.conversation_history) >= 2)

if __name__ == '__main__':
    unittest.main()
```

### 📊 **Tests de Performance**

#### **tests/test_performance.py**
```python
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chatbot_main import CompleteChatbot

class TestChatbotPerformance(unittest.TestCase):
    
    def setUp(self):
        self.chatbot = CompleteChatbot()
    
    def test_response_time(self):
        """Test du temps de réponse"""
        test_inputs = [
            "Bonjour",
            "J'ai un problème avec ma commande", 
            "Quel est le prix de ce produit ?",
            "Merci beaucoup"
        ]
        
        response_times = []
        
        for user_input in test_inputs:
            start_time = time.time()
            response = self.chatbot.process_user_input(user_input)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Le temps de réponse ne doit pas dépasser 2 secondes
            self.assertLess(response_time, 2.0, 
                           f"Temps de réponse trop long: {response_time:.2f}s")
        
        avg_response_time = sum(response_times) / len(response_times)
        print(f"Temps de réponse moyen: {avg_response_time:.3f}s")
    
    def test_concurrent_requests(self):
        """Test de la gestion de requêtes concurrentes"""
        def process_request(user_input):
            return self.chatbot.process_user_input(user_input)
        
        test_inputs = ["Bonjour"] * 10
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_request, test_inputs))
        end_time = time.time()
        
        # Vérifier que toutes les requêtes ont réussi
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('text', result)
        
        total_time = end_time - start_time
        print(f"Temps total pour 10 requêtes concurrentes: {total_time:.3f}s")

if __name__ == '__main__':
    unittest.main()
```

---

## 📈 **Analytics et Monitoring**

### 📊 **Dashboard d'Analytics** *(Bonus)*

```python
# src/analytics_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class ChatbotAnalytics:
    def __init__(self):
        self.conversation_logs = self.load_conversation_logs()
    
    def load_conversation_logs(self):
        """Charge les logs de conversation depuis les fichiers"""
        # Implémentation de chargement des logs
        pass
    
    def create_analytics_dashboard(self):
        """Crée un dashboard d'analytics avec Streamlit"""
        
        st.header("📊 Analytics du Chatbot")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_conversations = len(self.conversation_logs)
            st.metric("Conversations Totales", total_conversations)
        
        with col2:
            avg_confidence = self.calculate_average_confidence()
            st.metric("Confiance Moyenne", f"{avg_confidence:.1%}")
        
        with col3:
            resolution_rate = self.calculate_resolution_rate()
            st.metric("Taux de Résolution", f"{resolution_rate:.1%}")
        
        with col4:
            avg_conversation_length = self.calculate_avg_conversation_length()
            st.metric("Longueur Moyenne", f"{avg_conversation_length:.1f} tours")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des intentions
            intent_data = self.get_intent_distribution()
            fig_intent = px.pie(
                values=list(intent_data.values()),
                names=list(intent_data.keys()),
                title="Distribution des Intentions"
            )
            st.plotly_chart(fig_intent, use_container_width=True)
        
        with col2:
            # Évolution temporelle
            daily_data = self.get_daily_conversation_count()
            fig_timeline = px.line(
                x=daily_data.index,
                y=daily_data.values,
                title="Conversations par Jour"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Analyse des sentiments
        st.subheader("😊 Analyse des Sentiments")
        sentiment_data = self.get_sentiment_analysis()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_sentiment = px.bar(
                x=['Positif', 'Neutre', 'Négatif'],
                y=[sentiment_data['positif'], 
                   sentiment_data['neutre'], 
                   sentiment_data['negatif']],
                title="Distribution des Sentiments",
                color=['green', 'gray', 'red']
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Top des messages non résolus
            unresolved = self.get_unresolved_conversations()
            st.write("**Messages nécessitant attention:**")
            for msg in unresolved[:5]:
                st.write(f"- {msg['text'][:100]}...")
```

### 🔍 **Monitoring en Temps Réel**

```python
# src/monitoring.py
import logging
from datetime import datetime
import json

class ChatbotMonitoring:
    def __init__(self):
        self.setup_logging()
        self.alerts = []
    
    def setup_logging(self):
        """Configure le système de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/chatbot_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_conversation_turn(self, user_input, bot_response, metadata):
        """Log détaillé de chaque interaction"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'intent': metadata.get('intent'),
            'sentiment': metadata.get('sentiment'),
            'confidence': metadata.get('confidence'),
            'processing_time': metadata.get('processing_time'),
            'error': metadata.get('error')
        }
        
        self.logger.info(f"CONVERSATION_TURN: {json.dumps(log_entry)}")
        
        # Alertes automatiques
        self.check_for_alerts(log_entry)
    
    def check_for_alerts(self, log_entry):
        """Vérifie s'il faut déclencher des alertes"""
        
        # Alerte si confiance très faible
        if log_entry.get('confidence', 1.0) < 0.3:
            alert = {
                'type': 'LOW_CONFIDENCE',
                'timestamp': datetime.now().isoformat(),
                'message': f"Confiance très faible: {log_entry['confidence']:.2%}",
                'user_input': log_entry['user_input']
            }
            self.alerts.append(alert)
            self.logger.warning(f"ALERT: {json.dumps(alert)}")
        
        # Alerte si temps de traitement élevé
        if log_entry.get('processing_time', 0) > 5.0:
            alert = {
                'type': 'SLOW_RESPONSE',
                'timestamp': datetime.now().isoformat(),
                'message': f"Réponse lente: {log_entry['processing_time']:.2f}s"
            }
            self.alerts.append(alert)
            self.logger.warning(f"ALERT: {json.dumps(alert)}")
        
        # Alerte si erreur
        if log_entry.get('error'):
            alert = {
                'type': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'message': f"Erreur: {log_entry['error']}",
                'user_input': log_entry['user_input']
            }
            self.alerts.append(alert)
            self.logger.error(f"ALERT: {json.dumps(alert)}")
```

---

## 🎓 **Certification et Portfolio**

### 📋 **Checklist de Validation Finale**

#### **Compétences Techniques** *(70 points)*
- [ ] **Pipeline NLP Complet** (20 pts)
  - Preprocessing avec spaCy
  - Classification d'intentions robuste
  - Analyse de sentiments précise
  - Génération de réponses contextuelles

- [ ] **Gestion d'Erreurs** (15 pts)
  - Validation des entrées utilisateur
  - Fallback intelligent en cas d'échec
  - Gestion des cas limites

- [ ] **Interface Utilisateur** (15 pts)
  - Interface Streamlit professionnelle
  - Historique de conversation
  - Feedback utilisateur

- [ ] **Code Quality** (20 pts)
  - Code propre et commenté
  - Tests unitaires
  - Documentation technique

#### **Compétences Projet** *(30 points)*
- [ ] **Documentation** (10 pts)
  - README complet avec instructions
  - Vidéo de démonstration (3-5 min)
  - Explication des choix techniques

- [ ] **Déploiement** (10 pts)
  - Application déployée et accessible
  - Configuration de production
  - Monitoring basique

- [ ] **Innovation** (10 pts)
  - Fonctionnalités créatives ajoutées
  - Optimisations personnelles
  - Cas d'usage originaux

### 🏆 **Badges de Certification**

#### **🥉 NLP Practitioner** *(60-74 points)*
"Maîtrise les concepts fondamentaux du NLP et peut créer des applications basiques"

#### **🥈 NLP Developer** *(75-89 points)*
"Capable de développer des systèmes NLP robustes avec gestion d'erreurs et déploiement"

#### **🥇 NLP Expert** *(90-100 points)*
"Expert en NLP capable d'innover et d'optimiser des systèmes de production"

### 📄 **Template README Portfolio**

```markdown
# 🤖 Mon Chatbot NLP Intelligent

## 🎯 Description
Chatbot de support client utilisant les techniques modernes de NLP pour comprendre les intentions et émotions des utilisateurs.

## 🚀 Démo
🔗 **Application déployée**: [Lien Streamlit Cloud]
📹 **Vidéo de démonstration**: [Lien YouTube - 3 minutes]

## 🛠️ Techniques Utilisées
- **Preprocessing**: spaCy pour tokenisation et lemmatisation
- **Classification d'Intentions**: Logistic Regression avec TF-IDF
- **Analyse de Sentiments**: Naive Bayes avec features personnalisées
- **Génération de Réponses**: Templates contextuels avec variables dynamiques

## 📊 Performances
- ✅ **Accuracy Intentions**: 87%
- ✅ **F1-Score Sentiments**: 0.84
- ✅ **Temps de Réponse Moyen**: 0.8s
- ✅ **Satisfaction Utilisateur**: 8.2/10

## 🔧 Installation Locale
```bash
git clone [votre-repo]
cd chatbot-nlp
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
streamlit run app.py
```

## 📈 Fonctionnalités
- [x] Classification de 6 intentions métier
- [x] Analyse de sentiments en temps réel
- [x] Réponses contextuelles personnalisées
- [x] Gestion d'historique de conversation
- [x] Interface web responsive
- [x] Logging et analytics
- [x] Tests unitaires et d'intégration

## 🎨 Innovations Ajoutées
- Support des émojis dans l'analyse de sentiments
- Détection automatique des boucles conversationnelles
- Dashboard d'analytics en temps réel
- Mode debug pour développeurs

## 🧪 Tests
```bash
python -m pytest tests/ -v
```

## 📝 Leçons Apprises
- L'importance du preprocessing pour la qualité des prédictions
- La gestion du contexte conversationnel est cruciale
- Les fallbacks intelligents améliorent l'expérience utilisateur
- Le monitoring permet d'améliorer continuellement le système

## 🚀 Prochaines Étapes
- [ ] Support multilingue (anglais, espagnol)
- [ ] Intégration avec API externes
- [ ] Mode vocal avec speech-to-text
- [ ] Apprentissage actif à partir des conversations

## 👨‍💻 Auteur
**[Votre Nom]** - Étudiant NLP passionné
- LinkedIn: [votre-profil]
- GitHub: [votre-github]
- Email: [votre-email]
```

---

## 🔗 **Transition vers le Module LLM**

### 🌉 **Récapitulatif des Acquis**
Félicitations ! Vous maîtrisez maintenant :
- ✅ **Architecture NLP complète** de bout en bout
- ✅ **Classification supervisée** pour intentions et sentiments
- ✅ **Génération de réponses** basée sur des règles
- ✅ **Gestion de la conversation** et du contexte
- ✅ **Déploiement d'applications** NLP en production

### 🚀 **Préparation au Bloc LLM**
Dans le module suivant, vous découvrirez :
- 🧠 **Modèles de langage pré-entraînés** (GPT, BERT, T5)
- 🎯 **Fine-tuning** pour des tâches spécifiques
- 🎨 **Génération de texte** créative et contextuelle
- 🔧 **Prompt Engineering** avancé
- 🌟 **Création d'un mini-ChatGPT** personnel

**Votre chatbot actuel utilise des règles et templates... Imaginez maintenant qu'il puisse VRAIMENT comprendre et générer du texte comme un humain ! 🤯**

---

*Bravo ! Vous venez de terminer le module le plus complet du parcours NLP. Votre chatbot est maintenant prêt à impressionner le monde ! 🎉*