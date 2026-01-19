# Plan Détaillé pour Paper ICDAR 2026

## **Titre proposé**

*"A Comprehensive Benchmark of State-Space Models vs. Transformers for Historical Newspaper OCR: Line and Paragraph-Level Evaluation on the BnL Dataset"*

---

## **Structure du Paper**

### **1. Abstract** (150-200 mots)

**Éléments clés à inclure :**
- Contexte : OCR de presse historique, défi sous-exploré à l'échelle paragraphe
- Gap : absence de benchmark systématique comparant SSM (Mamba) et Transformers sur documents historiques
- Contribution : benchmark rigoureux à deux granularités (ligne/paragraphe) avec 6 architectures
- Résultats : aperçu des findings principaux (sans spoiler les chiffres exacts)
- Impact : recommandations pratiques pour la numérisation patrimoniale

---

### **2. Introduction** (1-1.5 pages)

**§1 - Contexte et motivation**
- Importance de la numérisation du patrimoine documentaire (bibliothèques nationales, archives)
- Spécificités de la presse historique : dégradations, polices anciennes
- Enjeux pour la BnL et institutions similaires (BnF)

**§2 - État des approches actuelles**
- Dominance des Transformers (DAN, TrOCR, etc.) pour l'OCR de documents
- Émergence des State-Space Models (Mamba) comme alternative efficiente
- Dichotomie AR vs non-AR peu explorée systématiquement

**§3 - Gap identifié**
- Absence de benchmark unifié comparant ces paradigmes sur documents historiques
- Manque de reproductibilité (datasets, protocoles variables)

**§4 - Contributions**
1. Premier benchmark systématique SSM vs Transformers pour l'OCR de presse historique
2. Évaluation à deux granularités : ligne et paragraphe
3. Introduction de trois variantes Mamba (AR, CTC, CE) permettant une analyse fine des paradigmes de décodage
4. Nouveau protocole d'évaluation sur le corpus BnL avec splits qui seront publics
5. Analyse approfondie : efficience computationnelle, robustesse aux dégradations, comportement selon la longueur

---

### **3. Related Work** (1-1.5 pages)

**§3.1 - OCR pour documents historiques**
- Évolution : LSTM-CTC → Attention-based → Transformers
- Travaux spécifiques presse historique (Transkribus, etc.)
- Challenges : ground-truth bruité, variabilité typographique

**§3.2 - Architectures pour la reconnaissance de texte**
- **CTC-based** : CRNN, VAN — avantages et limites
- **AR Transformers** : DAN, TrOCR — modélisation séquentielle riche mais coût quadratique
- **Tokenisation** : caractère vs sous-mots (BPE) — trade-offs

**§3.3 - State-Space Models pour la vision**
- Mamba et S4 : complexité linéaire, mémoire longue
- Applications vision : Vim, VMamba
- **Gap** : quasi-absence en reconnaissance de texte manuscrit/imprimé

**§3.4 - Reconnaissance niveau paragraphe**
- Travaux existants (DAN, SPAN, VAN, Daniel)
- Défis spécifiques : attention longue portée, segmentation implicite

---

### **4. Methodology** (2-2.5 pages)

**§4.1 - Formulation du problème**
- Input : image de ligne ou paragraphe
- Output : séquence de caractères/tokens
- Formalisation mathématique des trois paradigmes :
  - AR avec CE : $P(y|x) = \prod_t P(y_t|y_{<t}, x)$
  - Non-AR avec CTC : alignement marginalisé
  - Non-AR avec CE : prédiction parallèle (NAR)

**§4.2 - Architectures évaluées**

*§4.2.1 - Encodeur visuel (commun)*
- Architecture CNN utilisée
- Extraction de features : dimensions, résolution

*§4.2.2 - Variantes Mamba (nos contributions)*

| Modèle | Décodage | Loss | Spécificités |
|--------|----------|------|--------------|
| Mamba-AR | Autorégressif | Cross-Entropy | SSM bidirectionnel + décodeur AR |
| Mamba-CTC | Non-AR | CTC | SSM + projection linéaire |
| Mamba-CE | Non-AR | Cross-Entropy | SSM + prédiction parallèle |

- Détails architecturaux : nombre de couches, dimensions, mécanisme de sélection

*§4.2.3 - Baselines*
- **DAN** : Transformer AR niveau caractère — description succincte
- **VAN** : Architecture CTC état-de-l'art — description
- **Daniel** : Transformer AR avec tokenisation BPE — description

**§4.3 - Protocole d'entraînement**
- Hyperparamètres (tableau unifié pour équité)
- Data augmentation
- Critères d'arrêt, sélection de modèle

**§4.4 - Métriques d'évaluation**
- **Qualité** : CER, WER, (NED pour niveau paragraphe)
- **Efficience** : paramètres, FLOPs, latence (ms/image), throughput (images/s)
- **Robustesse** : analyse par niveau de dégradation, longueur de séquence

---

### **5. Experimental Setup** (1-1.5 pages)

**§5.1 - Dataset BnL**

*§5.1.1 - Description du corpus*
- Provenance : Bibliothèque nationale du Luxembourg
- Période couverte, langues (luxembourgeois, français, allemand)
- Caractéristiques : qualité de numérisation, types de dégradations

*§5.1.2 - Statistiques*

| Split | Lignes | Paragraphes | Caractères (moy.) | Période |
|-------|--------|-------------|-------------------|---------|
| Train | X | Y | Z | ... |
| Val | X | Y | Z | ... |
| Test | X | Y | Z | ... |

*§5.1.3 - Preprocessing*
- Binarisation, normalisation
- Extraction ligne/paragraphe (GT ou détection ?)
- Gestion des caractères spéciaux

**§5.2 - Configuration expérimentale**
- Hardware (GPUs utilisés)
- Framework (PyTorch version)
- Seeds pour reproductibilité

**§5.3 - Protocole de comparaison équitable**
- Même encodeur visuel pour tous ? Ou encodeurs natifs ?
- Budget computationnel équivalent ?
- Nombre de runs, intervalles de confiance

---

### **6. Results and Analysis** (2.5-3 pages)

**§6.1 - Résultats niveau ligne**

*§6.1.1 - Performance globale (Tableau principal)*

| Modèle | Type | CER (%) ↓ | WER (%) ↓ | Params (M) | Latence (ms) |
|--------|------|-----------|-----------|------------|--------------|
| VAN | CTC | | | | |
| DAN | AR-Char | | | | |
| Daniel | AR-Token | | | | |
| Mamba-CTC | CTC | | | | |
| Mamba-CE | NAR | | | | |
| Mamba-AR | AR | | | | |

*§6.1.2 - Analyse comparative*
- SSM vs Transformer : quel paradigme gagne ?
- AR vs Non-AR : impact sur la qualité
- Caractères vs Tokens : pertinence pour documents historiques

**§6.2 - Résultats niveau paragraphe**

*§6.2.1 - Performance globale*

| Modèle | CER (%) | WER (%) | Params | Latence |
|--------|---------|---------|--------|---------|
| VAN | | | | |
| DAN | | | | |
| Daniel | | | | |
| Mamba-AR | | | | |

*§6.2.2 - Scaling ligne → paragraphe*
- Dégradation relative des performances
- Quel modèle scale le mieux ?

**§6.3 - Analyses approfondies**

*§6.3.1 - Efficience computationnelle*
- Trade-off qualité/vitesse (graphe Pareto)
- Scaling avec la longueur de séquence (figure : latence vs nb caractères)

*§6.3.2 - Robustesse aux dégradations*
- Analyse qualitative sur exemples difficiles

*§6.3.3 - Analyse des erreurs*
- Types d'erreurs fréquents par modèle
- Confusion matrices (si pertinent)
- Exemples visuels annotés

*§6.3.4 - Comportement selon la longueur*
- Bucketing par longueur de ground-truth
- Courbes CER = f(longueur)

**§6.4 - Résumé des findings**
- Bullet points des insights clés
- Réponse aux research questions implicites

---

### **7. Discussion** (0.75-1 page)

**§7.1 - Interprétation des résultats**
- Pourquoi Mamba fonctionne (ou non) sur ce task ?
- Rôle de la mémoire longue des SSM pour les paragraphes
- Impact du paradigme de décodage

**§7.2 - Recommandations pratiques**
- Quel modèle pour quel use-case ?
  - Production haute-vitesse → ...
  - Qualité maximale → ...
  - Contraintes mémoire → ...

**§7.3 - Limitations**
- Mono-dataset (généralisation ?)
- Langues spécifiques
- Ground-truth potentiellement bruité

**§7.4 - Travaux futurs**
- Extension multi-datasets
- Focus sur le Scaling Mamba pour OCR/HTR
- Fine-tuning sur domaines spécifiques

---

### **8. Conclusion** (0.5 page)

- Rappel des contributions
- Résultats clés en une phrase
- Impact pour la communauté DIA/OCR
- Appel à la reproductibilité (code/données disponibles)
