# Rappel Théorique : Régression Polynomiale, MSE et Descente de Gradient

---

## 1. RÉGRESSION POLYNOMIALE

### 1.1 Principe de base

**Objectif :** Trouver une fonction polynomiale qui approxime au mieux la relation entre une variable d'entrée **x** et une sortie **y**.

### 1.2 Modèle mathématique

**Polynôme d'ordre m :**

```
y(x) = θ₀ + θ₁x + θ₂x² + θ₃x³ + ... + θₘxᵐ
```

Où :
- **θ = [θ₀, θ₁, θ₂, ..., θₘ]ᵀ** : vecteur de paramètres (coefficients)
- **m** : ordre du polynôme (degré)
- **θ₀** : terme constant (biais/intercept)
- **θ₁, θ₂, ...** : coefficients des puissances de x

### 1.3 Formulation matricielle

Pour **N échantillons** {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)} :

**Vecteur des observations :**
```
Y = [y₁, y₂, ..., yₙ]ᵀ    (dimension N×1)
```

**Matrice de design X :**
```
X = [1  x₁  x₁²  ...  x₁ᵐ]     (dimension N×(m+1))
    [1  x₂  x₂²  ...  x₂ᵐ]
    [⋮   ⋮   ⋮   ⋮    ⋮  ]
    [1  xₙ  xₙ²  ...  xₙᵐ]
```

**Prédictions :**
```
Ŷ = Xθ
```

### 1.4 Exemples concrets

**Ordre 1 (régression linéaire) :**
```
y(x) = θ₀ + θ₁x
→ Droite
```

**Ordre 2 (parabolique) :**
```
y(x) = θ₀ + θ₁x + θ₂x²
→ Parabole
```

**Ordre 3 (cubique) :**
```
y(x) = θ₀ + θ₁x + θ₂x² + θ₃x³
→ Courbe avec 1 point d'inflexion
```

### 1.5 Illustration visuelle

```
Ordre 1:  ___/          (sous-ajusté pour une sinusoïde)
         
Ordre 3:   ∿∿           (bon compromis)

Ordre 9:  ∿∿∿∿∿∿∿∿     (sur-ajusté, oscillations)
```

---

## 2. MSE (Mean Squared Error)

### 2.1 Définition

La **MSE** (Erreur Quadratique Moyenne) est la fonction de coût la plus utilisée en régression.

**Formule :**
```
MSE(θ) = 1/N Σᵢ₌₁ᴺ (yᵢ - ŷᵢ)²
       = 1/N Σᵢ₌₁ᴺ (yᵢ - y(xᵢ))²
```

Où :
- **yᵢ** : valeur réelle
- **ŷᵢ = y(xᵢ)** : prédiction du modèle
- **N** : nombre d'échantillons

### 2.2 Forme matricielle

```
MSE(θ) = 1/N ||Y - Xθ||²
       = 1/N (Y - Xθ)ᵀ(Y - Xθ)
```

### 2.3 Pourquoi l'élévation au carré ?

**Avantages :**
1. ✅ **Toujours positive** : pas d'annulation entre erreurs positives/négatives
2. ✅ **Pénalise fortement les grandes erreurs** : (erreur)²
3. ✅ **Différentiable partout** : calcul de gradient facile
4. ✅ **Convexe** (pour régression linéaire) : un seul minimum global
5. ✅ **Interprétation statistique** : estimateur du maximum de vraisemblance sous hypothèse gaussienne

**Illustration :**
```
Erreur: -2  -1   0   1   2
MSE:     4   1   0   1   4
→ Pénalise symétriquement et quadratiquement
```

### 2.4 Variantes de la MSE

**RMSE (Root MSE) :**
```
RMSE = √MSE
→ Même unité que y (interprétation plus facile)
```

**MSE avec régularisation L2 (Ridge) :**
```
MSE_ridge(θ) = 1/N Σᵢ₌₁ᴺ (yᵢ - ŷᵢ)² + λ Σⱼ₌₀ᵐ θⱼ²
```
- **λ > 0** : coefficient de régularisation
- Pénalise les paramètres θ de grande amplitude
- Évite le sur-apprentissage

### 2.5 Exemple numérique

```python
# Données
y_true = [1.0, 2.0, 3.0]
y_pred = [1.1, 2.3, 2.8]

# Calcul MSE
erreurs = [(1.0-1.1)², (2.0-2.3)², (3.0-2.8)²]
        = [0.01, 0.09, 0.04]
MSE = (0.01 + 0.09 + 0.04) / 3 = 0.047
```

---

## 3. DESCENTE DE GRADIENT (Gradient Descent)

### 3.1 Principe fondamental

**Idée :** Minimiser itérativement la fonction de coût en se déplaçant dans la direction opposée au gradient.

**Analogie :** Descendre une montagne dans le brouillard
- On ne voit que localement (gradient)
- On fait des petits pas vers le bas
- On s'arrête quand on ne peut plus descendre

### 3.2 L'algorithme

**Initialisation :**
```
θ⁽⁰⁾ = valeurs aléatoires (ou zéros)
t = 0  (itération)
```

**Mise à jour itérative :**
```
θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ - α × ∇MSE(θ⁽ᵗ⁾)
```

Où :
- **α** : learning rate (taux d'apprentissage)
- **∇MSE(θ)** : gradient de la MSE par rapport à θ
- **t** : numéro d'itération

**Arrêt :**
- Après un nombre fixe d'époques
- Quand le gradient devient très petit
- Quand la loss ne diminue plus

### 3.3 Calcul du gradient

**Pour la MSE :**
```
MSE(θ) = 1/N ||Y - Xθ||²
```

**Gradient par rapport à θ :**
```
∇MSE(θ) = -2/N Xᵀ(Y - Xθ)
        = -2/N Xᵀe    où e = Y - Xθ (erreurs)
```

**Pour chaque paramètre θⱼ :**
```
∂MSE/∂θⱼ = -2/N Σᵢ₌₁ᴺ (yᵢ - ŷᵢ) × xᵢⱼ
```

### 3.4 Mise à jour détaillée

**Formule complète :**
```
θⱼ⁽ᵗ⁺¹⁾ = θⱼ⁽ᵗ⁾ + α × (2/N) × Σᵢ₌₁ᴺ (yᵢ - ŷᵢ) × xᵢⱼ
```

**Interprétation :**
- Si **erreur positive** (sous-estimation) → augmenter θⱼ
- Si **erreur négative** (sur-estimation) → diminuer θⱼ
- Proportionnel à la feature xᵢⱼ

### 3.5 Le learning rate (α)

**Rôle crucial :**

```
α trop petit:  θ ---•---•---•---•---•→  (convergence lente)

α optimal:     θ -----•-----•-----→    (convergence rapide)

α trop grand:  θ •---------•          (divergence)
                   \       /
                    \     /
                     \   /
                      \ /
```

**Valeurs typiques :**
- 10⁻¹ à 10⁻⁶ selon le problème
- Dans le TP : **α = 10⁻³** est un bon départ

### 3.6 Variantes de la descente de gradient

**1. Batch Gradient Descent (BGD) :**
```
Utilise TOUS les échantillons à chaque itération
∇MSE = moyenne sur N échantillons
+ Convergence stable
- Lent pour grands datasets
```

**2. Stochastic Gradient Descent (SGD) :**
```
Utilise UN SEUL échantillon aléatoire à chaque itération
∇MSE ≈ gradient sur 1 exemple
+ Très rapide
+ Peut échapper minima locaux (bruit)
- Convergence bruitée
```

**3. Mini-Batch Gradient Descent :**
```
Utilise un PETIT LOT (ex: 32, 64, 128) d'échantillons
∇MSE = moyenne sur batch_size échantillons
+ Compromis vitesse/stabilité
+ Exploite parallélisation GPU
→ STANDARD en deep learning
```

### 3.7 Illustration de la convergence

```
MSE
 |
 |  •
 |   \
 |    •\
 |      \•
 |        \•
 |          •---•---•---  (convergence)
 +-------------------------> Itérations
```

**Avec régularisation L2 :**
```python
# Gradient avec weight decay
θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ - α × (∇MSE(θ⁽ᵗ⁾) + λθ⁽ᵗ⁾)
       = (1 - αλ)θ⁽ᵗ⁾ - α∇MSE(θ⁽ᵗ⁾)
```
→ Décroissance des poids à chaque itération

### 3.8 Pseudo-code complet

```python
# Initialisation
θ = np.random.randn(m+1, 1)  # Paramètres aléatoires
α = 1e-3                      # Learning rate
epochs = 5000                 # Nombre d'itérations

# Boucle d'apprentissage
for epoch in range(epochs):
    # Forward pass (prédiction)
    y_pred = X @ θ
    
    # Calcul de l'erreur
    erreur = y_pred - Y
    
    # Calcul du gradient
    gradient = (2/N) * X.T @ erreur
    
    # Mise à jour des paramètres
    θ = θ - α * gradient
    
    # (Optionnel) Calcul de la MSE
    mse = np.mean(erreur**2)
    
    # (Optionnel) Régularisation L2
    if weight_decay > 0:
        θ = θ - α * weight_decay * θ
```

---

## 4. COMPARAISON : Moindres Carrés vs Descente de Gradient

### 4.1 Tableau comparatif

| Aspect | Moindres Carrés | Descente Gradient |
|--------|-----------------|-------------------|
| **Solution** | θ* = (XᵀX + λI)⁻¹XᵀY | Itérative |
| **Calcul** | 1 opération | Milliers d'itérations |
| **Complexité** | O(m²N + m³) | O(Nm × epochs) |
| **Convergence** | Immédiate | Progressive |
| **Optimum** | Global exact | Approché |
| **Hyperparamètres** | λ seulement | α, epochs, batch_size, λ |
| **Scalabilité** | Limitée (mémoire) | Excellente (mini-batch) |

### 4.2 Quand utiliser quoi ?

**Moindres Carrés si :**
- Dataset petit/moyen (N < 100,000)
- Features peu nombreuses (m < 1,000)
- Besoin de solution exacte rapide
- Modèle linéaire en θ

**Descente de Gradient si :**
- Dataset très grand (millions)
- Beaucoup de features
- Modèle non-linéaire (deep learning)
- Apprentissage en ligne

---

## 5. APPLICATION AU TP

### 5.1 Le problème

**Données :**
```python
# Vraie fonction (inconnue en pratique)
y = sin(2πx) + bruit_gaussien

# N=10 échantillons d'apprentissage
# N=10 échantillons de validation
```

**Objectif :**
Trouver le polynôme qui approxime au mieux cette sinusoïde.

### 5.2 Pipeline complet

**Étape 1 : Génération des features**
```python
X = [1, x, x², x³, ..., xᵐ]  # Matrice de design
```

**Étape 2 : Choix de la méthode**

*Option A - Moindres Carrés :*
```python
θ* = np.linalg.inv(X.T @ X + λ*I) @ X.T @ Y
y_pred = X @ θ*
```

*Option B - Descente de Gradient :*
```python
model = NeuralNetwork(input_features=m)
optimizer = torch.optim.SGD(lr=1e-3, weight_decay=λ)
# Boucle d'entraînement sur epochs
```

**Étape 3 : Évaluation**
```python
mse_train = np.mean((Y_train - y_pred_train)**2)
mse_valid = np.mean((Y_valid - y_pred_valid)**2)
```

**Étape 4 : Choix du modèle**
```
Tester m = 1, 2, 3, ..., 9
Choisir m qui minimise mse_valid
```

### 5.3 Ce que vous allez observer

**Avec m qui augmente :**
1. **m = 1** : Droite, mauvais fit (biais élevé)
2. **m = 3-4** : Bonne approximation de la sinusoïde ✅
3. **m = 9** : Oscillations extrêmes (variance élevée)

**Avec λ (régularisation) :**
- λ = 0 : Sur-apprentissage pour m élevé
- λ = 0.01 : Courbes plus lisses
- λ = 0.1 : Sous-apprentissage (trop de régularisation)

**Avec le learning rate (descente gradient) :**
- α = 10⁻² : Divergence possible
- α = 10⁻³ : Convergence stable
- α = 10⁻⁵ : Convergence très lente

---

## 6. FORMULES CLÉS À RETENIR

### Régression polynomiale
```
ŷ = θ₀ + θ₁x + θ₂x² + ... + θₘxᵐ
Ŷ = Xθ
```

### MSE
```
MSE = (1/N) Σ(yᵢ - ŷᵢ)²
MSE_ridge = MSE + λΣθⱼ²
```

### Moindres carrés
```
θ* = (XᵀX + λI)⁻¹XᵀY
```

### Descente de gradient
```
θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ - α∇MSE(θ⁽ᵗ⁾)
∇MSE = (2/N)Xᵀ(Xθ - Y)
```

---

## 7. POINTS CLÉS POUR LE TP

✅ **Comprendre** que régression polynomiale = régression linéaire dans l'espace des features [1, x, x², ...]

✅ **Observer** le compromis biais-variance en faisant varier m

✅ **Expérimenter** l'effet de la régularisation λ

✅ **Comparer** moindres carrés (exact, rapide) vs gradient descent (itératif, flexible)

✅ **Maîtriser** l'influence du learning rate sur la convergence

✅ **Visualiser** systématiquement les courbes et les erreurs

---

**Questions ?** 🤔
