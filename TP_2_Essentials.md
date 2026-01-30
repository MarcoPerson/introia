# PARTIE 1 : SGD ET ADAM EN DÉTAILS

---

## 🎯 1. SGD (Stochastic Gradient Descent)

### **A. Gradient Descent Classique (Batch GD)**

**Principe :**
- Utilise **TOUS** les échantillons pour calculer le gradient
- Mise à jour **une fois par époque**

**Algorithme :**
```python
# Pour une époque
for epoch in range(epochs):
    # 1. Forward sur TOUT le dataset
    y_pred = model(X_all)  # X_all : (m, n) - tous les échantillons
    
    # 2. Loss sur TOUT le dataset
    loss = MSE(y_pred, Y_all)
    
    # 3. Gradient calculé sur TOUT le dataset
    gradient = (1/m) * X_all.T @ (y_pred - Y_all)
    
    # 4. UNE SEULE mise à jour
    theta = theta - learning_rate * gradient
```

**Dimensions :**
```
m = 10,000 échantillons
X_all : (10000, n)
y_pred : (10000, 1)
gradient : (n, 1)

→ Calcule le gradient en utilisant les 10,000 échantillons à la fois
```

**Problèmes :**
- ❌ Très lent si m est grand (millions de données)
- ❌ Nécessite beaucoup de mémoire (tout charger)
- ❌ Convergence lente (1 mise à jour par époque)
- ✅ Gradient précis et stable

---

### **B. Stochastic Gradient Descent (SGD)**

**Principe :**
- Utilise **UN SEUL** échantillon pour calculer le gradient
- Mise à jour **m fois par époque** (une par échantillon)

**Algorithme :**
```python
# Pour une époque
for epoch in range(epochs):
    # Mélanger les données
    indices = np.random.permutation(m)
    
    # Pour CHAQUE échantillon
    for i in indices:
        # 1. Forward sur UN échantillon
        x_i = X[i]  # (n,)
        y_i = Y[i]  # (1,)
        y_pred_i = model(x_i)
        
        # 2. Loss sur UN échantillon
        loss_i = (y_pred_i - y_i)**2
        
        # 3. Gradient calculé sur UN échantillon
        gradient = x_i.T * (y_pred_i - y_i)
        
        # 4. Mise à jour IMMÉDIATE
        theta = theta - learning_rate * gradient
```

**Dimensions :**
```
x_i : (n,)        - UN échantillon
y_i : (1,)        - UNE sortie
gradient : (n,)   - gradient pour cet échantillon

→ m mises à jour par époque
```

**Caractéristiques :**
- ✅ Très rapide (traite 1 échantillon à la fois)
- ✅ Peu de mémoire
- ✅ Peut échapper aux minima locaux (grâce au bruit)
- ❌ Gradient bruité (forte variance)
- ❌ Convergence instable (oscillations)

**Visualisation de la convergence :**
```
Batch GD :     ────────▼  (lisse, direct vers le minimum)

SGD :          ─╱─╲─╱─▼╲─╱─  (oscillant, bruité)
```

---

### **C. Mini-Batch SGD (Le plus utilisé)**

**Principe :**
- Utilise **un petit lot** (batch) d'échantillons
- Compromis entre Batch GD et SGD
- **C'est ce qu'on utilise dans le TP !**

**Algorithme :**
```python
batch_size = 32  # Typiquement 16, 32, 64, 128, 256

# Pour une époque
for epoch in range(epochs):
    # Mélanger les données
    indices = np.random.permutation(m)
    
    # Découper en mini-batches
    num_batches = m // batch_size
    
    for batch_idx in range(num_batches):
        # Extraire un mini-batch
        start = batch_idx * batch_size
        end = start + batch_size
        batch_indices = indices[start:end]
        
        X_batch = X[batch_indices]  # (batch_size, n)
        Y_batch = Y[batch_indices]  # (batch_size, 1)
        
        # 1. Forward sur le batch
        y_pred_batch = model(X_batch)
        
        # 2. Loss sur le batch
        loss = MSE(y_pred_batch, Y_batch)
        
        # 3. Gradient moyenné sur le batch
        gradient = (1/batch_size) * X_batch.T @ (y_pred_batch - Y_batch)
        
        # 4. Mise à jour
        theta = theta - learning_rate * gradient
```

**Dimensions :**
```
m = 10,000 échantillons
batch_size = 32

X_batch : (32, n)      - un mini-batch
Y_batch : (32, 1)
gradient : (n, 1)      - moyenné sur 32 échantillons

→ 10000/32 ≈ 312 mises à jour par époque
```

**Avantages :**
- ✅ Gradient plus stable que SGD pur (moyenné sur batch_size)
- ✅ Plus rapide que Batch GD (plusieurs mises à jour par époque)
- ✅ Exploite la parallélisation GPU (matrices)
- ✅ Bon compromis vitesse/stabilité

**Courbe de convergence :**
```
Loss
  |
  |  ╲
  |   ╲_
  |     ╲___
  |        ╲_____
  |             ╲_______  (oscillations modérées)
  +─────────────────────► Itérations
```

---

### **D. SGD avec Momentum**

**Problème du SGD :**
- Oscillations dans les directions perpendiculaires au minimum
- Progression lente dans la direction du minimum

**Solution : Momentum**

**Principe :**
- Accumule un "élan" dans la direction des gradients précédents
- Comme une boule qui roule et prend de la vitesse

**Algorithme :**
```python
# Initialisation
velocity = 0
beta = 0.9  # coefficient de momentum (typiquement 0.9)

# À chaque itération
for epoch in range(epochs):
    for X_batch, Y_batch in dataloader:
        # Calcul du gradient
        gradient = compute_gradient(X_batch, Y_batch)
        
        # Mise à jour de la vélocité (élan)
        velocity = beta * velocity + (1 - beta) * gradient
        
        # Mise à jour des paramètres
        theta = theta - learning_rate * velocity
```

**Explication mathématique :**
```
v_t = β·v_{t-1} + (1-β)·g_t

où :
- v_t : vélocité au temps t
- g_t : gradient au temps t
- β : facteur d'amortissement (0.9 = 90% de l'ancien élan conservé)

θ_t = θ_{t-1} - α·v_t
```

**Effet visuel :**
```
Sans momentum :  ─╱─╲─╱─╲─╱─  (zigzag)

Avec momentum : ───────▼───   (plus direct)
```

**Dimensions :**
```
gradient : (n, 1)
velocity : (n, 1)  - même shape que gradient
theta : (n, 1)
```

**Avantages :**
- ✅ Accélère dans les directions consistantes
- ✅ Amortit les oscillations
- ✅ Converge plus vite
- ✅ Peut franchir de petits plateaux

---

## 🚀 2. ADAM (Adaptive Moment Estimation)

**Adam = SGD + Momentum + Adaptation du learning rate**

C'est l'**optimiseur le plus utilisé** en deep learning !

### **A. Principe**

Adam combine **deux idées** :
1. **Momentum** : accumule l'élan des gradients
2. **RMSprop** : adapte le learning rate pour chaque paramètre

**Pourquoi c'est puissant ?**
- Certains paramètres ont besoin d'un grand learning rate
- D'autres ont besoin d'un petit learning rate
- Adam s'adapte automatiquement !

---

### **B. Algorithme Complet**

```python
# Hyperparamètres
learning_rate = 0.001  # α
beta1 = 0.9            # pour le momentum (first moment)
beta2 = 0.999          # pour RMSprop (second moment)
epsilon = 1e-8         # pour éviter division par zéro

# Initialisation
m = 0  # first moment (momentum)
v = 0  # second moment (variance)
t = 0  # compteur d'itérations

# À chaque itération
for epoch in range(epochs):
    for X_batch, Y_batch in dataloader:
        t += 1
        
        # 1. Calcul du gradient
        gradient = compute_gradient(X_batch, Y_batch)  # g_t
        
        # 2. Mise à jour du first moment (moyenne mobile du gradient)
        m = beta1 * m + (1 - beta1) * gradient
        
        # 3. Mise à jour du second moment (moyenne mobile du carré du gradient)
        v = beta2 * v + (1 - beta2) * gradient**2
        
        # 4. Correction du biais (bias correction)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # 5. Mise à jour des paramètres
        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
```

---

### **C. Décortiquons Chaque Étape**

#### **Étape 2 : First Moment (m)**

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t

C'est une moyenne mobile exponentielle des gradients
→ Équivalent au momentum
→ "Dans quelle direction dois-je aller ?"
```

**Dimensions :**
```
gradient : (n, 1)
m : (n, 1)  - même shape que les paramètres
```

**Exemple numérique :**
```python
beta1 = 0.9
m = 0
gradients = [1.0, 1.2, 0.8, 1.1]

Itération 1: m = 0.9*0 + 0.1*1.0 = 0.1
Itération 2: m = 0.9*0.1 + 0.1*1.2 = 0.21
Itération 3: m = 0.9*0.21 + 0.1*0.8 = 0.269
Itération 4: m = 0.9*0.269 + 0.1*1.1 = 0.352

→ m accumule une "mémoire" des gradients passés
```

---

#### **Étape 3 : Second Moment (v)**

```
v_t = β₂·v_{t-1} + (1-β₂)·g_t²

C'est une moyenne mobile des carrés des gradients
→ Mesure la "variance" des gradients
→ "À quelle vitesse dois-je aller ?"
```

**Dimensions :**
```
gradient**2 : (n, 1)  - carré élément par élément
v : (n, 1)
```

**Exemple numérique :**
```python
beta2 = 0.999
v = 0
gradients = [1.0, 1.2, 0.8, 1.1]

Itération 1: v = 0.999*0 + 0.001*1.0² = 0.001
Itération 2: v = 0.999*0.001 + 0.001*1.2² = 0.002439
Itération 3: v = 0.999*0.002439 + 0.001*0.8² = 0.003077
Itération 4: v = 0.999*0.003077 + 0.001*1.1² = 0.004285

→ v suit la variance des gradients
```

---

#### **Étape 4 : Bias Correction**

**Pourquoi ?**

Au début, m et v sont initialisés à 0, donc ils sont **biaisés** vers 0.

**Exemple :**
```python
# Premier gradient
g1 = 1.0
m1 = 0.9*0 + 0.1*1.0 = 0.1  # Devrait être proche de 1.0, pas 0.1 !
```

**Correction :**
```
m_hat = m / (1 - β₁^t)
v_hat = v / (1 - β₂^t)

Au début (t petit) : correction forte
Plus tard (t grand) : correction négligeable
```

**Exemple numérique :**
```python
beta1 = 0.9
t = 1: m_hat = m / (1 - 0.9^1) = m / 0.1 = 10·m  ← grosse correction
t = 2: m_hat = m / (1 - 0.9^2) = m / 0.19 = 5.26·m
t = 10: m_hat = m / (1 - 0.9^10) = m / 0.651 = 1.54·m
t = 100: m_hat = m / (1 - 0.9^100) ≈ m  ← correction négligeable
```

---

#### **Étape 5 : Mise à Jour Adaptive**

```
θ_t = θ_{t-1} - α · m_hat / (√v_hat + ε)
```

**Décomposons :**

```
Learning rate adaptatif = α / (√v_hat + ε)

- Si v_hat grand (gradient varie beaucoup) → petit pas
- Si v_hat petit (gradient stable) → grand pas
```

**Exemple numérique :**
```python
alpha = 0.001
epsilon = 1e-8

# Paramètre 1 : gradient stable
m_hat_1 = 0.5
v_hat_1 = 0.01
update_1 = 0.001 * 0.5 / (sqrt(0.01) + 1e-8)
         = 0.001 * 0.5 / 0.1
         = 0.005  ← grand pas

# Paramètre 2 : gradient très variable
m_hat_2 = 0.5
v_hat_2 = 4.0
update_2 = 0.001 * 0.5 / (sqrt(4.0) + 1e-8)
         = 0.001 * 0.5 / 2.0
         = 0.00025  ← petit pas

→ Adam adapte automatiquement la taille du pas !
```

---

### **D. Code Complet d'Adam**

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}  # first moments
        self.v = {}  # second moments
    
    def update(self, params, grads):
        """
        params : dict {nom: theta}
        grads : dict {nom: gradient}
        """
        self.t += 1
        
        for name in params:
            # Initialisation si première fois
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])
            
            # Récupérer gradient
            g = grads[name]
            
            # Mise à jour first moment
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            
            # Mise à jour second moment
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g**2
            
            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)
            
            # Mise à jour des paramètres
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Utilisation
optimizer = AdamOptimizer(learning_rate=0.001)

for epoch in range(epochs):
    for X_batch, Y_batch in dataloader:
        # Forward
        y_pred = model.forward(X_batch)
        
        # Backward
        grads = model.backward(X_batch, Y_batch)
        
        # Update avec Adam
        optimizer.update(model.params, grads)
```

---

### **E. Dimensions dans Adam**

```python
# Pour un réseau : input(10) → hidden(50) → output(1)

# Paramètres
W1 : (50, 10)
b1 : (50, 1)
W2 : (1, 50)
b2 : (1, 1)

# Gradients (même shape que paramètres)
dW1 : (50, 10)
db1 : (50, 1)
dW2 : (1, 50)
db2 : (1, 1)

# Adam stocke pour CHAQUE paramètre :
m['W1'] : (50, 10)  - first moment de W1
v['W1'] : (50, 10)  - second moment de W1
m['b1'] : (50, 1)   - first moment de b1
v['b1'] : (50, 1)   - second moment de b1
...

→ Adam double l'utilisation mémoire !
```

---

## 📊 3. COMPARAISON SGD vs ADAM

### **A. Tableau Comparatif**

| Aspect | SGD (+ Momentum) | Adam |
|--------|------------------|------|
| **Learning rate** | Fixe (ou scheduler manuel) | Adaptatif par paramètre |
| **Convergence** | Plus lente au début | Très rapide au début |
| **Mémoire** | m stocke seulement velocity | m + v (×2 mémoire) |
| **Hyperparamètres** | α, β (2) | α, β₁, β₂, ε (4) |
| **Stabilité** | Sensible au LR | Plus robuste |
| **Généralisation** | Souvent meilleure | Peut sur-ajuster |
| **Popularité** | Recherche | Production |

---

### **B. Visualisation de la Convergence**

```
Loss
  |
  | SGD sans momentum
  | ╲  ╱╲  ╱╲
  |  ╲╱  ╲╱  ╲___
  |            ╲___
  |
  | SGD avec momentum
  |  ╲
  |   ╲____
  |       ╲____
  |
  | Adam
  |  ╲____
  |      ╲_______
  +──────────────────► Itérations
```

**Adam converge plus vite mais peut osciller près du minimum**

---

### **C. Quand Utiliser Quoi ?**

**Utilisez SGD (+Momentum) si :**
- ✅ Vous voulez la meilleure généralisation possible
- ✅ Vous pouvez tuner finement le learning rate
- ✅ Vous avez du temps pour l'entraînement
- ✅ Dataset petit/moyen
- **Exemple :** Recherche, compétitions Kaggle

**Utilisez Adam si :**
- ✅ Vous voulez converger rapidement
- ✅ Vous n'avez pas le temps de tuner
- ✅ Dataset très grand
- ✅ Réseau très profond
- ✅ Prototypage rapide
- **Exemple :** Production, deadline serrée

---

### **D. Exemple Concret sur le TP**

```python
# ===== AVEC SGD =====
optimizer_sgd = torch.optim.SGD(
    model.parameters(), 
    lr=0.01,           # Doit être choisi avec soin !
    momentum=0.9,
    weight_decay=0.01
)

# Entraînement
for epoch in range(5000):  # Peut nécessiter beaucoup d'époques
    ...
    optimizer_sgd.step()

# Résultat : convergence lente mais stable
# Nécessite tuning de lr (0.1 trop grand, 0.001 trop petit, 0.01 OK)

# ===== AVEC ADAM =====
optimizer_adam = torch.optim.Adam(
    model.parameters(),
    lr=0.001,          # Valeur par défaut marche souvent
    betas=(0.9, 0.999),  # Valeurs par défaut
    weight_decay=0.01
)

# Entraînement
for epoch in range(2000):  # Converge plus vite
    ...
    optimizer_adam.step()

# Résultat : convergence rapide, moins de tuning nécessaire
```

**Comparaison des courbes :**
```python
import matplotlib.pyplot as plt

epochs_sgd = range(5000)
losses_sgd = [...]  # Décroissance progressive

epochs_adam = range(2000)
losses_adam = [...]  # Décroissance rapide puis plateau

plt.plot(epochs_sgd, losses_sgd, label='SGD')
plt.plot(epochs_adam, losses_adam, label='Adam')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.show()
```

---

# PARTIE 2 : RÉSEAUX PROFONDS ET FONCTIONS D'ACTIVATION

---

## 🏗️ 1. POURQUOI DES RÉSEAUX PROFONDS ?

### **A. Réseau Peu Profond vs Profond**

**Réseau peu profond (shallow) :**
```
Input → Hidden (large) → Output

Exemple : 100 → 1000 neurones → 10
```

**Réseau profond (deep) :**
```
Input → Hidden1 → Hidden2 → Hidden3 → Output

Exemple : 100 → 256 → 128 → 64 → 10
```

---

### **B. Théorème d'Approximation Universelle**

**Énoncé :**
> Un réseau avec UNE SEULE couche cachée (suffisamment grande) peut approximer n'importe quelle fonction continue.

**Alors pourquoi aller plus profond ?**

**Réponse : Efficacité !**

---

### **C. Exemple Concret : Reconnaissance d'Images**

**Tâche :** Classifier une image de chat vs chien

**Réseau peu profond :**
```
Image (28×28 pixels) → 10,000 neurones → Chat/Chien

Problèmes :
- Doit apprendre TOUS les patterns en une étape
- Chaque neurone voit l'image ENTIÈRE
- Pas de réutilisation des motifs
- Nécessite énormément de paramètres
```

**Réseau profond :**
```
Image → Couche 1 : détecte les bords
      → Couche 2 : détecte les formes (oreilles, yeux)
      → Couche 3 : détecte les parties (tête, pattes)
      → Couche 4 : détecte l'animal complet
      → Output : Chat/Chien

Avantages :
- Hiérarchie de représentations
- Réutilisation des features
- Moins de paramètres
- Meilleure généralisation
```

---

### **D. Analogie : Construire un Château**

**Approche peu profonde :**
```
Un seul ouvrier géant qui doit tout faire en une fois
→ Très difficile, inefficace
```

**Approche profonde :**
```
Couche 1 : Ouvriers qui posent les fondations
Couche 2 : Ouvriers qui montent les murs
Couche 3 : Ouvriers qui posent le toit
Couche 4 : Ouvriers qui décorent

→ Chaque étape spécialisée, efficace, réutilisable
```

---

### **E. Comparaison Nombre de Paramètres**

**Pour la même capacité d'approximation :**

```python
# Réseau peu profond
shallow = [100, 10000, 10]
params_shallow = 100*10000 + 10000*10 = 1,100,000 paramètres

# Réseau profond
deep = [100, 256, 128, 64, 10]
params_deep = 100*256 + 256*128 + 128*64 + 64*10
            = 25,600 + 32,768 + 8,192 + 640
            = 67,200 paramètres

Réduction : ×16 moins de paramètres !
```

**Le réseau profond est BEAUCOUP plus efficace**

---

### **F. Représentations Hiérarchiques**

**Dans un réseau profond, chaque couche apprend des concepts de plus en plus abstraits :**

```
RÉSEAU DE RECONNAISSANCE FACIALE

Input : Image 256×256

Couche 1 (basse) : Détecte les bords
│ ╱  │  ╲  ─  |  /

Couche 2 : Combine les bords en formes
○  □  △  ◇

Couche 3 : Détecte les parties du visage
👁  👃  👄  👂

Couche 4 : Assemble en visages
👤  👤  👤

Output : Identité de la personne
```

**Preuve expérimentale :**
On peut visualiser ce que chaque couche "voit" en utilisant des techniques comme :
- Activation Maximization
- Grad-CAM
- Feature Visualization

---

## 🎨 2. FONCTIONS D'ACTIVATION

### **A. Pourquoi les Activations sont Essentielles**

**Sans activation (tout linéaire) :**

```python
# Réseau à 3 couches
z1 = W1 @ x + b1
z2 = W2 @ z1 + b2
z3 = W3 @ z2 + b3

# Développons
z3 = W3 @ (W2 @ (W1 @ x + b1) + b2) + b3
   = W3 @ W2 @ W1 @ x + W3 @ W2 @ b1 + W3 @ b2 + b3
   = W_combined @ x + b_combined

→ C'est équivalent à UNE SEULE couche linéaire !
```

**Avec activation (non-linéaire) :**

```python
z1 = W1 @ x + b1
a1 = ReLU(z1)        # ← Non-linéarité !
z2 = W2 @ a1 + b2
a2 = ReLU(z2)        # ← Non-linéarité !
z3 = W3 @ a2 + b3

→ Impossible de simplifier en une couche
→ Peut apprendre des fonctions complexes
```

**RÈGLE D'OR : Sans activation, un réseau profond = 1 couche linéaire**

---

### **B. Catalogue des Fonctions d'Activation**

#### **1. ReLU (Rectified Linear Unit)** ⭐ **LA PLUS UTILISÉE**

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

**Graphique :**
```
    |     /
  2 |    /
  1 |   /
  0 |  /
-1  | /___________
    -2  -1  0  1  2
```

**Propriétés :**
- ✅ Très simple : `max(0, x)`
- ✅ Calcul ultra-rapide
- ✅ Pas de saturation pour x > 0
- ✅ Sparsité : environ 50% des neurones à 0
- ❌ "Dying ReLU" : neurones qui ne s'activent jamais

**Quand l'utiliser :**
- Couches cachées de TOUS les réseaux (par défaut)
- CNN, ResNet, Transformers

**Dimensions :**
```python
z : (n, m)  # n neurones, m échantillons
a = relu(z) : (n, m)  # élément par élément
```

**Exemple numérique :**
```python
z = np.array([[-2, -1, 0, 1, 2]])
a = relu(z)
# a = [[0, 0, 0, 1, 2]]

# 60% des valeurs sont devenues 0 → sparsité
```

---

#### **2. Leaky ReLU**

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)
```

**Graphique :**
```
    |     /
  2 |    /
  1 |   /
  0 |  /
-1  | ╱___________
    -2  -1  0  1  2
```

**Propriétés :**
- ✅ Résout le problème de dying ReLU
- ✅ Garde un petit gradient pour x < 0
- α typique = 0.01

**Quand l'utiliser :**
- Quand ReLU cause des dying neurons
- GAN (Generative Adversarial Networks)

**Exemple numérique :**
```python
alpha = 0.01
z = np.array([[-2, -1, 0, 1, 2]])
a = leaky_relu(z, alpha)
# a = [[-0.02, -0.01, 0, 1, 2]]
```

---

#### **3. Sigmoid**

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

**Graphique :**
```
  1 |        ───────
    |      /
0.5 |    /
    |  /
  0 |───────
   -5  -2  0  2  5
```

**Propriétés :**
- ✅ Sortie entre 0 et 1
- ✅ Interprétable comme probabilité
- ❌ Vanishing gradient pour |x| > 3
- ❌ Sortie non centrée en 0
- ❌ Calcul exponentiel coûteux

**Quand l'utiliser :**
- **Couche de sortie** pour classification binaire
- LSTM/GRU (gates)
- **JAMAIS dans les couches cachées** (sauf cas très spécifiques)

**Exemple numérique :**
```python
z = np.array([[-5, -2, 0, 2, 5]])
a = sigmoid(z)
# a = [[0.007, 0.119, 0.5, 0.881, 0.993]]

# Gradient pour x=-5
grad = sigmoid_derivative(-5)  
# grad ≈ 0.0066 → quasi 0 ! (vanishing gradient)
```

---

#### **4. Tanh (Tangente Hyperbolique)**

```python
def tanh(x):
    return np.tanh(x)
    # ou : (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

**Graphique :**
```
  1 |        ───────
    |      /
  0 |    /
    |  /
 -1 |───────
   -5  -2  0  2  5
```

**Propriétés :**
- ✅ Sortie entre -1 et 1
- ✅ Centrée en 0 (meilleure que sigmoid)
- ❌ Vanishing gradient pour |x| > 2
- ❌ Calcul exponentiel coûteux

**Quand l'utiliser :**
- LSTM/RNN (cellules)
- Quand on veut des sorties centrées

**Exemple numérique :**
```python
z = np.array([[-2, -1, 0, 1, 2]])
a = tanh(z)
# a = [[-0.964, -0.762, 0, 0.762, 0.964]]
```

---

#### **5. Softmax** (Couche de sortie seulement)

```python
def softmax(z):
    """
    z : (n_classes, m_samples)
    """
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # stabilité numérique
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
```

**Propriétés :**
- Transforme des scores en probabilités
- Σᵢ softmax(z)ᵢ = 1
- Chaque sortie ∈ [0, 1]

**Quand l'utiliser :**
- **Couche de sortie** pour classification multi-classes

**Exemple numérique :**
```python
# 3 classes, 2 échantillons
z = np.array([[2.0, 1.0],
              [1.0, 0.0],
              [0.1, 2.0]])  # (3, 2)

probs = softmax(z)
# [[0.659, 0.259],   # Probabilité classe 1
#  [0.242, 0.095],   # Probabilité classe 2
#  [0.099, 0.646]]   # Probabilité classe 3

# Vérification
print(np.sum(probs, axis=0))
# [1.0, 1.0] ✓

# Échantillon 1 : 65.9% classe 1, 24.2% classe 2, 9.9% classe 3
# Échantillon 2 : 25.9% classe 1, 9.5% classe 2, 64.6% classe 3
```

---

### **C. Problème du Vanishing Gradient**

**Qu'est-ce que c'est ?**

Dans un réseau profond, les gradients deviennent de plus en plus petits en remontant vers les premières couches.

**Pourquoi ?**

```
Gradient = ∂L/∂W1 = ∂L/∂a3 × ∂a3/∂z3 × ∂z3/∂a2 × ∂a2/∂z2 × ∂z2/∂a1 × ∂a1/∂z1 × ∂z1/∂W1
                                ↑          ↑          ↑
                          σ'(z3)     σ'(z2)     σ'(z1)

Si σ = sigmoid : σ'(z) ≤ 0.25 pour tout z

Donc : gradient ∝ 0.25 × 0.25 × 0.25 = 0.0156
```

**Avec un réseau de 10 couches :**
```
gradient ∝ 0.25^10 = 0.0000001 → quasi 0 !
```

**Conséquence :** Les premières couches n'apprennent presque pas.

**Solution : ReLU !**
```
ReLU'(z) = 1 si z > 0
         = 0 si z < 0

Pas de saturation pour z > 0 !
Gradients passent sans diminution.
```

---

### **D. Influence des Activations sur l'Apprentissage**

**Expérience : Réseau profond (10 couches) sur MNIST**

```python
# Avec Sigmoid
model_sigmoid = [784, 128, 128, 128, ..., 10]  # 10 couches
activations = sigmoid

# Entraînement
Epoch 10: Loss = 2.3, Accuracy = 10% (pas mieux que hasard)
Epoch 100: Loss = 2.3, Accuracy = 11%
→ N'apprend PAS (vanishing gradient)

# Avec ReLU
model_relu = [784, 128, 128, 128, ..., 10]  # 10 couches
activations = ReLU

# Entraînement
Epoch 10: Loss = 0.5, Accuracy = 85%
Epoch 100: Loss = 0.05, Accuracy = 98%
→ Apprend TRÈS BIEN
```

---

### **E. Tableau Récapitulatif**

| Activation | Où l'utiliser | Avantages | Inconvénients |
|------------|---------------|-----------|---------------|
| **ReLU** | Couches cachées (défaut) | Rapide, pas de vanishing | Dying neurons |
| **Leaky ReLU** | Couches cachées (si dying ReLU) | Résout dying ReLU | Un peu plus lent |
| **Sigmoid** | Output (classification binaire) | Probabilité | Vanishing gradient |
| **Tanh** | RNN/LSTM | Centré en 0 | Vanishing gradient |
| **Softmax** | Output (multi-classes) | Distribution de prob | - |
| **Linear** | Output (régression) | Pas de limite | Pas de non-linéarité |

---

## 🎯 3. RÉSUMÉ FINAL

### **SGD vs Adam**

```python
# Pour le TP (régression simple)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Fonctionne bien, nécessite tuning de lr

# Pour un gros projet (CNN, Transformer)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Converge plus vite, moins de tuning
```

### **Profondeur**

- Réseau peu profond : moins de paramètres mais moins efficace
- Réseau profond : représentations hiérarchiques, plus efficace
- **Règle** : Commencer avec 2-3 couches, augmenter si nécessaire

### **Activations**

- **Couches cachées** : ReLU (défaut) ou Leaky ReLU
- **Output régression** : Linear (pas d'activation)
- **Output classification binaire** : Sigmoid
- **Output classification multi-classes** : Softmax

**Voilà ! Tout est clair maintenant ?** 🚀
