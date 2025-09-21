# Plan Détaillé - Formations SQL

## 📚 Module 0.3 : SQL, le Langage des Bases de Données (Tronc Commun)

### **Objectif pédagogique** 
Acquérir les compétences fondamentales en SQL pour interroger et manipuler des bases de données relationnelles.

---

### **Leçon 1 : Introduction aux Bases de Données et SQL**
**Durée estimée :** 45 min
- Comprendre ce qu'est une base de données relationnelle
- Découvrir le rôle de SQL dans l'écosystème data
- Installation et configuration de l'environnement de travail (SQLite/PostgreSQL)
- Interface et premiers pas avec un SGBD
- **Quiz :** Concepts fondamentaux des BDD

### **Leçon 2 : Anatomie d'une Base de Données**
**Durée estimée :** 40 min
- Structure des tables, lignes et colonnes
- Types de données (INT, VARCHAR, DATE, BOOLEAN, etc.)
- Clés primaires et étrangères
- Relations entre les tables
- **Quiz :** Structure et types de données

### **Leçon 3 : Première Requête SELECT**
**Durée estimée :** 50 min
- Syntaxe de base de SELECT
- Sélectionner toutes les colonnes (*)
- Sélectionner des colonnes spécifiques
- Premiers exemples pratiques sur une table simple
- **Quiz :** Requêtes SELECT de base

### **Leçon 4 : Filtrer les Données avec WHERE**
**Durée estimée :** 55 min
- Clause WHERE et ses opérateurs
- Opérateurs de comparaison (=, >, <, >=, <=, <>)
- Opérateurs logiques (AND, OR, NOT)
- Opérateur IN et NOT IN
- Gestion des valeurs NULL (IS NULL, IS NOT NULL)
- **Quiz :** Filtrage avec WHERE

### **Leçon 5 : Tri et Limitation des Résultats**
**Durée estimée :** 40 min
- ORDER BY pour trier les résultats (ASC, DESC)
- Tri sur plusieurs colonnes
- LIMIT pour limiter le nombre de résultats
- Cas pratiques de pagination
- **Quiz :** Tri et limitation

### **Leçon 6 : Fonctions d'Agrégation**
**Durée estimée :** 60 min
- Introduction aux fonctions d'agrégation
- COUNT, SUM, AVG, MIN, MAX
- Compter les valeurs non-nulles vs COUNT(*)
- Utilisation avec et sans GROUP BY
- **Quiz :** Fonctions d'agrégation

### **Leçon 7 : Regroupement avec GROUP BY**
**Durée estimée :** 65 min
- Principe du regroupement avec GROUP BY
- Combinaison GROUP BY + fonctions d'agrégation
- Règles importantes : colonnes dans SELECT et GROUP BY
- Ordre d'exécution des clauses SQL
- **Quiz :** GROUP BY et agrégation

### **Leçon 8 : Filtrer les Groupes avec HAVING**
**Durée estimée :** 45 min
- Différence entre WHERE et HAVING
- Utilisation de HAVING pour filtrer les groupes
- Combinaison WHERE + GROUP BY + HAVING
- Cas d'usage typiques
- **Quiz :** HAVING vs WHERE

### **Leçon 9 : Les Jointures - Partie 1 (INNER JOIN)**
**Durée estimée :** 70 min
- Comprendre le concept de jointure
- INNER JOIN : syntaxe et fonctionnement
- Jointures sur clés primaires/étrangères
- Alias de tables pour simplifier l'écriture
- Exemples pratiques avec 2 puis 3 tables
- **Quiz :** INNER JOIN

### **Leçon 10 : Les Jointures - Partie 2 (LEFT, RIGHT, FULL OUTER)**
**Durée estimée :** 75 min
- LEFT JOIN : récupérer toutes les lignes de la table de gauche
- RIGHT JOIN : récupérer toutes les lignes de la table de droite
- FULL OUTER JOIN : récupérer toutes les lignes des deux tables
- Gestion des valeurs NULL dans les jointures
- Comparaison visuelle des différents types de jointures
- **Quiz :** Tous types de jointures

### **Leçon 11 : Sous-Requêtes (Subqueries)**
**Durée estimée :** 80 min
- Introduction aux sous-requêtes
- Sous-requêtes dans WHERE (avec IN, EXISTS, ANY, ALL)
- Sous-requêtes scalaires dans SELECT
- Sous-requêtes corrélées vs non-corrélées
- Exemples pratiques et cas d'usage
- **Quiz :** Sous-requêtes

### **Leçon 12 : Introduction aux CTE (Common Table Expressions)**
**Durée estimée :** 60 min
- Qu'est-ce qu'une CTE et pourquoi l'utiliser
- Syntaxe WITH ... AS
- CTE simples vs sous-requêtes
- CTE multiples dans une même requête
- Avantages en lisibilité et réutilisabilité
- **Quiz :** CTE de base

### **Leçon 13 : Manipulation de Chaînes et Dates**
**Durée estimée :** 55 min
- Fonctions de chaînes (CONCAT, SUBSTRING, UPPER, LOWER, TRIM)
- Fonctions de dates (DATEPART, DATE_ADD, DATEDIFF)
- Formatage des dates
- Cas pratiques d'analyse temporelle
- **Quiz :** Fonctions chaînes et dates

### **Leçon 14 : Assignment Final - Module 0.3**
**Durée estimée :** 120 min
- Projet pratique intégrant tous les concepts
- Base de données d'une entreprise fictive (clients, commandes, produits)
- Série de requêtes progressives à résoudre
- Auto-évaluation et correction détaillée

---

## 🚀 Module DA1 : SQL pour l'Analyse Avancée (Parcours Data Analyst)

### **Objectif pédagogique**
Maîtriser les techniques SQL avancées pour l'analyse de données et l'optimisation des performances.

---

### **Leçon 1 : CTE Récursives et Hiérarchies**
**Durée estimée :** 70 min
- Comprendre la récursion en SQL
- Syntaxe des CTE récursives
- Cas d'usage : hiérarchies organisationnelles, chemins de navigation
- Limiter la récursion et éviter les boucles infinies
- **Quiz :** CTE récursives

### **Leçon 2 : Fonctions Fenêtres - Introduction**
**Durée estimée :** 85 min
- Concept des fonctions fenêtres (Window Functions)
- Structure : fonction OVER (PARTITION BY ... ORDER BY ...)
- ROW_NUMBER(), RANK(), DENSE_RANK()
- Différences et cas d'usage de chaque fonction
- **Quiz :** Fonctions de rang

### **Leçon 3 : Fonctions Fenêtres - Agrégation Mobile**
**Durée estimée :** 80 min
- Fonctions d'agrégation avec OVER
- Clauses ROWS et RANGE
- Calculs de moyennes mobiles, cumuls
- FIRST_VALUE, LAST_VALUE, LAG, LEAD
- **Quiz :** Agrégation mobile et décalages

### **Leçon 4 : Fonctions Fenêtres - Partitionnement Avancé**
**Durée estimée :** 75 min
- Partitionnement complexe avec plusieurs colonnes
- Comparaisons inter-groupes
- Calculs de pourcentages et percentiles
- NTILE pour créer des quartiles/déciles
- **Quiz :** Partitionnement avancé

### **Leçon 5 : Analyse Temporelle et Cohortage**
**Durée estimée :** 90 min
- Techniques d'analyse de séries temporelles en SQL
- Calculs period-over-period (MoM, YoY)
- Construction de tables de cohortes
- Analyse de rétention client
- **Quiz :** Analyse temporelle

### **Leçon 6 : Pivotement et Dépivotement de Données**
**Durée estimée :** 70 min
- PIVOT et UNPIVOT (selon le SGBD)
- Techniques de pivotement avec CASE WHEN
- Transformation de données pour l'analyse
- Création de tableaux croisés dynamiques
- **Quiz :** Pivotement de données

### **Leçon 7 : Requêtes Avancées et Patterns Complexes**
**Durée estimée :** 85 min
- Requêtes avec plusieurs niveaux de CTE
- Patterns d'analyse complexes (Top N par groupe, Running totals)
- Techniques de déduplication avancées
- Résolution de problèmes analytiques complexes
- **Quiz :** Patterns complexes

### **Leçon 8 : Comprendre les Plans d'Exécution**
**Durée estimée :** 80 min
- Introduction aux plans d'exécution
- Lecture et interprétation des plans
- Identifier les goulots d'étranglement
- Coût des opérations (Scan, Seek, Join, Sort)
- **Quiz :** Plans d'exécution

### **Leçon 9 : Optimisation avec les Index**
**Durée estimée :** 90 min
- Comprendre les index clustered et non-clustered
- Impact des index sur les performances SELECT
- Index composites et ordre des colonnes
- Stratégies d'indexation pour l'analytique
- Surveillance de l'utilisation des index
- **Quiz :** Stratégies d'indexation

### **Leçon 10 : Optimisation des Requêtes Complexes**
**Durée estimée :** 85 min
- Techniques de réécriture de requêtes
- Éviter les anti-patterns de performance
- Optimisation des jointures multiples
- Quand utiliser les hints (avec parcimonie)
- **Quiz :** Optimisation de requêtes

### **Leçon 11 : Statistiques et Monitoring de Performance**
**Durée estimée :** 75 min
- Statistiques des tables et colonnes
- Outils de monitoring des performances
- Identification des requêtes problématiques
- Maintenance des statistiques
- **Quiz :** Monitoring et statistiques

### **Leçon 12 : SQL pour le Big Data et le Cloud**
**Durée estimée :** 70 min
- Spécificités du SQL sur les plateformes Big Data
- Optimisations pour Snowflake, BigQuery, Redshift
- Techniques de partitioning et clustering
- Considérations de coût sur le cloud
- **Quiz :** SQL Cloud et Big Data

### **Leçon 13 : Assignment Final - Module DA1**
**Durée estimée :** 180 min
- Projet d'analyse complexe sur un jeu de données volumineux
- Optimisation d'un ensemble de requêtes lentes
- Création de rapports analytiques avec métriques avancées
- Présentation des résultats et justification des choix techniques

---

## 📋 Ressources Transversales

### **Pour chaque leçon :**
- **Vidéo explicative** (15-25 min selon la complexité)
- **Markdown détaillé** avec exemples pratiques
- **Scripts SQL** téléchargeables
- **Jeu de données** pour les exercices
- **Ressources annexes** (articles, documentation officielle)

### **Outils et Environnement :**
- **Base de données d'exercice** : Schéma e-commerce complet
- **Plateforme recommandée** : PostgreSQL ou SQL Server
- **Alternative cloud** : BigQuery ou Snowflake (selon disponibilité)
- **Outil de visualisation** : Integration avec des outils BI pour certains exercices

### **Évaluation :**
- **Quizzes** : 10-15 questions par leçon (QCM + questions ouvertes)
- **Assignments** : Projets pratiques avec datasets réels
- **Auto-correction** : Scripts de validation automatique pour certains exercices