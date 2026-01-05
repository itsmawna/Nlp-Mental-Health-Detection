# Détection de la Santé Mentale à partir des Réseaux Sociaux-NLP

Ce projet académique vise à exploiter les techniques de **Traitement Automatique du Langage Naturel (NLP)** afin de détecter automatiquement différents états de santé mentale à partir de publications issues des réseaux sociaux.

Il est réalisé dans le cadre de la filière **Big Data & Intelligence Artificielle (BDIA)**  
**ENSA Tétouan – Année universitaire 2025–2026**

**Avertissement** : Ce travail est strictement académique et expérimental.  
Il ne constitue en aucun cas un outil de diagnostic médical.

---

## Objectifs du projet

- Analyser des données textuelles issues des réseaux sociaux (principalement Reddit)
- Détecter automatiquement plusieurs états de santé mentale (classification multi-classes)
- Comparer des approches classiques et avancées en NLP :
  - **TF-IDF + SVM linéaire calibré**
  - **BERT**
  - **RoBERTa**
- Évaluer les performances avec des métriques adaptées aux données déséquilibrées
- Déployer une application interactive pour comparer les prédictions des modèles

---

## États de santé mentale étudiés

Le problème est formulé comme une tâche de **classification multi-classes** couvrant 14 états :

- Depression  
- Suicidal  
- ADHD  
- Bipolar disorder  
- Normal  
- OCD  
- PTSD  
- Anxiety  
- Stress  
- Personality disorder  
- Aspergers  
- Schizophrenia  
- Addiction  
- Alcoholism  

---

## Modèles implémentés

### 🔹 TF-IDF + SVM linéaire (Baseline)
- Vectorisation TF-IDF (unigrammes + bigrammes)
- Vocabulaire limité à 50 000 termes
- Pondération automatique des classes
- Calibration des probabilités (Platt Scaling)
- Modèle rapide, interprétable et robuste

### 🔹 BERT (Fine-tuning supervisé)
- Tokenisation WordPiece
- Représentations contextuelles bidirectionnelles
- Pondération des classes dans la fonction de perte
- Dropout, gradient clipping et early stopping
- Très bonne détection des classes minoritaires

### 🔹 RoBERTa
- Suppression de la tâche NSP
- Masquage dynamique
- Pré-entraînement optimisé
- Fine-tuning supervisé avec régularisation
- Performances globales très élevées et stables

---

## Résultats expérimentaux des modèles

Les modèles ont été évalués à l’aide de métriques adaptées aux jeux de données déséquilibrés :  
**Accuracy**, **F1-score Macro** et **F1-score Weighted**.

### 🔢 Résultats globaux

| Modèle                   | Accuracy | F1-score (Macro) | F1-score (Weighted) |
|--------------------------|----------|------------------|---------------------|
| TF-IDF + SVM (baseline) | 0.7647   | 0.7889           | 0.7650              |
| BERT                    | 0.8176   | 0.8429           | 0.8185              |
| RoBERTa                 | 0.8100   | 0.8400           | 0.8100              |

### 🔍 Analyse

- Les modèles **Transformers (BERT et RoBERTa)** surpassent largement la baseline TF-IDF + SVM  
- **BERT obtient les meilleures performances globales**, notamment en F1-score Macro  
- **RoBERTa reste très compétitif**, avec des performances proches et une grande stabilité  
- Le **F1-score Macro** confirme une meilleure prise en compte des classes minoritaires, ce qui est crucial en santé mentale  

---
## Reproduction des résultats 

Cette section décrit les étapes nécessaires pour reproduire les résultats expérimentaux présentés dans ce projet.

### Prérequis

- Python **3.9+**
- Environnement virtuel recommandé
- Système d’exploitation : Windows / Linux / macOS
- GPU recommandé pour l’entraînement de BERT et RoBERTa (optionnel mais conseillé)

---

### Installation de l’environnement

- Cloner le dépôt :
```bash
git clone https://github.com/itsmawna/Nlp-Mental-Health-Detection.git
cd NLP-Mental-Health-Detection
```
- Créer et activer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```
- Installer les dépendances :
```bash
pip install -r requirements.txt
```
### Préparation des données

Les jeux de données ne sont pas inclus dans le dépôt pour des raisons de **taille** et d’**éthique**.

Les notebooks suivants permettent de reproduire l’ensemble de la préparation des données :

- `notebooks/dataextraction_from_hugging_face.ipynb`
- `notebooks/preprocessing.ipynb`
- `notebooks/preprocessing final.ipynb`

Les données prétraitées doivent être placées dans le dossier `data/`.

---

### Entraînement des modèles

Chaque modèle peut être reproduit à l’aide de son notebook dédié :

- **TF-IDF + SVM**
  - `notebooks/tf-idf-linear-svm-calibrated.ipynb`

- **BERT**
  - `notebooks/nlp-bert.ipynb`

- **RoBERTa**
  - `notebooks/roberta-mental-health.ipynb`

Les modèles entraînés sont générés **localement** et sauvegardés dans le dossier `models/`  
(ce dossier est ignoré par Git en raison de la taille des fichiers).

---

### Évaluation des performances

Les notebooks incluent :
- Les métriques globales : **Accuracy**, **F1-score Macro**, **F1-score Weighted**
- Les rapports de classification détaillés par classe
- Les matrices de confusion
- La sélection du meilleur modèle selon le **F1-score Macro**

---

### Lancement de l’application

Une fois les modèles entraînés, l’application Streamlit peut être lancée avec la commande suivante :

```bash
streamlit run app/streamlit_compare.py
```
L’interface permet :
- La saisie de textes libres
- La comparaison des prédictions des modèles
- L’affichage des probabilités associées
- Une politique de décision prudente

### Reproductibilité

- Les splits des données sont réalisés de manière stratifiée
- Les graines aléatoires sont fixées afin d’assurer la reproductibilité
- Les hyperparamètres finaux sont documentés dans les notebooks
- Les performances peuvent varier légèrement selon le matériel et la configuration logicielle.
---


## Considérations éthiques

- Les résultats produits ne doivent **jamais être interprétés comme un diagnostic médical**
- Le projet vise uniquement l’analyse, la recherche et la démonstration académique
- Une attention particulière est portée à la gestion des biais et à la protection de la vie privée

---

## Contexte académique

<p align="center">
  <strong>Projet réalisé dans le cadre du module NLP / Intelligence Artificielle</strong><br>
  ENSA Tétouan – Filière Big Data & Intelligence Artificielle
</p>

