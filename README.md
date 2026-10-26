# Hand Gesture Recognition

## Présentation

Ce projet consiste à reconnaître des gestes de la main en utilisant l'intelligence artificielle. Les gestes sont détectés à partir d'une webcam grâce à MediaPipe Hands puis classifiés par un réseau de neurones artificiel (ANN) développé avec TensorFlow/Keras.

Le système reconnaît les gestes suivants :

- thumb_up
- open_hand
- fist
- peace
- ok

## Objectifs du projet

- Créer un petit dataset de gestes de la main.
- Extraire les landmarks de la main avec MediaPipe.
- Entraîner un modèle ANN pour reconnaître les gestes.
- Développer une interface utilisateur avec Streamlit.
- Réaliser une détection en temps réel à l'aide d'une webcam.

## Structure du projet

- data/dataset/ : images des gestes
- data/gestures.csv : données extraites des landmarks
- models/gesture_model.h5 : modèle entraîné
- models/encoder.pkl : encodeur des classes
- scripts/capture_images.py : capture des images
- scripts/extract_landmarks.py : extraction des landmarks
- scripts/train_model.py : entraînement du modèle
- scripts/real_time_detection.py : détection en temps réel
- scripts/app.py : interface Streamlit
- requirements.txt : bibliothèques nécessaires

## Technologies utilisées

- Python
- OpenCV
- MediaPipe Hands
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit

## Création du dataset

Le dataset a été créé manuellement avec une webcam.

- Nombre de gestes : 5
- Nombre d'images par geste : 100
- Nombre total d'images : 500

Après la capture des images, MediaPipe est utilisé pour extraire les 21 landmarks de la main. Chaque landmark possède trois coordonnées (x, y, z), ce qui donne un total de 63 caractéristiques pour chaque échantillon.

## Installation

Créer un environnement virtuel puis installer les dépendances :

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

### 1. Capture des images

```powershell
python scripts/capture_images.py
```

Entrer le nom du geste puis appuyer sur :

- s : enregistrer une image
- q : quitter

### 2. Extraction des landmarks

```powershell
python scripts/extract_landmarks.py
```

Cette étape génère le fichier :

```text
data/gestures.csv
```

### 3. Entraînement du modèle

```powershell
python scripts/train_model.py
```

Les fichiers suivants seront créés :

```text
models/gesture_model.h5
models/encoder.pkl
```
python train_model.py
### 4. Détection en temps réel

```powershell
python scripts/real_time_detection.py
```
python real_time_detection.py
### 5. Lancer l'interface Streamlit

```powershell
streamlit run scripts/app.py
```

## Résultats

Le modèle est capable de reconnaître les cinq gestes en temps réel à partir de la webcam. Les prédictions sont affichées directement sur l'écran.

## Difficultés rencontrées

- Variations de l'éclairage.
- Différences dans la position de la main.
- Certaines erreurs de prédiction lorsque la main n'est pas bien visible.

## Améliorations possibles

- Ajouter de nouveaux gestes.
- Augmenter la taille du dataset.
- Améliorer la précision du modèle.
- Supporter la reconnaissance de deux mains.

## Réalisé par

Taoufik Mammad

Projet réalisé dans le cadre d'un mini-projet de reconnaissance des gestes de la main utilisant l'intelligence artificielle.