# 🐱🐶 CNN Cats vs Dogs — From Scratch & Transfer Learning

Comparaison d'un **CNN entraîné from scratch** et d'un modèle en **transfert learning (ResNet-18)** sur la classification binaire chats/chiens.

---

## Objectif

| | Expérience A | Expérience B |
|---|---|---|
| Architecture | CNN from scratch (3 blocs Conv) | ResNet-18 pré-entraîné |
| Régularisation | Dropout + Batch Normalization | Dropout + Batch Normalization |
| Optimiseurs testés | SGD, Adam | SGD, Adam |
| Scheduler | StepLR | CosineAnnealingLR |

---

## Environnement

### Prérequis

- Python 3.8+
- GPU recommandé (Google Colab avec GPU activé)

### Installation

```bash
pip install -r requirements.txt
```

Ou avec conda :

```bash
conda env create -f environment.yml
conda activate catdog-cnn
```

---

## Organisation des données

Les données ne sont **pas incluses dans ce dépôt**. Voici comment les obtenir :

```bash
wget https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip
unzip Cat_Dog_data.zip
```

Placez ensuite le dossier dans votre Google Drive à l'emplacement suivant :

```
MyDrive/
└── cnn_cat_dog_image_classification/
    ├── Cat_Dog_data/
    │   ├── train/
    │   │   ├── cat/
    │   │   └── dog/
    │   └── test/
    │       ├── cat/
    │       └── dog/
    └── TP_CNN_CatsVsDogs.ipynb
```

> Le notebook monte automatiquement Google Drive et pointe vers ce chemin. Adaptez `BASE_DIR` dans la cellule de configuration si votre structure est différente.

---

## Structure du dépôt

```
cnn-catsdogs-<NomPrenom>/
├── TP_CNN_CatsVsDogs.ipynb   # Notebook principal
├── requirements.txt           # Dépendances Python
├── environment.yml            # Environnement conda (alternative)
├── .gitignore
├── README.md
└── LICENSE
```

---

## Lancer l'entraînement

Ouvrez `TP_CNN_CatsVsDogs.ipynb` dans Google Colab avec **GPU activé** (Exécution → Modifier le type d'exécution → GPU).

### Hyperparamètres principaux

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `IMG_SIZE` | 64 | Taille des images en entrée |
| `BATCH_SIZE` | 64 | Taille des lots |
| `EPOCHS_A` | 10 | Epochs pour le CNN from scratch |
| `EPOCHS_B` | 15 | Epochs pour le transfert learning |
| `VAL_SPLIT` | 0.15 | Part du train utilisée pour la validation |
| `SEED` | 42 | Seed pour la reproductibilité |

### Expérience A — CNN from scratch

Modifiez dans la cellule des hyperparamètres :

```python
EPOCHS_A  = 10       # nombre d'epochs
LR_A_SGD  = 0.01     # learning rate SGD
LR_A_ADAM = 1e-3     # learning rate Adam
BATCH_SIZE = 64
IMG_SIZE   = 64      # réduire pour accélérer (min 32)
```

Le modèle utilise :
- 3 blocs convolutifs (Conv → BatchNorm → ReLU → MaxPool → Dropout)
- Classifieur FC avec Dropout=0.5
- Scheduler : StepLR (÷10 tous les 7 epochs)

### Expérience B — Transfert Learning (ResNet-18)

```python
EPOCHS_B   = 15
LR_B_SGD   = 5e-3
LR_B_ADAM  = 1e-4    # LR faible car poids pré-entraînés
IMG_SIZE   = 224     # garder 224 pour ResNet-18
```

Le modèle utilise :
- ResNet-18 pré-entraîné sur ImageNet
- Tête FC remplacée : `Linear(512 → 2)` + Dropout=0.4
- Scheduler : CosineAnnealingLR

---

## Évaluation & rechargement du modèle

Les checkpoints sont sauvegardés automatiquement dans `checkpoints/` (non poussés sur GitHub).

Pour recharger et évaluer le meilleur modèle :

```python
# Rechargement depuis le disque
checkpoint   = torch.load('checkpoints/best_model_final.pth', map_location=device)
loaded_model = build_transfer_model(num_classes=2).to(device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

# Évaluation sur le jeu de test
loss, acc, precision, recall = evaluate(loaded_model, testloader)
print(f'Acc={acc:.4f}  Prec={precision:.4f}  Rec={recall:.4f}')
```

---

## Résultats

### Tableau comparatif (exemple après 10 epochs, IMG_SIZE=64)

| Modèle | Optimiseur | Val Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CNN scratch | SGD | ~0.72 | ~0.74 | ~0.68 | ~0.71 |
| CNN scratch | Adam | ~0.75 | ~0.76 | ~0.72 | ~0.74 |
| ResNet-18 | SGD | ~0.88 | ~0.89 | ~0.87 | ~0.88 |
| ResNet-18 | Adam | ~0.91 | ~0.92 | ~0.90 | ~0.91 |

> Les valeurs exactes varient selon le nombre d'epochs et le matériel utilisé. Vos courbes générées dans le notebook font foi.

### Analyse

**Convergence** : le transfert learning converge en 3 à 5 epochs là où le CNN from scratch nécessite 15 à 20 epochs pour atteindre un résultat équivalent. Les poids ImageNet fournissent des features génériques (bords, textures, formes) immédiatement exploitables.

**Impact des optimiseurs** : Adam converge plus vite en début d'entraînement grâce à son adaptation du learning rate par paramètre. SGD avec momentum peut toutefois atteindre de meilleures performances finales avec un bon scheduler. Le CosineAnnealingLR lisse la descente et évite les oscillations en fin d'entraînement.

**Rôle de Dropout et BatchNorm** : la Batch Normalization stabilise les activations et accélère la convergence en normalisant les entrées de chaque couche. Le Dropout (p=0.5 en FC, p=0.25 en conv) force le réseau à ne pas sur-spécialiser ses neurones, réduisant significativement le sur-apprentissage visible sur les courbes train/val.

### Limites & pistes d'amélioration

- Tester des architectures plus profondes (ResNet-50, EfficientNet-B0) pour potentiellement gagner 2 à 3 points d'accuracy.
- Fine-tuning progressif : geler d'abord le backbone ResNet, entraîner uniquement la tête, puis dégeler par étapes avec un LR décroissant.
- Augmentation plus agressive : mixup, cutout, ou AutoAugment.
- Déséquilibre des classes : vérifier si le dataset est équilibré ; si non, utiliser un `WeightedRandomSampler`.

---

## GPU

Le notebook vérifie automatiquement la disponibilité du GPU :

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device utilisé :', device)
```

Sur Google Colab : **Exécution → Modifier le type d'exécution → Accélérateur matériel → GPU**.

---

## Reproductibilité

Le seed est fixé en début de notebook :

```python
SEED = 42
set_seed(SEED)  # fixe random, numpy, torch, cudnn
```

---

## Remise

- Poussez votre code sur GitHub **sans les données ni les modèles** (voir `.gitignore`).
- Envoyez le lien du dépôt à **diallomous@gmail.com**
- **Date limite : jeudi 04 juin 2026 avant 23h00 (Africa/Dakar)**
