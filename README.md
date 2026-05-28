#  CNN Cats vs Dogs — From Scratch & Transfer Learning

**Auteur** : Oumar Camara  
**Cours** : DV Deep Learning  


Classification binaire chats/chiens en comparant un **CNN entraîné from scratch** et un modèle en **transfert learning (ResNet-18 pré-entraîné sur ImageNet)**.

---

## Environnement

### Configuration utilisée

| | |
|---|---|
| Framework | PyTorch 2.10.0+cu128 |
| GPU | NVIDIA T4 (Google Colab) |
| CUDA | disponible |
| Python | 3.10 |

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

Placez le dossier dans votre Google Drive à l'emplacement suivant :

```
MyDrive/
└── TP_Deep_Learning/
    └── cnn_transfert_learning/
        ├── Cat_Dog_data/
        │   ├── train/
        │   │   ├── cat/    (images d'entraînement)
        │   │   └── dog/
        │   └── test/
        │       ├── cat/    (images de test)
        │       └── dog/
        ├── checkpoints/    (créé automatiquement, non versionné)
        └── TP_CNN_CatsVsDogs.ipynb
```

> **Dataset** : 22 618 images d'entraînement · 2 500 images de test · 2 classes : `cat`, `dog`

---

## Structure du dépôt

```
cnn-catsdogs-OumarCamara/
├── TP_CNN_CatsVsDogs.ipynb   # Notebook principal (15 sections)
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

---

## Lancer l'entraînement

Ouvrez `TP_CNN_CatsVsDogs.ipynb` dans **Google Colab** avec le GPU activé :  
`Exécution → Modifier le type d'exécution → Accélérateur matériel → GPU (T4)`

### Hyperparamètres globaux

| Paramètre | Valeur | Description |
|---|---|---|
| `IMG_SIZE` | 224 | Taille des images (standard ResNet) |
| `BATCH_SIZE` | 64 | Taille des lots |
| `NUM_WORKERS` | 2 | Workers DataLoader |
| `SEED` | 42 | Seed de reproductibilité |
| `MEAN` | [0.485, 0.456, 0.406] | Normalisation ImageNet |
| `STD` | [0.229, 0.224, 0.225] | Normalisation ImageNet |

### Expérience A — CNN from scratch

Architecture : **3 blocs convolutifs** (Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout2d) + classifieur FC.

```python
EPOCHS_A  = 2          # epochs (augmenter pour de meilleurs résultats)
LR_A_SGD  = 0.01       # learning rate SGD
LR_A_ADAM = 1e-3       # learning rate Adam
BATCH_SIZE = 64
IMG_SIZE   = 224
```

- **Batch Normalization** : après chaque convolution, stabilise les activations
- **Dropout conv** : p=0.25 par bloc
- **Dropout FC** : p=0.5 dans le classifieur
- **Scheduler** : StepLR (÷10 tous les 7 epochs)

### Expérience B — Transfer Learning (ResNet-18)

```python
EPOCHS_B   = 5
LR_B_SGD   = 5e-3
LR_B_ADAM  = 1e-4      # LR faible car poids pré-entraînés
IMG_SIZE   = 224       # obligatoire pour ResNet-18
```

- **Base** : ResNet-18 pré-entraîné sur ImageNet (11 177 538 paramètres)
- **Tête remplacée** : `Dropout(0.4)` + `Linear(512 → 2)`
- **Scheduler** : CosineAnnealingLR (descente progressive sur toute la durée)

---

## Évaluation & Rechargement du modèle

Les checkpoints sont sauvegardés automatiquement dans `checkpoints/` (non poussés sur GitHub).

```
checkpoints/
├── scratch_sgd_best.pth
├── scratch_adam_best.pth
├── transfer_sgd_best.pth
├── transfer_adam_best.pth
└── best_model_final.pth    ← meilleur modèle toutes expériences confondues
```

Pour recharger et évaluer :

```python
import torch
from torchvision import models
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reconstruction de l'architecture
def build_transfer_model(num_classes=2, dropout_fc=0.4):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(dropout_fc),
        nn.Linear(512, num_classes)
    )
    return model

# Rechargement
checkpoint   = torch.load('checkpoints/best_model_final.pth', map_location=device)
loaded_model = build_transfer_model().to(device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

# Évaluation
loss, acc, precision, recall = evaluate(loaded_model, testloader)
print(f'Acc={acc:.4f}  Prec={precision:.4f}  Rec={recall:.4f}')
```

---

## Résultats

### Tableau comparatif (résultats réels obtenus)

| Modèle | Optimiseur | Epochs | Val Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| CNN scratch | SGD | 2 | 0.638 | 0.722 | 0.450 |
| CNN scratch | Adam | 2 | **0.680** | 0.754 | 0.534 |
| ResNet-18 | SGD | 5 | 0.989 | 0.990 | 0.989 |
| ResNet-18 | Adam | 5 | **0.990** | 0.994 | 0.986 |

**Meilleur modèle** : ResNet-18 + Adam → `Acc=0.9896  Prec=0.9935  Rec=0.9856` (test final après rechargement)

**Matrice de confusion (meilleur modèle — Transfer Adam)**

|  | Prédit chat | Prédit chien |
|---|---|---|
| **Réel chat** | — | 8 faux positifs |
| **Réel chien** | 18 faux négatifs | — |

### Analyse comparative

**Convergence** : le transfert learning atteint 98% d'accuracy dès la 1ère epoch, là où le CNN from scratch plafonne à 64% après 2 epochs. Les features ImageNet (bords, textures, formes) sont immédiatement exploitables pour distinguer chats et chiens, tâche très proche des données d'ImageNet.

**Impact des optimiseurs** : Adam converge plus vite que SGD sur les deux expériences. Pour le CNN scratch, l'écart est net dès la 1ère epoch (acc 0.577 SGD vs 0.577 Adam à l'epoch 1, mais 0.638 vs 0.680 à l'epoch 2). Pour le transfer learning, les deux optimiseurs sont quasi équivalents — la qualité des poids pré-entraînés efface les différences d'optimiseur.

**Dropout + BatchNorm** : la Batch Normalization stabilise l'entraînement du CNN scratch et évite les explosions de gradient fréquentes sans elle. Le Dropout réduit le sur-apprentissage visible par l'écart train/val : val_loss < train_loss sur certaines epochs confirme que le modèle généralise.

### Limites & pistes d'amélioration

- Le CNN scratch n'a tourné que 2 epochs — il faut 15-20 epochs pour atteindre 75-80% d'accuracy.
- Fine-tuning progressif pour ResNet-18 : geler le backbone les 2 premières epochs puis dégeler avec un LR 10× plus faible.
- Tester EfficientNet-B0 ou MobileNetV3 pour un meilleur compromis vitesse/performance.
- Les 18 faux négatifs (chiens ratés) suggèrent un léger biais vers la classe `cat` — un `WeightedRandomSampler` pourrait corriger ça.

---

## GPU

Le notebook détecte automatiquement le GPU :

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# → Device utilisé : cuda  (T4 sur Colab)
```

Sur Google Colab : **Exécution → Modifier le type d'exécution → GPU**.

---

## Reproductibilité

```python
SEED = 42
set_seed(SEED)  # fixe random, numpy, torch, cudnn.deterministic=True
```





