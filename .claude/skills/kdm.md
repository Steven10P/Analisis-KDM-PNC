---
name: kdm
description: >
  Guide for applying the kdm-torch library (Kernel Density Matrices) to machine learning
  classification tasks. Use this skill whenever the user mentions KDM, KDMClassModel,
  kernel density matrices, init_kdm_layer, quantum-inspired classifiers, or wants to
  train/evaluate a KDM model on any dataset. Trigger even for questions like "cómo
  entreno el KDM", "qué hiperparámetros tiene KDM", or "el KDM no converge".
---

# KDM Skill — Kernel Density Matrices (kdm-torch)

The `kdm-torch` library implements **Kernel Density Matrix** classifiers: quantum-inspired
probabilistic models that represent data distributions as positive semi-definite matrices
(density matrices). They are trained with Negative Log-Likelihood (NLL) loss and output
class probabilities via Born's rule.

**GitHub:** `https://github.com/fagonzalezo/kdm`
**Package name:** `kdm-torch` (import as `from kdm import ...`)

---

## Core classes

```python
from kdm.models import KDMClassModel
from kdm.init import init_kdm_layer
```

### `KDMClassModel` (v2 API)

In kdm-torch v2 the encoder is a **separate `nn.Module`** you provide. Constructor:

| Parameter | Type | Description |
|-----------|------|-------------|
| `encoded_size` | int | Output dim of your encoder (= KDM input dim) |
| `dim_y` | int | Number of output classes |
| `encoder` | nn.Module | Any PyTorch module: Linear, MLP, CNN, etc. |
| `n_comp` | int | Number of prototype components (capacity) |
| `sigma` | float | RBF kernel bandwidth |

```python
import torch.nn as nn
from kdm.models import KDMClassModel

encoder = nn.Sequential(nn.Linear(3, 16), nn.Tanh())   # raw input → encoded space

model = KDMClassModel(
    encoded_size=16,    # must match encoder output dim
    dim_y=10,           # number of classes
    encoder=encoder,
    n_comp=128,
    sigma=0.5,
)
```

### `init_kdm_layer`

Must be called **before** the first forward pass. It initializes the KDM prototype
weights from a sample of encoded training data + one-hot labels.

```python
import torch.nn.functional as F
from kdm.init import init_kdm_layer

# x_init: (n_comp, raw_input_dim) — sample from training data
x_init = x_train[:128].float()
y_init = y_train[:128].long()

with torch.no_grad():
    encoded_init = model.encoder(x_init)              # (n_comp, encoded_size)
y_onehot = F.one_hot(y_init, num_classes=10).float()  # (n_comp, 10)

init_kdm_layer(model.kdm, encoded_init, y_onehot, init_sigma=True)
# init_sigma=True auto-sets sigma from nearest-neighbor distances (recommended)
```

---

## Training loop pattern

```python
import torch
import torch.nn as nn

criterion = nn.NLLLoss()   # KDM forward() returns log-probabilities
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb = xb.float()          # must be float32
        yb = yb.long()
        optimizer.zero_grad()
        out = model(xb)           # (batch, n_classes) log-probs
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
```

**Important input rules:**
- Input must be `float32` — always call `.float()` before passing to model
- KDM v2 `forward(x)` applies the encoder internally — pass **raw** features, not encoded
- Labels must be `long` (int64)
- The `model.kdm` attribute is the internal KDMLayer; `model.encoder` is your encoder

---

## Evaluation

```python
model.eval()
with torch.no_grad():
    logprobs = model(x_test.float())          # (N, n_classes) log-probs
    preds = logprobs.argmax(dim=1)            # predicted class indices
    acc = (preds == y_test).float().mean().item()
```

To get probabilities (for ROC curves, etc.):
```python
probs = logprobs.exp()    # converts log-prob to prob
```

---

## Hyperparameter guidance

| Hyperparameter | Typical range | Effect |
|----------------|--------------|--------|
| `n_comp` | 64–512 | Higher = more capacity, slower training |
| `encoded_sz` | 8–64 | Higher = richer kernel representation |
| `sigma` | 0.1–2.0 | Controls RBF width; tune with validation set |
| `lr` | 1e-4 – 1e-2 | Adam works well; start at 1e-3 |
| `epochs` | 20–100 | KDM typically converges in 20–50 epochs |

---

## Known behaviors in this project

- **3D PCA + noise σ=1**: KDM reaches ~42% accuracy — expected given low inter-class
  separability in 3D space. This is a scientifically valid result, not a bug.
- **sigma auto-tuning**: Pass `init_sigma=True` to `init_kdm_layer` — it sets sigma
  from the 2nd-nearest-neighbor distances of the encoded support points. Recommended.
- **NLL loss**: If loss starts high (~2.3 for 10 classes) and slowly decreases, that
  is normal behavior — NLL for a uniform 10-class distribution is ln(10) ≈ 2.303.
- **Saving/loading weights**: Use `torch.save(model.state_dict(), path)` and
  `model.load_state_dict(torch.load(path))`. Always call `init_kdm_layer` before
  loading if building a new model instance.

---

## Quick diagnostic checklist

If KDM is not converging or accuracy is stuck at ~10%:
1. Did you call `init_kdm_layer` before training? (most common cause)
2. Is input `float32`? (check with `xb.dtype`)
3. Is sigma too large or too small? (try values 0.1, 0.3, 0.5, 1.0)
4. Is `n_comp` too small for the dataset complexity? (try 256 or 512)
5. Is the learning rate too high? (reduce to 1e-4)
