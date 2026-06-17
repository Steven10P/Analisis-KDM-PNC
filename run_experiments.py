"""
run_experiments.py
Pipeline unificado KDM vs PNC sobre mnist_dim_3_min_3_noise_1-dataset.tar

Dataset: MNIST reducido a 3 dimensiones (PCA) con ruido gaussiano sigma=1.0.
Genera el .tar automáticamente si no existe.

Artefactos MLflow:
  - figures/loglog_loss_curves.png     → gráfica log-log épocas vs loss
  - figures/train_conf_matrix_kdm.png  → CM train KDM
  - figures/test_conf_matrix_kdm.png   → CM test KDM
  - figures/train_conf_matrix_pnc.png  → CM train PNC
  - figures/test_conf_matrix_pnc.png   → CM test PNC
  - models/kdm_model.pth
  - models/pnc_model.pth
"""
import sys
import os
import tarfile
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import mlflow
import mlflow.pytorch

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_DIR   = Path(__file__).resolve().parent
PNC_CIRCS  = REPO_DIR / "models" / "pnc_circuits"
sys.path.insert(0, str(PNC_CIRCS))

from kdm.models.kdm_class_model import KDMClassModel
from kdm.init import init_kdm_layer
from circuits.pncrc import GenDisPNCRC

# ── Dataset paths ──────────────────────────────────────────────────────────────
DATA_DIR   = REPO_DIR / "data"
TAR_PATH   = DATA_DIR / "mnist_dim_3_min_3_noise_1-dataset.tar"
OUT_DIR    = REPO_DIR / "resultados" / "mnist_noise"
FIG_DIR    = OUT_DIR / "graficas"
MDL_DIR    = OUT_DIR / "modelos"
MLFLOW_URI = f"sqlite:///{REPO_DIR / 'mlflow.db'}"

for d in (DATA_DIR, FIG_DIR, MDL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ─────────────────────────────────────────────────────────────
SEED       = 42
EPOCHS     = 30
BATCH_SIZE = 256
NUM_CLASSES = 10
PCA_DIM    = 3
NOISE_STD  = 1.0

KDM_CFG = dict(encoded_size=16, n_comp=128, lr=1e-3, sigma=0.5)
PNC_CFG = dict(components=5, mixing="sum", lr=0.01, momentum=0.9)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET — genera el .tar si no existe, luego extrae y carga
# ══════════════════════════════════════════════════════════════════════════════

def _build_tar(path: Path) -> None:
    """Carga MNIST completo, aplica PCA(3) + ruido N(0,1) y guarda en tar."""
    print("[data] Generando mnist_dim_3_min_3_noise_1-dataset.tar …")
    mnist = fetch_openml("mnist_784", version=1, data_home=str(DATA_DIR),
                         as_frame=False, parser="auto")
    X_raw  = mnist.data.astype(np.float32) / 255.0   # (70000, 784)
    y_raw  = mnist.target.astype(np.int64)

    # 80 % train / 20 % test  (seed fijo → reproducible)
    rng   = np.random.default_rng(SEED)
    idx   = rng.permutation(len(y_raw))
    split = int(0.857 * len(idx))                    # ≈ 60 000 train / 10 000 test
    tr, te = idx[:split], idx[split:]

    pca  = PCA(n_components=PCA_DIM, random_state=SEED)
    X_tr = pca.fit_transform(X_raw[tr]).astype(np.float32)
    X_te = pca.transform(X_raw[te]).astype(np.float32)

    # Ruido gaussiano (noise_1 → std = 1.0)
    X_tr += rng.normal(0, NOISE_STD, X_tr.shape).astype(np.float32)
    X_te += rng.normal(0, NOISE_STD, X_te.shape).astype(np.float32)

    arrays = {
        "X_train.npy": X_tr, "y_train.npy": y_raw[tr],
        "X_test.npy":  X_te, "y_test.npy":  y_raw[te],
    }
    with tarfile.open(path, "w") as tf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr)
            buf.seek(0)
            info = tarfile.TarInfo(name=name)
            info.size = len(buf.getvalue())
            tf.addfile(info, buf)
    print(f"[data] Guardado en {path}")


def load_dataset(path: Path):
    """Extrae el .tar y retorna (X_tr, y_tr, X_te, y_te) en numpy."""
    if not path.exists():
        _build_tar(path)
    print(f"[data] Cargando {path.name} …")
    with tarfile.open(path, "r") as tf:
        def _arr(name):
            f = tf.extractfile(name)
            return np.load(io.BytesIO(f.read()))
        X_tr = _arr("X_train.npy")
        y_tr = _arr("y_train.npy")
        X_te = _arr("X_test.npy")
        y_te = _arr("y_test.npy")
    print(f"[data] Train {X_tr.shape}  Test {X_te.shape}")
    return X_tr, y_tr, X_te, y_te


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DataLoaders — dos versiones del mismo split
# ══════════════════════════════════════════════════════════════════════════════

def make_loaders_kdm(X_tr, y_tr, X_te, y_te):
    """KDM: entrada float32, shape (N, 3)."""
    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
    te_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long))
    return (DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(te_ds, batch_size=BATCH_SIZE))


def make_loaders_pnc(X_tr, y_tr, X_te, y_te):
    """PNC: MinMax → [0,255], reshape (N, 1, 3) ≡ imagen 1×3 px."""
    scaler = MinMaxScaler(feature_range=(0.0, 255.0))
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    # shape: (N, 3) → (N, 1, 3)  (height=1, width=3)
    X_tr_t = torch.tensor(X_tr_s).unsqueeze(1)
    X_te_t = torch.tensor(X_te_s).unsqueeze(1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)
    tr_ds = TensorDataset(X_tr_t, y_tr_t)
    te_ds = TensorDataset(X_te_t, y_te_t)
    return (DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(te_ds, batch_size=BATCH_SIZE),
            X_tr_t, y_tr_t, X_te_t, y_te_t)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  KDM — construcción, init y loop de entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

def build_kdm():
    encoder = nn.Sequential(
        nn.Linear(PCA_DIM, 32), nn.ReLU(),
        nn.Linear(32, KDM_CFG["encoded_size"]), nn.ReLU(),
    )
    model = KDMClassModel(
        encoded_size=KDM_CFG["encoded_size"],
        dim_y=NUM_CLASSES,
        encoder=encoder,
        n_comp=KDM_CFG["n_comp"],
        sigma=KDM_CFG["sigma"],
        sigma_trainable=True,
    )
    return model.to(DEVICE)


def init_kdm(model: KDMClassModel, X_tr: np.ndarray, y_tr: np.ndarray):
    """Inicializa prototipos de KDM con las primeras n_comp muestras."""
    n = KDM_CFG["n_comp"]
    x_init = torch.tensor(X_tr[:n], dtype=torch.float32).to(DEVICE)
    y_init = torch.tensor(y_tr[:n], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        enc = model.encoder(x_init).cpu().numpy()
    y_oh = F.one_hot(y_init, NUM_CLASSES).float().cpu().numpy()
    init_kdm_layer(model.kdm, enc, y_oh, init_sigma=True)
    print(f"[KDM] Inicializado con {n} prototipos. sigma={model.kernel.sigma:.4f}")


def train_kdm(model, tr_loader, te_loader):
    opt    = torch.optim.Adam(model.parameters(), lr=KDM_CFG["lr"])
    hist   = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            probs = model(xb)
            loss  = F.nll_loss(torch.log(probs + 1e-8), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(yb)
            tc += (probs.argmax(1) == yb).sum().item()
            tn += len(yb)

        # ── eval ──
        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                probs = model(xb)
                loss  = F.nll_loss(torch.log(probs + 1e-8), yb)
                vl += loss.item() * len(yb)
                vc += (probs.argmax(1) == yb).sum().item()
                vn += len(yb)

        hist["train_loss"].append(tl / tn)
        hist["train_acc"].append(tc / tn)
        hist["test_loss"].append(vl / vn)
        hist["test_acc"].append(vc / vn)

        mlflow.log_metrics({
            "kdm_train_loss": tl / tn, "kdm_train_acc": tc / tn,
            "kdm_test_loss":  vl / vn, "kdm_test_acc":  vc / vn,
        }, step=epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[KDM] ep{epoch:03d} | "
                  f"train_loss={tl/tn:.4f} acc={tc/tn:.4f} | "
                  f"test_loss={vl/vn:.4f} acc={vc/vn:.4f}")
    return hist


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PNC — construcción y loop de entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

def build_pnc():
    return GenDisPNCRC(
        height=1, width=PCA_DIM,
        components=PNC_CFG["components"],
        n_classes=NUM_CLASSES,
        mixing=PNC_CFG["mixing"],
    ).to(DEVICE)


def train_pnc(model, tr_loader, te_loader):
    opt  = torch.optim.SGD(model.parameters(),
                           lr=PNC_CFG["lr"], momentum=PNC_CFG["momentum"])
    crit = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb.squeeze(1).float())     # (B, H, W) → (B, n_classes)
            loss   = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(yb)
            tc += (logits.argmax(1) == yb).sum().item()
            tn += len(yb)

        # ── eval ──
        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb.squeeze(1).float())
                loss   = crit(logits, yb)
                vl += loss.item() * len(yb)
                vc += (logits.argmax(1) == yb).sum().item()
                vn += len(yb)

        hist["train_loss"].append(tl / tn)
        hist["train_acc"].append(tc / tn)
        hist["test_loss"].append(vl / vn)
        hist["test_acc"].append(vc / vn)

        mlflow.log_metrics({
            "pnc_train_loss": tl / tn, "pnc_train_acc": tc / tn,
            "pnc_test_loss":  vl / vn, "pnc_test_acc":  vc / vn,
        }, step=epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[PNC] ep{epoch:03d} | "
                  f"train_loss={tl/tn:.4f} acc={tc/tn:.4f} | "
                  f"test_loss={vl/vn:.4f} acc={vc/vn:.4f}")
    return hist


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Predicciones completas (para matrices de confusión)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_kdm(model, loader):
    model.eval()
    preds, truths = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        preds.extend(model(xb).argmax(1).cpu().tolist())
        truths.extend(yb.tolist())
    return np.array(truths), np.array(preds)


@torch.no_grad()
def predict_pnc(model, loader):
    model.eval()
    preds, truths = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb.squeeze(1).float())
        preds.extend(logits.argmax(1).cpu().tolist())
        truths.extend(yb.tolist())
    return np.array(truths), np.array(preds)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Visualizaciones
# ══════════════════════════════════════════════════════════════════════════════

LABELS = [str(i) for i in range(10)]


def plot_loglog_loss(kdm_hist, pnc_hist) -> Path:
    epochs = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, split in zip(axes, ("train", "test")):
        kdm_loss = np.array(kdm_hist[f"{split}_loss"])
        pnc_loss = np.array(pnc_hist[f"{split}_loss"])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.plot(epochs, kdm_loss, "o-", color="#2196F3", lw=2, ms=4, label="KDM")
        ax.plot(epochs, pnc_loss, "s-", color="#FF5722", lw=2, ms=4, label="PNC")
        ax.set_xlabel("Época (log)", fontsize=11)
        ax.set_ylabel("Loss (log)", fontsize=11)
        ax.set_title(f"Log-Log Loss — {split.capitalize()}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.suptitle("Curvas de Convergencia KDM vs PNC\n"
                 "(mnist_dim_3_min_3_noise_1)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "loglog_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _conf_matrix_fig(y_true, y_pred, title: str) -> plt.Figure:
    cm   = confusion_matrix(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=LABELS, yticklabels=LABELS)
    axes[0].set_title("Conteos absolutos")
    axes[0].set_xlabel("Predicción"); axes[0].set_ylabel("Real")

    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=LABELS, yticklabels=LABELS)
    axes[1].set_title("Normalizada por fila")
    axes[1].set_xlabel("Predicción"); axes[1].set_ylabel("Real")

    fig.suptitle(f"{title}  |  Accuracy = {acc:.4f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def save_conf_matrices(kdm_tr, kdm_te, pnc_tr, pnc_te) -> list[Path]:
    paths = []
    specs = [
        (*kdm_tr, "KDM — Train",  "train_conf_matrix_kdm.png"),
        (*kdm_te, "KDM — Test",   "test_conf_matrix_kdm.png"),
        (*pnc_tr, "PNC — Train",  "train_conf_matrix_pnc.png"),
        (*pnc_te, "PNC — Test",   "test_conf_matrix_pnc.png"),
    ]
    for y_true, y_pred, title, fname in specs:
        fig  = _conf_matrix_fig(y_true, y_pred, title)
        path = FIG_DIR / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # 7.1 Datos
    X_tr, y_tr, X_te, y_te = load_dataset(TAR_PATH)

    tr_kdm, te_kdm             = make_loaders_kdm(X_tr, y_tr, X_te, y_te)
    tr_pnc, te_pnc, \
    Xtr_pnc, ytr_pnc, \
    Xte_pnc, yte_pnc          = make_loaders_pnc(X_tr, y_tr, X_te, y_te)

    # 7.2 MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("kdm-pnc-mnist-noise")

    with mlflow.start_run(run_name="KDM_vs_PNC_mnist_dim3_noise1"):

        # ── params ──────────────────────────────────────────────────────────
        mlflow.log_params({
            "dataset":        TAR_PATH.name,
            "pca_dim":        PCA_DIM,
            "noise_std":      NOISE_STD,
            "epochs":         EPOCHS,
            "batch_size":     BATCH_SIZE,
            # KDM
            "kdm_encoded_sz": KDM_CFG["encoded_size"],
            "kdm_n_comp":     KDM_CFG["n_comp"],
            "kdm_lr":         KDM_CFG["lr"],
            "kdm_sigma":      KDM_CFG["sigma"],
            # PNC
            "pnc_components": PNC_CFG["components"],
            "pnc_lr":         PNC_CFG["lr"],
            "pnc_momentum":   PNC_CFG["momentum"],
            "pnc_mixing":     PNC_CFG["mixing"],
        })

        # ── KDM ─────────────────────────────────────────────────────────────
        print("\n" + "═" * 60)
        print("  ENTRENANDO KDM")
        print("═" * 60)
        kdm = build_kdm()
        init_kdm(kdm, X_tr, y_tr)
        kdm_hist = train_kdm(kdm, tr_kdm, te_kdm)

        torch.save(kdm.state_dict(), MDL_DIR / "kdm_model.pth")

        kdm_tr_pred = predict_kdm(kdm, tr_kdm)
        kdm_te_pred = predict_kdm(kdm, te_kdm)

        mlflow.log_metrics({
            "kdm_final_train_acc": float(accuracy_score(*kdm_tr_pred)),
            "kdm_final_test_acc":  float(accuracy_score(*kdm_te_pred)),
        })

        # ── PNC ─────────────────────────────────────────────────────────────
        print("\n" + "═" * 60)
        print("  ENTRENANDO PNC")
        print("═" * 60)
        pnc = build_pnc()
        pnc_hist = train_pnc(pnc, tr_pnc, te_pnc)

        torch.save(pnc.state_dict(), MDL_DIR / "pnc_model.pth")

        pnc_tr_pred = predict_pnc(pnc, tr_pnc)
        pnc_te_pred = predict_pnc(pnc, te_pnc)

        mlflow.log_metrics({
            "pnc_final_train_acc": float(accuracy_score(*pnc_tr_pred)),
            "pnc_final_test_acc":  float(accuracy_score(*pnc_te_pred)),
        })

        # ── Visualizaciones ─────────────────────────────────────────────────
        print("\n[plots] Generando gráficas …")
        loglog_path = plot_loglog_loss(kdm_hist, pnc_hist)
        cm_paths    = save_conf_matrices(
            kdm_tr_pred, kdm_te_pred,
            pnc_tr_pred, pnc_te_pred,
        )

        # ── Artefactos MLflow ────────────────────────────────────────────────
        mlflow.log_artifact(str(loglog_path),          artifact_path="figures")
        for p in cm_paths:
            mlflow.log_artifact(str(p),                artifact_path="figures")
        mlflow.log_artifact(str(MDL_DIR / "kdm_model.pth"), artifact_path="models")
        mlflow.log_artifact(str(MDL_DIR / "pnc_model.pth"), artifact_path="models")

        run_id = mlflow.active_run().info.run_id

    # ── Resumen ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RESULTADOS FINALES")
    print("═" * 60)
    print(f"  KDM  train_acc = {accuracy_score(*kdm_tr_pred):.4f}  "
          f"test_acc = {accuracy_score(*kdm_te_pred):.4f}")
    print(f"  PNC  train_acc = {accuracy_score(*pnc_tr_pred):.4f}  "
          f"test_acc = {accuracy_score(*pnc_te_pred):.4f}")
    print(f"\n  MLflow run_id : {run_id}")
    print(f"  Figuras       : {FIG_DIR}")
    print(f"  Modelos       : {MDL_DIR}")
    print("═" * 60)

    print("\n[KDM] Classification report (test):")
    print(classification_report(*kdm_te_pred, target_names=LABELS))
    print("\n[PNC] Classification report (test):")
    print(classification_report(*pnc_te_pred, target_names=LABELS))


if __name__ == "__main__":
    main()
