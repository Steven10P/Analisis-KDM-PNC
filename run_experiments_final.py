"""
run_experiments_final.py
Pipeline definitivo KDM vs PNC — mnist_dim_3_min_3_noise_1

Mejoras sobre run_experiments.py:
  - Lee configuracion desde configs/experiment_config.yaml
  - Preprocesamiento estandar Z-score (comparable entre modelos)
  - KDM v2 API con encoder MLP configurable
  - Checkpoints guardados en checkpoints/ cada 10 epochs
  - MLflow tracking completo (params, metricas por epoch, artefactos)
  - Dispara experiments/analysis.py al finalizar
"""
import sys, os, io, tarfile, yaml, subprocess
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
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize, StandardScaler
import mlflow
import mlflow.pytorch

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_DIR  = Path(__file__).resolve().parent
PNC_CIRCS = REPO_DIR / "models" / "pnc_circuits"
sys.path.insert(0, str(PNC_CIRCS))

from kdm.models import KDMClassModel
from kdm.init import init_kdm_layer
from circuits.pncrc import GenDisPNCRC

# ── Config ─────────────────────────────────────────────────────────────────────
CFG_PATH = REPO_DIR / "configs" / "experiment_config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SEED        = CFG["experiment"]["seed"]
NUM_CLASSES = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

# Directorios de salida
DATA_DIR   = REPO_DIR / "data"
CKPT_DIR   = REPO_DIR / "checkpoints"
FIG_DIR    = REPO_DIR / "resultados" / "mnist_noise" / "graficas"
MDL_DIR    = REPO_DIR / "resultados" / "mnist_noise" / "modelos"
for d in (DATA_DIR, CKPT_DIR, FIG_DIR, MDL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset():
    tar_path = REPO_DIR / CFG["dataset"]["path"]
    if not tar_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {tar_path}\n"
            "Ejecuta primero run_experiments.py para generarlo."
        )
    print(f"[data] Cargando {tar_path.name} ...")
    with tarfile.open(tar_path, "r") as tf:
        def _arr(name):
            return np.load(io.BytesIO(tf.extractfile(name).read()))
        X_tr, y_tr = _arr("X_train.npy"), _arr("y_train.npy")
        X_te, y_te = _arr("X_test.npy"),  _arr("y_test.npy")
    print(f"[data] Train {X_tr.shape}  Test {X_te.shape}")
    return X_tr, y_tr, X_te, y_te


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESAMIENTO ESTANDAR (embedding-kit equivalente)
#    Z-score sobre entrenamiento; mismo scaler aplicado al test.
#    Hace el espacio latente comparable entre KDM y PNC.
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_zscore(X_tr, X_te):
    """Estandarizacion Z-score: mu y sigma calculados solo sobre train."""
    scaler = StandardScaler()
    X_tr_z = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_z = scaler.transform(X_te).astype(np.float32)
    print(f"[preproc] Z-score: mu={scaler.mean_.round(3)}, std={scaler.scale_.round(3)}")
    return X_tr_z, X_te_z, scaler


def make_loaders(X_tr, y_tr, X_te, y_te, batch_size):
    """DataLoaders estandar: shape (N, 3), float32."""
    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
    te_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long))
    return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True),
            DataLoader(te_ds, batch_size=batch_size, shuffle=False))


def make_loaders_pnc(X_tr, y_tr, X_te, y_te, batch_size):
    """
    PNC espera entradas en rango [0, 255] con shape (N, 1, D).
    Partimos del dato ya estandarizado y lo re-escalamos a [0, 255].
    """
    lo, hi = X_tr.min(), X_tr.max()
    def scale(X):
        return ((X - lo) / (hi - lo + 1e-8) * 255.0).astype(np.float32)
    X_tr_s = scale(X_tr)
    X_te_s = scale(X_te)
    X_tr_t = torch.tensor(X_tr_s).unsqueeze(1)   # (N, 1, 3)
    X_te_t = torch.tensor(X_te_s).unsqueeze(1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)
    tr_ds  = TensorDataset(X_tr_t, y_tr_t)
    te_ds  = TensorDataset(X_te_t, y_te_t)
    return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True),
            DataLoader(te_ds, batch_size=batch_size, shuffle=False),
            X_tr_t, y_tr_t, X_te_t, y_te_t)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODELO KDM (kdm-torch v2 API)
# ══════════════════════════════════════════════════════════════════════════════

def build_kdm(cfg):
    enc_sz = cfg["encoded_size"]
    hidden = cfg["encoder_hidden"]
    encoder = nn.Sequential(
        nn.Linear(3, hidden), nn.ReLU(),
        nn.Linear(hidden, enc_sz), nn.Tanh(),
    )
    model = KDMClassModel(
        encoded_size=enc_sz,
        dim_y=NUM_CLASSES,
        encoder=encoder,
        n_comp=cfg["n_comp"],
        sigma=cfg["sigma"],
        sigma_trainable=cfg["sigma_trainable"],
    )
    return model.to(DEVICE)


def init_kdm(model, X_tr, y_tr, n_comp):
    x_init = torch.tensor(X_tr[:n_comp], dtype=torch.float32).to(DEVICE)
    y_init = torch.tensor(y_tr[:n_comp], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        enc = model.encoder(x_init).cpu().numpy()
    y_oh = F.one_hot(y_init, NUM_CLASSES).float().cpu().numpy()
    init_kdm_layer(model.kdm, enc, y_oh, init_sigma=True)
    print(f"[KDM] Inicializado — {n_comp} prototipos | sigma={model.kernel.sigma:.4f}")


def train_kdm(model, tr_loader, te_loader, cfg, run):
    opt  = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    nll  = nn.NLLLoss()
    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    epochs = cfg["epochs"]

    for ep in range(1, epochs + 1):
        # --- entrenamiento ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out  = model(xb.float())
            loss = nll(out, yb)
            loss.backward()
            opt.step()
            t_loss    += loss.item() * len(yb)
            t_correct += (out.argmax(1) == yb).sum().item()
            t_total   += len(yb)

        # --- evaluacion ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out  = model(xb.float())
                loss = nll(out, yb)
                v_loss    += loss.item() * len(yb)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total   += len(yb)

        tr_l = t_loss / t_total;  tr_a = t_correct / t_total
        te_l = v_loss / v_total;  te_a = v_correct / v_total
        hist["train_loss"].append(tr_l);  hist["train_acc"].append(tr_a)
        hist["test_loss"].append(te_l);   hist["test_acc"].append(te_a)

        # Log MLflow por epoch
        mlflow.log_metrics({
            "kdm_train_loss": tr_l, "kdm_train_acc": tr_a,
            "kdm_test_loss":  te_l, "kdm_test_acc":  te_a,
        }, step=ep)

        if ep % 10 == 0 or ep == 1:
            print(f"[KDM] ep{ep:03d} | "
                  f"train_loss={tr_l:.4f} acc={tr_a:.4f} | "
                  f"test_loss={te_l:.4f} acc={te_a:.4f}")

        # Checkpoint cada 10 epochs
        if ep % 10 == 0:
            ckpt = CKPT_DIR / f"kdm_ep{ep:03d}.pth"
            torch.save(model.state_dict(), ckpt)

    return hist


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELO PNC (GenDisPNCRC)
# ══════════════════════════════════════════════════════════════════════════════

def build_pnc(cfg):
    return GenDisPNCRC(
        num_var=3,
        num_vals=256,
        num_classes=NUM_CLASSES,
        components=cfg["components"],
        mixing=cfg["mixing"],
    ).to(DEVICE)


def train_pnc(model, tr_loader, te_loader, X_tr_t, y_tr_t, X_te_t, y_te_t, cfg):
    opt   = torch.optim.SGD(model.parameters(),
                             lr=cfg["lr"], momentum=cfg["momentum"])
    ce    = nn.CrossEntropyLoss()
    hist  = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    epochs = cfg["epochs"]

    for ep in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out  = model(xb.float())
            loss = ce(out, yb)
            if torch.isnan(loss):
                break
            loss.backward()
            opt.step()
            t_loss    += loss.item() * len(yb)
            t_correct += (out.argmax(1) == yb).sum().item()
            t_total   += len(yb)

        tr_l = t_loss / max(t_total, 1)
        tr_a = t_correct / max(t_total, 1)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out  = model(xb.float())
                loss = ce(out, yb)
                if not torch.isnan(loss):
                    v_loss    += loss.item() * len(yb)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total   += len(yb)

        te_l = v_loss / max(v_total, 1)
        te_a = v_correct / max(v_total, 1)
        hist["train_loss"].append(tr_l if not np.isnan(tr_l) else float("nan"))
        hist["train_acc"].append(tr_a)
        hist["test_loss"].append(te_l if not np.isnan(te_l) else float("nan"))
        hist["test_acc"].append(te_a)

        mlflow.log_metrics({
            "pnc_train_loss": tr_l if not np.isnan(tr_l) else -1,
            "pnc_train_acc":  tr_a,
            "pnc_test_loss":  te_l if not np.isnan(te_l) else -1,
            "pnc_test_acc":   te_a,
        }, step=ep)

        if ep % 10 == 0 or ep == 1:
            loss_str = f"{tr_l:.4f}" if not np.isnan(tr_l) else "nan"
            print(f"[PNC] ep{ep:03d} | "
                  f"train_loss={loss_str} acc={tr_a:.4f} | "
                  f"test_loss={te_l:.4f} acc={te_a:.4f}")

    return hist


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(kdm_hist, pnc_hist):
    epochs = range(1, len(kdm_hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in zip(axes,
                               ["train_loss", "test_loss"],
                               ["Loss de Entrenamiento", "Loss de Prueba"]):
        ax.plot(epochs, kdm_hist[key], label="KDM", color="#2196F3", lw=2)
        pnc_vals = [v if not np.isnan(v) else None for v in pnc_hist[key]]
        ax.plot(epochs, pnc_hist[key], label="PNC", color="#FF5722", lw=2,
                linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (NLL / CE)")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Curvas de Perdida: KDM vs PNC (MNIST-noise-σ1)", fontsize=13)
    out = FIG_DIR / "loss_curves_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(out), "figures")
    print(f"[fig] {out.name}")


def plot_loglog_loss(kdm_hist, pnc_hist):
    """Log-log de loss de entrenamiento para analizar velocidad de convergencia."""
    epochs = np.arange(1, len(kdm_hist["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(epochs, kdm_hist["train_loss"], "o-", color="#2196F3",
              label="KDM", lw=2, markersize=4)
    pnc_finite = [(e, v) for e, v in zip(epochs, pnc_hist["train_loss"])
                  if not np.isnan(v) and v > 0]
    if pnc_finite:
        ep_p, vl_p = zip(*pnc_finite)
        ax.loglog(ep_p, vl_p, "s--", color="#FF5722", label="PNC", lw=2, markersize=4)
    ax.set_xlabel("Epoch (escala log)")
    ax.set_ylabel("Train Loss (escala log)")
    ax.set_title("Convergencia Log-Log: KDM vs PNC")
    ax.legend()
    out = FIG_DIR / "loglog_loss_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(out), "figures")
    print(f"[fig] {out.name}")


def plot_confusion_matrix(y_true, y_pred, model_name, split):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(10), yticklabels=range(10),
                linewidths=0.3, linecolor="gray")
    ax.set_title(f"Matriz de Confusion — {model_name} ({split})")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    out = FIG_DIR / f"conf_matrix_{model_name.lower()}_{split}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(out), "figures")
    print(f"[fig] {out.name}")


def predict_all(model, loader, model_type="kdm"):
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE).float()
            out = model(xb)
            if model_type == "kdm":
                probs = out.exp()      # log-probs → probs
            else:
                probs = F.softmax(out, dim=1)
            all_preds.append(out.argmax(1).cpu())
            all_probs.append(probs.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PIPELINE DEFINITIVO: KDM vs PNC")
    print(f"  Config: {CFG_PATH.name}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Datos ──────────────────────────────────────────────────────────────
    X_tr_raw, y_tr, X_te_raw, y_te = load_dataset()

    # Preprocesamiento estandar Z-score (espacio latente comparable)
    X_tr, X_te, scaler = preprocess_zscore(X_tr_raw, X_te_raw)

    # DataLoaders
    kdm_tr, kdm_te = make_loaders(X_tr, y_tr, X_te, y_te,
                                   CFG["kdm"]["batch_size"])
    pnc_tr, pnc_te, X_tr_t, y_tr_t, X_te_t, y_te_t = make_loaders_pnc(
        X_tr, y_tr, X_te, y_te, CFG["pnc"]["batch_size"])

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CFG["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="kdm-pnc-final") as run:
        # Log parametros
        mlflow.log_params({
            "dataset":       CFG["dataset"]["path"],
            "pca_dim":       CFG["dataset"]["pca_dim"],
            "noise_std":     CFG["dataset"]["noise_std"],
            "preprocessing": CFG["preprocessing"]["method"],
            "seed":          SEED,
            **{f"kdm_{k}": v for k, v in CFG["kdm"].items()},
            **{f"pnc_{k}": v for k, v in CFG["pnc"].items()},
        })

        # ── KDM ────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  ENTRENANDO KDM")
        print("=" * 60)
        kdm_model = build_kdm(CFG["kdm"])
        init_kdm(kdm_model, X_tr, y_tr, CFG["kdm"]["n_comp"])
        kdm_hist = train_kdm(kdm_model, kdm_tr, kdm_te, CFG["kdm"], run)

        # Predicciones KDM
        kdm_tr_preds, kdm_tr_probs = predict_all(kdm_model, kdm_tr, "kdm")
        kdm_te_preds, kdm_te_probs = predict_all(kdm_model, kdm_te, "kdm")
        kdm_tr_acc = accuracy_score(y_tr, kdm_tr_preds)
        kdm_te_acc = accuracy_score(y_te, kdm_te_preds)

        mlflow.log_metrics({
            "kdm_final_train_acc": kdm_tr_acc,
            "kdm_final_test_acc":  kdm_te_acc,
        })

        # Guardar modelo KDM
        kdm_path = MDL_DIR / "kdm_model_final.pth"
        torch.save(kdm_model.state_dict(), kdm_path)
        mlflow.log_artifact(str(kdm_path), "models")

        # ── PNC ────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  ENTRENANDO PNC")
        print("=" * 60)
        pnc_model = build_pnc(CFG["pnc"])
        pnc_hist  = train_pnc(pnc_model, pnc_tr, pnc_te,
                               X_tr_t, y_tr_t, X_te_t, y_te_t, CFG["pnc"])

        pnc_tr_preds, pnc_tr_probs = predict_all(pnc_model, pnc_tr, "pnc")
        pnc_te_preds, pnc_te_probs = predict_all(pnc_model, pnc_te, "pnc")
        pnc_tr_acc = accuracy_score(y_tr, pnc_tr_preds)
        pnc_te_acc = accuracy_score(y_te, pnc_te_preds)

        mlflow.log_metrics({
            "pnc_final_train_acc": pnc_tr_acc,
            "pnc_final_test_acc":  pnc_te_acc,
        })

        pnc_path = MDL_DIR / "pnc_model_final.pth"
        torch.save(pnc_model.state_dict(), pnc_path)
        mlflow.log_artifact(str(pnc_path), "models")

        # ── Figuras ────────────────────────────────────────────────────────
        print("\n[figs] Generando figuras ...")
        plot_loss_curves(kdm_hist, pnc_hist)
        plot_loglog_loss(kdm_hist, pnc_hist)
        plot_confusion_matrix(y_tr, kdm_tr_preds, "KDM", "train")
        plot_confusion_matrix(y_te, kdm_te_preds, "KDM", "test")
        plot_confusion_matrix(y_tr, pnc_tr_preds, "PNC", "train")
        plot_confusion_matrix(y_te, pnc_te_preds, "PNC", "test")

        # ── Resultados finales ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  RESULTADOS FINALES")
        print("=" * 60)
        print(f"  KDM  train_acc={kdm_tr_acc:.4f}  test_acc={kdm_te_acc:.4f}")
        print(f"  PNC  train_acc={pnc_tr_acc:.4f}  test_acc={pnc_te_acc:.4f}")
        print(f"\n  MLflow run_id: {run.info.run_id}")

        print("\n[KDM] Classification report (test):")
        print(classification_report(y_te, kdm_te_preds, zero_division=0))
        print("\n[PNC] Classification report (test):")
        print(classification_report(y_te, pnc_te_preds, zero_division=0))

        mlflow.log_artifact(str(CFG_PATH), "config")

    # ── Disparar analysis.py automaticamente ──────────────────────────────
    print("\n" + "=" * 60)
    print("  DISPARANDO ANALISIS ESTADISTICO (data-analyst)")
    print("=" * 60)
    analysis_script = REPO_DIR / "experiments" / "analysis.py"
    if analysis_script.exists():
        result = subprocess.run(
            [sys.executable, str(analysis_script)],
            cwd=str(REPO_DIR),
            capture_output=False,
        )
        if result.returncode == 0:
            print("[OK] experiments/analysis.py completado")
        else:
            print(f"[WARN] analysis.py termino con codigo {result.returncode}")
    else:
        print(f"[WARN] {analysis_script} no encontrado — ejecuta experiments/analysis.py manualmente")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"  Figuras    : {FIG_DIR}")
    print(f"  Modelos    : {MDL_DIR}")
    print(f"  Checkpoints: {CKPT_DIR}")
    print(f"  Plots stats: {REPO_DIR / 'experiments' / 'plots'}")
    print(f"  Reporte    : {REPO_DIR / 'experiments' / 'statistical_report.md'}")


if __name__ == "__main__":
    main()
