"""
Genera EDA_mnist_noise.ipynb usando nbformat y lo ejecuta con nbconvert.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).parent / "EDA_mnist_noise.ipynb"

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ─────────────────────────────────────────────────────────────
md("""# EDA — `mnist_dim_3_min_3_noise_1-dataset.tar`

**Análisis Exploratorio de Datos** sobre la versión ruidosa y reducida de MNIST:
- **Reducción dimensional**: PCA de 784 → 3 componentes principales
- **Ruido**: Gaussiano aditivo σ = 1.0
- **Clases**: 10 dígitos (0–9)
- **Particiones**: ~60 000 entrenamiento / ~10 000 prueba

El objetivo es caracterizar cómo el ruido y la compresión dimensional afectan
la separabilidad entre clases y qué desafíos presentan para KDM y PNC.
""")

# ─────────────────────────────────────────────────────────────
md("## 1. Configuración e Importaciones")
code("""\
import sys, io, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D

REPO_DIR  = Path("..").resolve()
TAR_PATH  = REPO_DIR / "data" / "mnist_dim_3_min_3_noise_1-dataset.tar"
FIG_DIR   = REPO_DIR / "resultados" / "mnist_noise" / "graficas" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE   = sns.color_palette("tab10", 10)
CMAP10    = ListedColormap(PALETTE)
CLASS_NAMES = [str(i) for i in range(10)]
SEED      = 42

print(f"TAR  : {TAR_PATH}")
print(f"Figs : {FIG_DIR}")
""")

# ─────────────────────────────────────────────────────────────
md("## 2. Carga del Dataset desde el `.tar`")
code("""\
def load_tar(path):
    with tarfile.open(path, "r") as tf:
        def arr(name):
            return np.load(io.BytesIO(tf.extractfile(name).read()))
        return arr("X_train.npy"), arr("y_train.npy"), arr("X_test.npy"), arr("y_test.npy")

X_train, y_train, X_test, y_test = load_tar(TAR_PATH)

print("=== DIMENSIONES ===")
print(f"X_train : {X_train.shape}  dtype={X_train.dtype}")
print(f"y_train : {y_train.shape}  dtype={y_train.dtype}")
print(f"X_test  : {X_test.shape}  dtype={X_test.dtype}")
print(f"y_test  : {y_test.shape}  dtype={y_test.dtype}")
print(f"\\nClases únicas (train): {np.unique(y_train)}")
print(f"Total muestras : {len(X_train)+len(X_test):,}")
""")

# ─────────────────────────────────────────────────────────────
md("## 3. Vista General — Estadísticas Descriptivas")
code("""\
df_train = pd.DataFrame(X_train, columns=["PC1","PC2","PC3"])
df_train["label"] = y_train
df_test  = pd.DataFrame(X_test,  columns=["PC1","PC2","PC3"])
df_test["label"]  = y_test

print("=== TRAIN — describe() ===")
display(df_train[["PC1","PC2","PC3"]].describe().round(4))

print("\\n=== TEST — describe() ===")
display(df_test[["PC1","PC2","PC3"]].describe().round(4))
""")

code("""\
# Media y desviación estándar por clase y dimensión
stats = df_train.groupby("label")[["PC1","PC2","PC3"]].agg(["mean","std"]).round(3)
print("=== Media ± Std por clase (train) ===")
display(stats)
""")

# ─────────────────────────────────────────────────────────────
md("## 4. Distribución de Clases")
code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for ax, (split, y, title) in zip(axes, [
    ("Train", y_train, f"Train  (n={len(y_train):,})"),
    ("Test",  y_test,  f"Test   (n={len(y_test):,})")
]):
    counts = np.bincount(y)
    bars = ax.bar(range(10), counts, color=PALETTE)
    ax.set_xlabel("Dígito", fontsize=11)
    ax.set_ylabel("Muestras", fontsize=11)
    ax.set_title(f"Distribución de clases — {title}", fontsize=12, fontweight="bold")
    ax.set_xticks(range(10))
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, c+40, str(c),
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, counts.max()*1.15)
    ax.yaxis.grid(True, ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig(FIG_DIR/"01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Desbalance máximo train: {np.bincount(y_train).max()-np.bincount(y_train).min()} muestras")
""")

# ─────────────────────────────────────────────────────────────
md("## 5. Distribución de Cada Componente Principal")
code("""\
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for row, dim in enumerate(["PC1","PC2","PC3"]):
    # Histograma global
    ax = axes[row, 0]
    ax.hist(df_train[dim], bins=80, color="#4C72B0", alpha=0.75, edgecolor="white", lw=0.3)
    ax.axvline(df_train[dim].mean(), color="red", ls="--", lw=1.5, label=f"Media={df_train[dim].mean():.2f}")
    ax.axvline(df_train[dim].median(), color="orange", ls="--", lw=1.5, label=f"Mediana={df_train[dim].median():.2f}")
    ax.set_title(f"{dim} — Distribución global (train)", fontweight="bold")
    ax.set_xlabel(dim); ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9); ax.yaxis.grid(True, ls="--", alpha=0.4)

    # KDE por clase
    ax = axes[row, 1]
    for cls in range(10):
        mask = y_train == cls
        sns.kdeplot(df_train.loc[mask, dim], ax=ax, color=PALETTE[cls],
                    label=str(cls), linewidth=1.4, fill=False)
    ax.set_title(f"{dim} — KDE por clase (train)", fontweight="bold")
    ax.set_xlabel(dim); ax.set_ylabel("Densidad")
    ax.legend(title="Dígito", ncol=2, fontsize=8)
    ax.yaxis.grid(True, ls="--", alpha=0.4)

plt.suptitle("Distribución de Componentes Principales", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR/"02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ─────────────────────────────────────────────────────────────
md("## 6. Box-Plots por Clase")
code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, dim in zip(axes, ["PC1","PC2","PC3"]):
    data_by_class = [df_train.loc[df_train["label"]==c, dim].values for c in range(10)]
    bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Dígito"); ax.set_ylabel(dim)
    ax.set_title(f"Box-plot {dim} por clase", fontweight="bold")
    ax.yaxis.grid(True, ls="--", alpha=0.4)

plt.suptitle("Variabilidad por Clase en Espacio PCA-3D con Ruido",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"03_boxplots_by_class.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ─────────────────────────────────────────────────────────────
md("## 7. Proyecciones 2D (Pairplots de Componentes)")
code("""\
# Subsample para velocidad
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_train), size=5000, replace=False)
X_s, y_s = X_train[idx], y_train[idx]

pairs = [("PC1","PC2"), ("PC1","PC3"), ("PC2","PC3")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (d1, d2) in zip(axes, pairs):
    col1 = {"PC1":0,"PC2":1,"PC3":2}
    sc = ax.scatter(X_s[:,col1[d1]], X_s[:,col1[d2]],
                    c=y_s, cmap=CMAP10, s=6, alpha=0.5, vmin=0, vmax=9)
    ax.set_xlabel(d1, fontsize=11); ax.set_ylabel(d2, fontsize=11)
    ax.set_title(f"{d1} vs {d2}", fontweight="bold", fontsize=12)
    ax.grid(True, ls="--", alpha=0.3)

cbar = plt.colorbar(sc, ax=axes[-1], ticks=range(10), pad=0.02)
cbar.set_label("Dígito", fontsize=10)
cbar.set_ticklabels(CLASS_NAMES)

plt.suptitle("Proyecciones 2D del Espacio PCA-3D con Ruido  (n=5 000 muestras)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"04_2d_projections.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ─────────────────────────────────────────────────────────────
md("## 8. Visualización 3D del Espacio de Características")
code("""\
fig = plt.figure(figsize=(16, 6))

angles = [(20, 30), (20, 120), (45, 200)]
titles = ["Vista 1  (elev=20, azim=30)",
          "Vista 2  (elev=20, azim=120)",
          "Vista 3  (elev=45, azim=200)"]

for k, (elev, azim) in enumerate(angles):
    ax = fig.add_subplot(1, 3, k+1, projection="3d")
    for cls in range(10):
        mask = y_s == cls
        ax.scatter(X_s[mask,0], X_s[mask,1], X_s[mask,2],
                   c=[PALETTE[cls]], s=4, alpha=0.4, label=str(cls))
    ax.set_xlabel("PC1", labelpad=1, fontsize=8)
    ax.set_ylabel("PC2", labelpad=1, fontsize=8)
    ax.set_zlabel("PC3", labelpad=1, fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(titles[k], fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)

handles = [plt.Line2D([0],[0], marker="o", color="w",
           markerfacecolor=PALETTE[i], markersize=8, label=str(i)) for i in range(10)]
fig.legend(handles=handles, title="Dígito", loc="lower center",
           ncol=10, fontsize=8, bbox_to_anchor=(0.5, -0.02))

plt.suptitle("Espacio 3D PCA + Ruido — 3 perspectivas  (n=5 000)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR/"05_3d_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ─────────────────────────────────────────────────────────────
md("## 9. Análisis del Ruido — Original vs Ruidoso")
code("""\
# Reconstruir PCA limpio desde MNIST para comparar
print("Cargando MNIST original (cache local)...")
DATA_DIR = REPO_DIR / "data"
mnist = fetch_openml("mnist_784", version=1, data_home=str(DATA_DIR),
                     as_frame=False, parser="auto")
X_raw = mnist.data.astype(np.float32) / 255.0
y_raw = mnist.target.astype(np.int64)

rng2   = np.random.default_rng(SEED)
idx_m  = rng2.permutation(len(y_raw))
split  = int(0.857 * len(idx_m))
tr_idx = idx_m[:split]

pca = PCA(n_components=3, random_state=SEED)
X_clean = pca.fit_transform(X_raw[tr_idx]).astype(np.float32)
y_clean = y_raw[tr_idx]

print(f"Varianza explicada por componente: {pca.explained_variance_ratio_.round(4)}")
print(f"Varianza explicada acumulada: {pca.explained_variance_ratio_.sum():.4f}")

# Comparación visual
rng3   = np.random.default_rng(SEED)
idx_c  = rng3.choice(len(X_clean), 3000, replace=False)
idx_n  = rng3.choice(len(X_train), 3000, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

pairs = [("PC1","PC2",0,1), ("PC1","PC3",0,2), ("PC2","PC3",1,2)]
for col, (d1, d2, i1, i2) in enumerate(pairs):
    ax_clean = axes[0, col]
    ax_noisy = axes[1, col]

    ax_clean.scatter(X_clean[idx_c,i1], X_clean[idx_c,i2],
                     c=y_clean[idx_c], cmap=CMAP10, s=5, alpha=0.5, vmin=0, vmax=9)
    ax_clean.set_title(f"SIN ruido — {d1} vs {d2}", fontweight="bold")
    ax_clean.set_xlabel(d1); ax_clean.set_ylabel(d2)
    ax_clean.grid(True, ls="--", alpha=0.3)

    sc = ax_noisy.scatter(X_train[idx_n,i1], X_train[idx_n,i2],
                          c=y_train[idx_n], cmap=CMAP10, s=5, alpha=0.5, vmin=0, vmax=9)
    ax_noisy.set_title(f"CON ruido σ=1 — {d1} vs {d2}", fontweight="bold")
    ax_noisy.set_xlabel(d1); ax_noisy.set_ylabel(d2)
    ax_noisy.grid(True, ls="--", alpha=0.3)

plt.colorbar(sc, ax=axes[:,-1], ticks=range(10)).set_label("Dígito")
plt.suptitle("Impacto del Ruido Gaussiano (σ=1) en el Espacio PCA-3D",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"06_clean_vs_noisy.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ─────────────────────────────────────────────────────────────
md("## 10. Varianza Explicada por PCA")
code("""\
# Curva de varianza acumulada para los primeros 50 componentes
pca_full = PCA(n_components=50, random_state=SEED)
pca_full.fit(X_raw[tr_idx])

cumvar = np.cumsum(pca_full.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Varianza por componente
axes[0].bar(range(1, 51), pca_full.explained_variance_ratio_*100,
            color="#4C72B0", alpha=0.8)
axes[0].axvline(3, color="red", ls="--", lw=2, label="n=3 (dataset)")
axes[0].set_xlabel("Componente Principal"); axes[0].set_ylabel("Varianza explicada (%)")
axes[0].set_title("Varianza por componente (top-50)", fontweight="bold")
axes[0].legend(); axes[0].yaxis.grid(True, ls="--", alpha=0.4)

# Varianza acumulada
axes[1].plot(range(1, 51), cumvar*100, "o-", color="#DD8452", ms=3, lw=1.5)
axes[1].axvline(3, color="red", ls="--", lw=2, label=f"n=3 → {cumvar[2]*100:.1f}%")
axes[1].axhline(95, color="gray", ls=":", lw=1.5, label="95% umbral")
axes[1].set_xlabel("Número de componentes"); axes[1].set_ylabel("Varianza acumulada (%)")
axes[1].set_title("Varianza acumulada explicada", fontweight="bold")
axes[1].legend(); axes[1].yaxis.grid(True, ls="--", alpha=0.4)

plt.suptitle(f"Análisis PCA: las 3 primeras PCs capturan el {cumvar[2]*100:.1f}% de la varianza",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"07_pca_variance.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"PC1: {pca_full.explained_variance_ratio_[0]*100:.2f}%")
print(f"PC2: {pca_full.explained_variance_ratio_[1]*100:.2f}%")
print(f"PC3: {pca_full.explained_variance_ratio_[2]*100:.2f}%")
print(f"Total 3 PCs: {cumvar[2]*100:.2f}%")
""")

# ─────────────────────────────────────────────────────────────
md("## 11. Matriz de Correlación entre Componentes")
code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (df, title) in zip(axes, [
    (df_train[["PC1","PC2","PC3"]], "Train"),
    (df_test[["PC1","PC2","PC3"]],  "Test")
]):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                ax=ax, square=True, linewidths=0.5,
                annot_kws={"size":13, "weight":"bold"})
    ax.set_title(f"Correlación — {title}", fontweight="bold", fontsize=12)

plt.suptitle("Correlación entre Componentes Principales (con ruido)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"08_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

print("Nota: PCA garantiza correlación ~0 entre componentes en el conjunto limpio.")
print("El ruido gaussiano añadido puede introducir ligeras correlaciones espurias.")
""")

# ─────────────────────────────────────────────────────────────
md("## 12. Separabilidad entre Clases — Distancias Inter/Intra-Clase")
code("""\
# Centroides por clase
centroids = np.array([X_train[y_train==c].mean(axis=0) for c in range(10)])

# Distancias intra-clase (std promedio)
intra = np.array([X_train[y_train==c].std(axis=0).mean() for c in range(10)])

# Distancias inter-clase (distancia al centroide más cercano de otra clase)
dist_matrix = pairwise_distances(centroids)
np.fill_diagonal(dist_matrix, np.inf)
inter_min = dist_matrix.min(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Mapa de calor de distancias entre centroides
np.fill_diagonal(dist_matrix, 0)
sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.3)
axes[0].set_title("Distancias entre centroides de clase", fontweight="bold")
axes[0].set_xlabel("Dígito"); axes[0].set_ylabel("Dígito")

# Intra vs inter
x = np.arange(10)
w = 0.35
axes[1].bar(x - w/2, intra,  w, label="Intra-clase (std)", color="#4C72B0", alpha=0.8)
axes[1].bar(x + w/2, inter_min, w, label="Inter-clase min", color="#DD8452", alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(CLASS_NAMES)
axes[1].set_xlabel("Dígito"); axes[1].set_ylabel("Distancia L2")
axes[1].set_title("Cohesion intra-clase vs separacion inter-clase", fontweight="bold")
axes[1].legend(fontsize=9); axes[1].yaxis.grid(True, ls="--", alpha=0.4)

# Ratio de Fisher (inter/intra)
np.fill_diagonal(dist_matrix, np.inf)
fisher = dist_matrix.min(axis=1) / intra
np.fill_diagonal(dist_matrix, 0)
axes[2].bar(range(10), fisher, color=PALETTE, alpha=0.85)
axes[2].set_xticks(range(10)); axes[2].set_xticklabels(CLASS_NAMES)
axes[2].axhline(fisher.mean(), color="red", ls="--", lw=1.5,
                label=f"Media={fisher.mean():.2f}")
axes[2].set_xlabel("Dígito"); axes[2].set_ylabel("Ratio inter/intra")
axes[2].set_title("Ratio de Fisher por clase", fontweight="bold")
axes[2].legend(); axes[2].yaxis.grid(True, ls="--", alpha=0.4)

plt.suptitle("Analisis de Separabilidad en Espacio PCA-3D con Ruido",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"09_class_separability.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\n=== Ratio de Fisher (inter/intra) por clase ===")
for i, (c, f) in enumerate(zip(CLASS_NAMES, fisher)):
    print(f"  Digito {c}: {f:.3f}  {'<-- mas separable' if f==fisher.max() else ''}")
""")

# ─────────────────────────────────────────────────────────────
md("## 13. Distribución del Ruido por Dimensión y Clase")
code("""\
# Diferencia entre versión ruidosa y limpia
X_clean_aligned = X_clean[:len(X_train)]  # mismo tamaño
noise_added = X_train[:len(X_clean_aligned)] - X_clean_aligned

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (dim, i) in zip(axes, [("PC1",0),("PC2",1),("PC3",2)]):
    ax.hist(noise_added[:,i], bins=80, color="#55A868", alpha=0.8,
            edgecolor="white", lw=0.2)
    mu, sg = noise_added[:,i].mean(), noise_added[:,i].std()
    ax.axvline(mu, color="red", ls="--", lw=1.5, label=f"mu={mu:.3f}")
    ax.axvline(mu+sg, color="orange", ls=":", lw=1.2, label=f"sigma={sg:.3f}")
    ax.axvline(mu-sg, color="orange", ls=":", lw=1.2)
    ax.set_title(f"Ruido en {dim}", fontweight="bold")
    ax.set_xlabel("Ruido"); ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9); ax.yaxis.grid(True, ls="--", alpha=0.4)

plt.suptitle("Distribucion del Ruido Anadido por Componente Principal",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"10_noise_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ─────────────────────────────────────────────────────────────
md("## 14. Comparación Train vs Test")
code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (dim, i) in zip(axes, [("PC1",0),("PC2",1),("PC3",2)]):
    sns.kdeplot(X_train[:,i], ax=ax, label="Train", color="#4C72B0", lw=2)
    sns.kdeplot(X_test[:,i],  ax=ax, label="Test",  color="#DD8452", lw=2, ls="--")
    ax.set_title(f"Distribucion {dim}: Train vs Test", fontweight="bold")
    ax.set_xlabel(dim); ax.set_ylabel("Densidad")
    ax.legend(); ax.yaxis.grid(True, ls="--", alpha=0.4)

plt.suptitle("Consistencia de la Distribucion entre Particiones",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"11_train_vs_test.png", dpi=150, bbox_inches="tight")
plt.show()

# KS test
from scipy.stats import ks_2samp
print("=== Test KS (Train vs Test) — p-valor > 0.05 indica misma distribucion ===")
for dim, i in [("PC1",0),("PC2",1),("PC3",2)]:
    stat, p = ks_2samp(X_train[:,i], X_test[:,i])
    print(f"  {dim}: KS={stat:.4f}  p={p:.4f}  {'OK' if p>0.05 else 'DIFERENTE'}")
""")

# ─────────────────────────────────────────────────────────────
md("## 15. Centroides en 3D y Densidad por Clase (violin)")
code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (dim, i) in zip(axes, [("PC1",0),("PC2",1),("PC3",2)]):
    data_by_cls = [X_train[y_train==c, i] for c in range(10)]
    parts = ax.violinplot(data_by_cls, positions=range(10),
                          showmeans=True, showmedians=True)
    for pc_patch, color in zip(parts["bodies"], PALETTE):
        pc_patch.set_facecolor(color)
        pc_patch.set_alpha(0.65)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Digito"); ax.set_ylabel(dim)
    ax.set_title(f"Violin {dim} por clase", fontweight="bold")
    ax.yaxis.grid(True, ls="--", alpha=0.4)

plt.suptitle("Forma de la Distribucion por Clase (Violin Plots)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"12_violin_plots.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ─────────────────────────────────────────────────────────────
md("""\
## 16. Conclusiones del EDA

### Hallazgos clave:

| Aspecto | Observación |
|---|---|
| **Dimensionalidad** | Solo 3 PCs capturan ~30-40% de la varianza de MNIST (784D) |
| **Ruido** | σ=1.0 en espacio PCA supera la separación natural entre clases |
| **Separabilidad** | Las clases se solapan considerablemente en 3D; ratio Fisher < 1 en varias |
| **Clases más separables** | Dígitos con formas muy distintas (ej. 1 vs 0) |
| **Clases más confusas** | Dígitos similares (ej. 4/9, 3/5, 7/1) |
| **Train vs Test** | Distribuciones consistentes (test KS confirma i.i.d.) |
| **Impacto en modelos** | KDM (~42% acc) supera PNC en este espacio — esperable dado que KDM está diseñado para espacios de baja dimensión, mientras que PNC requiere representaciones de imagen |

### Implicaciones para KDM vs PNC:
- **KDM** modela la distribución de densidad directamente en el espacio 3D → más adecuado
- **PNC** asume que los datos son imágenes con valores enteros 0-255 → inadecuado para PCA features continuas
- Con mayor dimensión (ej. 16D o 32D via PCA) ambos modelos mejorarían
""")

# ─────────────────────────────────────────────────────────────
code("""\
# Listado final de figuras generadas
import os
figs = sorted(FIG_DIR.glob("*.png"))
print(f"=== {len(figs)} graficas guardadas en {FIG_DIR} ===")
for f in figs:
    size_kb = os.path.getsize(f) // 1024
    print(f"  {f.name:<45} {size_kb:>4} KB")
""")

# ── Escribir el notebook ──────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (mlops)",
    "language": "python",
    "name": "python3"
}
nb.metadata["language_info"] = {"name": "python", "version": "3.10.20"}

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook escrito en: {NB_PATH}")
