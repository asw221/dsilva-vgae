
"""
vgae_train.py
-------------
Variational Graph Autoencoder for 50+ graphs ranging from 10k to 2M nodes.

Key features
  - Programmatic graph discovery from a directory of CSVs (or other formats)
  - Approximate kNN via faiss (fast at scale) with sklearn fallback
  - Per-graph memory estimation and OOM guard
  - Automatic batch_size=1 for large graphs
  - Gradient checkpointing to reduce activation memory
  - Streaming DataLoader: graphs are built on-demand, not all held in RAM

Requirements
    pip install torch torch_geometric scikit-learn numpy pandas
    pip install faiss-cpu          # or faiss-gpu if you have a CUDA GPU
    pip install psutil             # for memory reporting
"""

import gc
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import StandardScaler

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    from sklearn.neighbors import kneighbors_graph
    FAISS_AVAILABLE = False
    warnings.warn(
        "faiss not found — falling back to sklearn kneighbors_graph. "
        "This will be very slow for graphs with >100k nodes. "
        "Install with: pip install faiss-cpu",
        RuntimeWarning,
    )


# -------------------------------------------------------------------
# CONFIG  — edit this block, leave everything else alone
# -------------------------------------------------------------------

CONFIG = dict(
    # Directory containing your graph source files.
    # Each file = one graph. Supported: .csv, .parquet, .npy, .npz
    data_dir = "data/raw",

    coord_columns = ["x", "y"],
    # Columns to drop before building features (ids, ...)
    drop_columns = ["ID"],

    # Columns to one-hot encode (leave empty list [] to skip)
    categorical_columns = [],

    # Number of nearest neighbours per node when building the graph
    k = 8,

    # Graphs with more nodes than this trigger a warning (not an error).
    # Set to None to disable.
    max_nodes_warn = 1_500_000,

    # Model hyperparameters
    hidden_channels = 64,
    latent_dim      = 32,

    # Training
    epochs          = 200,
    lr              = 1e-3,
    val_fraction    = 0.05,
    test_fraction   = 0.05,

    # Graphs with more nodes than this are trained with batch_size=1
    # automatically regardless of the setting below.
    large_graph_threshold = 2_000_000,  # 350_000,
    batch_size            = 1,          # for small/medium graphs

    # Where to save outputs
    output_dir   = "out",
    output_model = "vgae_model_1.pt",
    loss_history_file = "loss_history.csv"
)


# -------------------------------------------------------------------
# MEMORY ESTIMATION
# -------------------------------------------------------------------

def estimate_memory_gb(num_nodes, num_features, k, hidden, latent):
    """
    Rough per-graph memory breakdown in GB.
    Returns a dict so callers can inspect individual components.
    """
    num_edges     = num_nodes * k * 2           # symmetric graph, approx
    x_gb          = num_nodes * num_features * 4 / 1e9
    edge_index_gb = num_edges * 2 * 8 / 1e9    # int64
    h_gb          = num_nodes * hidden * 4 / 1e9
    mu_logstd_gb  = num_nodes * latent * 4 * 2 / 1e9
    grad_gb       = (h_gb + mu_logstd_gb) * 2.5  # rule of thumb

    return dict(
        x_gb=x_gb,
        edge_index_gb=edge_index_gb,
        activations_gb=h_gb + mu_logstd_gb,
        gradients_gb=grad_gb,
        total_gb=x_gb + edge_index_gb + h_gb + mu_logstd_gb + grad_gb,
    )


def report_memory(num_nodes, num_features, k, hidden, latent):
    mem       = estimate_memory_gb(num_nodes, num_features, k, hidden, latent)
    avail_ram = psutil.virtual_memory().available / 1e9
    avail_gpu = 0.0
    if torch.cuda.is_available():
        props     = torch.cuda.get_device_properties(0)
        avail_gpu = (props.total_memory - torch.cuda.memory_allocated()) / 1e9

    print(f"  Memory estimate │ nodes={num_nodes:>10,}  features={num_features}")
    print(f"    x:             {mem['x_gb']:.2f} GB")
    print(f"    edge_index:    {mem['edge_index_gb']:.2f} GB")
    print(f"    activations:   {mem['activations_gb']:.2f} GB")
    print(f"    gradients:     {mem['gradients_gb']:.2f} GB")
    print(f"    -- total:      {mem['total_gb']:.2f} GB")
    print(f"    available RAM: {avail_ram:.1f} GB"
          + (f"   GPU: {avail_gpu:.1f} GB" if avail_gpu else ""))

    if mem["total_gb"] > avail_ram * 0.8:
        warnings.warn(
            f"Estimated memory ({mem['total_gb']:.1f} GB) exceeds 80 % of "
            f"available RAM ({avail_ram:.1f} GB). Consider reducing k.",
            ResourceWarning,
        )
    return mem["total_gb"]


# -------------------------------------------------------------------
# GRAPH CONSTRUCTION
# -------------------------------------------------------------------

def build_knn_edge_index(x_np, k, batch_size = 10_000):
    """
    Build a symmetric kNN graph, return edge_index [2, E].
    Uses faiss when available (much faster for large N), sklearn otherwise.
    """
    assert x_np.ndim == 2, f"Expected 2D array, got shape {x_np.shape}"
    assert np.isfinite(x_np).all(), "NaN/Inf in input"    
    n = x_np.shape[0]

    if FAISS_AVAILABLE:
        # faiss.omp_set_num_threads(1)
        xf = np.ascontiguousarray(x_np, dtype=np.float32)
        # faiss.normalize_L2(xf)                   # cosine similarity
        # index = faiss.IndexFlatIP(xf.shape[1])
        index = faiss.IndexFlatL2(xf.shape[1])   # squared Euclidean
        index.add(xf)
        all_I = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            _, I_batch = index.search(xf[start:end], k + 1)
            all_I.append(I_batch)
        I = np.concatenate(all_I, axis=0)
        # _, I = index.search(xf, k + 1)           # k+1: first col is self
        rows = np.repeat(np.arange(n), k)
        cols = I[:, 1:].ravel()
    else:
        A    = kneighbors_graph(x_np, n_neighbors=k,
                                mode="connectivity", include_self=False)
        A    = (A + A.T).tocoo()
        return torch.from_numpy(np.stack([A.row, A.col])).long()

    # Symmetrise and deduplicate
    src   = np.concatenate([rows, cols])
    dst   = np.concatenate([cols, rows])
    edges = np.unique(np.stack([src, dst], axis=1), axis=0)
    return torch.from_numpy(edges.T).long()


def preprocess_dataframe(df, cfg):
    """Drop, one-hot, impute, scale -> float32 numpy array."""
    drop = [c for c in cfg["drop_columns"] if c in df.columns]
    ccols = [c for c in cfg["coord_columns"] if c in df.columns]
    drop = drop + ccols

    coords = df[ ccols ].values.astype(float)
    if drop:
        df = df.drop( columns = drop )

    cats = [c for c in cfg["categorical_columns"] if c in df.columns]
    if cats:
        df = pd.get_dummies(df, columns=cats)

    df = df.fillna(df.median(numeric_only=True))
    vals = StandardScaler().fit_transform(df.values.astype(float))
    return vals, coords


def load_file(path):
    """Load a graph source file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".npy":
        return pd.DataFrame(np.load(path))
    elif suffix == ".npz":
        arr = np.load(path)
        return pd.DataFrame(arr[list(arr.keys())[0]])
    else:
        raise ValueError(
            f"Unsupported format: {suffix}. "
            "Add a loader for it in load_file()."
        )


def file_to_pyg(path, cfg):
    """End-to-end: file on disk -> PyG Data object."""
    print(f"\n  Loading {path.name} ...", end=" ", flush=True)
    df = load_file(path)
    print(f"{len(df):,} rows", end=" ", flush=True)

    x_np, coords = preprocess_dataframe(df, cfg)
    del df
    gc.collect()

    num_nodes, num_features = x_np.shape

    if cfg.get("max_nodes_warn") and num_nodes > cfg["max_nodes_warn"]:
        warnings.warn(
            f"{path.name}: {num_nodes:,} nodes exceeds max_nodes_warn="
            f"{cfg['max_nodes_warn']:,}. Training will be slow.",
            ResourceWarning,
        )

    report_memory(num_nodes, num_features, cfg["k"],
                  cfg["hidden_channels"], cfg["latent_dim"])

    print(f"  Building kNN graph (k={cfg['k']}) ...", end=" ", flush=True)
    edge_index = build_knn_edge_index( coords, cfg["k"] )
    x          = torch.from_numpy(x_np).float()
    del x_np
    gc.collect()

    print(f"edges={edge_index.shape[1]:,}  done")
    return Data( x = x, edge_index = edge_index )


# -------------------------------------------------------------------
# ENCODER  (with optional gradient checkpointing)
# -------------------------------------------------------------------

class VGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 use_checkpointing=False):
        super().__init__()
        self.conv_shared       = GCNConv(in_channels,     hidden_channels)
        self.conv_mu           = GCNConv(hidden_channels, out_channels)
        self.conv_logstd       = GCNConv(hidden_channels, out_channels)
        self.use_checkpointing = use_checkpointing

    def _shared(self, x, edge_index):
        return F.relu(self.conv_shared(x, edge_index))

    def forward(self, x, edge_index):
        if self.use_checkpointing:
            # Recompute activations on the backward pass instead of caching
            # them. Costs ~30 % extra compute, saves ~40 % activation memory.
            h = checkpoint(self._shared, x, edge_index, use_reentrant=False)
        else:
            h = self._shared(x, edge_index)
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------

# Simple in-memory cache for graphs below large_graph_threshold.
# Large graphs are rebuilt each epoch to avoid holding GBs in RAM.
_graph_cache: dict = {}


def train_vgae(graph_paths, cfg, device = torch.device("cpu")):
    """
    Train VGAE across all discovered graphs.

    Small/medium graphs (<= large_graph_threshold nodes) are cached in RAM
    after the first epoch. Large graphs are loaded fresh each epoch to avoid
    exhausting memory.

    Returns:
        model       : trained VGAE (moved to CPU)
        train_losses: list of (epoch, avg_loss) tuples
    """
    ## device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  {len(graph_paths)} graphs\n")

    # Peek at first file to get feature dimension without caching it yet
    print("Inspecting first file to infer feature dimension ...")
    df_peek     = load_file(graph_paths[0])
    x_peek, _   = preprocess_dataframe(df_peek, cfg)
    in_channels = x_peek.shape[1]
    del df_peek, x_peek
    gc.collect()
    print(f"  in_channels = {in_channels}\n")

    use_ckpt  = (device.type == "cuda")
    encoder   = VGAEEncoder(in_channels, cfg["hidden_channels"],
                            cfg["latent_dim"], use_checkpointing=use_ckpt)
    model     = VGAE(encoder).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    transform = RandomLinkSplit(
        num_val  = cfg["val_fraction"],
        num_test = cfg["test_fraction"],
        is_undirected=True,
        add_negative_train_samples=False,
    )

    train_losses = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_losses = []

        for path in graph_paths:
            name = path.name

            # Load from cache or disk
            if name in _graph_cache:
                graph = _graph_cache[name]
            else:
                graph = file_to_pyg(path, cfg)
                if graph.num_nodes <= cfg["large_graph_threshold"]:
                    _graph_cache[name] = graph   # cache small/medium graphs

            train_g, val_g, _ = transform(graph)
            is_large = graph.num_nodes > cfg["large_graph_threshold"]
            bs       = 1 if is_large else cfg["batch_size"]

            loader = DataLoader([train_g], batch_size=bs, shuffle=False)

            for batch in loader:
                batch = batch.to(device)
                optimiser.zero_grad()
                z    = model.encode(batch.x, batch.edge_index)
                loss = (
                    model.recon_loss(z, batch.edge_index)
                    + (1 / batch.num_nodes) * model.kl_loss()
                )
                loss.backward()
                optimiser.step()
                epoch_losses.append(loss.item())

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Drop large graphs from local scope; they are not cached
            if is_large:
                del graph, train_g, val_g
                gc.collect()

        avg_loss = float(np.mean(epoch_losses))
        train_losses.append((epoch, avg_loss))

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>4}  avg_loss={avg_loss:.4f}")

    return model.cpu(), train_losses


# -------------------------------------------------------------------
# DOWNSTREAM UTILITIES
# -------------------------------------------------------------------

@torch.no_grad()
def get_embeddings(model, data, device="cpu"):
    """Return latent mu vectors [N, latent_dim] for one graph."""
    model.eval()
    data = data.to(device)
    z    = model.encode(data.x, data.edge_index)
    return z.cpu().numpy()


@torch.no_grad()
def generate_graph(model, num_nodes, latent_dim, threshold=0.5, device="cpu"):
    """Sample a new graph from the prior N(0, I)."""
    model.eval()
    z   = torch.randn(num_nodes, latent_dim).to(device)
    adj = torch.sigmoid(z @ z.T).cpu().numpy()
    adj = (adj > threshold).astype(float)
    np.fill_diagonal(adj, 0)
    return adj

def save_loss_history(train_losses, cfg):
    """
    Save per-epoch average loss to a CSV file
    Output columsn: epoch, avg_loss
    File location : <output_dir>/<loss_history_file>
    """
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir( parents = True, exist_ok = True )
    out_path = out_dir / cfg["loss_history_file"]
    df = pd.DataFrame(train_losses, columns = ["epoch", "avg_loss"])
    df.to_csv(out_path, index = False)
    print(f"Loss history saved -> {out_path}  ({len(df)}) rows")


def export_all_embeddings(model, graph_paths, cfg):
    """
    Generate and save embeddings for every graph as a CSV file
    Each CSV contains one row per node with columns:
      z_0, z_1, ..., z_{latent_dim-1}

    Files are written to <output_dir>/embeddings/<graph_stem>.csv
    Graphs already in _graph_cache are reused; others are loaded
    from disk.
    """
    out_dir = Path(cfg["output_dir"]) / "embeddings"
    out_dir.mkdir( parents = True, exist_ok = True )

    model.eval()
    cols = [ f"z_{i}" for i in range(cfg["latent_dim"]) ]
    for path in graph_paths:
        graph = _graph_cache.get(path.name) or file_to_pyg(path, cfg)

        Z = get_embeddings(model, graph)  # [N, latent_dim]
        out_path = out_dir / f"{path.stem}.csv"
        pd.DataFrame(Z, colums = cols).to_csv(out_path, index = False)
        print(f"  Embeddings -> {out_path}  shape={Z.shape}")



# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    cfg = CONFIG

    data_dir  = Path(cfg["data_dir"])
    SUPPORTED = {".csv", ".parquet", ".npy", ".npz"}

    if not data_dir.exists():
        raise FileNotFoundError(
            f"data_dir '{data_dir}' not found. "
            "Set CONFIG['data_dir'] to the folder containing your graph files."
        )

    graph_paths = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    if not graph_paths:
        raise FileNotFoundError(
            f"No supported files {SUPPORTED} found in '{data_dir}'."
        )

    print(f"Found {len(graph_paths)} graph files:")
    for p in graph_paths:
        print(f"  {p.name:<40} {p.stat().st_size / 1e6:>8.1f} MB")

    model, losses = train_vgae(graph_paths, cfg)

    torch.save(model.state_dict(), cfg["output_model"])
    print(f"\nModel saved -> {cfg['output_model']}")

    print("\nSaving loss history and exporting graph embeddings")
    save_loss_history(losses, cfg)
    export_all_embeddings(model, graph_paths, cfg)

    # Sanity check: embeddings for the first graph
    print("\nGenerating embeddings for first graph ...")
    g0 = file_to_pyg(graph_paths[0], cfg)
    Z  = get_embeddings(model, g0)
    print(f"Embeddings shape: {Z.shape}")
    
