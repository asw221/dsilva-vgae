
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from scipy.sparse import coo_matrix


df = pd.read_csv("../data/raw/cells_pat2.csv")


# 1. Extract (x, y) coordinates
xy = df[[ "x", "y" ]]

## Extra examples of filtering
# # Extract with a row filter
# adults = df.loc[df["age"] >= 18, ["name", "age", "email"]]
# # Extract columns matching a pattern
# score_cols = df.filter(regex=".*_score$")

df = df.drop(columns = ["ID", "x", "y"])


# 2. Scale numerics
scaler = StandardScaler()
x_np = scaler.fit_transform( df.values.astype(float) )
x = torch.tensor( x_np, dtype=torch.float )

xy = xy.values.astype(float)

# 3. Build kNN graph 
A = kneighbors_graph(
    xy, n_neighbors = 12,
    mode = "connectivity", include_self = False
)
A = A + A.T                        # make symmetric (undirected)
A = A.tocoo()

edge_index = torch.from_numpy(
    np.stack([A.row, A.col])
).long()


# 4. Wrap in PyG Data object
data = Data( x = x, edge_index = edge_index )
print(data)  # Data(x=[N, F], edge_index=[2, E])

