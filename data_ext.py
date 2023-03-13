import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import time

root = 'E:\data\european_cities_network'

len_e = []
max_e = 0
min_e = 2
start = time.time()
data = []

for dirpath, dirnames, filenames in os.walk(root):
    for filepath in filenames:

        file = os.path.join(dirpath, filepath)

        gdf = gpd.read_parquet(file)
        G = nx.Graph()

        edges = gdf[['u', 'v', 'level']]
        G.add_edges_from(list(edges[['u', 'v']].itertuples(index=False, name=None)), attr=list(edges.level))
        # largest_cc = max(nx.connected_components(G), key=len)
        # S = G.subgraph(largest_cc).copy()
        # pos = nx.spring_layout(S)

        L_n = nx.normalized_laplacian_matrix(G)
        e = np.linalg.eigvals(L_n.todense())
        # e = np.around(e, 1)

        len_e.append(len(e))
        if max_e < max(e):
            max_e = max(e)
        if min_e > min(e):
            min_e = min(e)

        f1 = ((-0.01 < e) & (e < 0.2)).sum()
        f2 = ((0.2 < e) & (e < 0.4)).sum()
        f3 = ((0.4 < e) & (e < 0.6)).sum()
        f4 = ((0.6 < e) & (e < 0.8)).sum()
        f5 = ((0.8 < e) & (e < 1.0)).sum()
        f6 = ((1.0 < e) & (e < 1.2)).sum()
        f7 = ((1.2 < e) & (e < 1.4)).sum()
        f8 = ((1.4 < e) & (e < 1.6)).sum()
        f9 = ((1.6 < e) & (e < 1.8)).sum()
        f10 = ((1.8 < e) & (e < 2.01)).sum()

        data.append([filepath.split('.')[0], f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
        # break

df = pd.DataFrame(data, columns=['Area', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10'], dtype=str)

df.to_csv('data.csv')
end = time.time()
print(end - start, 's')
# print(max(len_e))
# print(min(len_e))
# print(max_e)
# print(min_e)
# print(list_e)