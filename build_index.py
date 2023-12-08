import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numba as nb
import os
import heapq
from tqdm import tqdm
import argparse

@nb.njit()
def _distance(data_matrix, node_data):
    return np.linalg.norm(data_matrix - node_data)

@nb.njit(parallel=False, cache=True)
def _square_distance(data_matrix, node_data):
    return np.sum((data_matrix - node_data)**2)

@nb.njit(parallel=False, cache=True)
def _square_distances(data_matrix, node_data):
    # Pre-allocate an array for distances
    distances = np.empty(data_matrix.shape[0], dtype=np.float32)
    
    # Parallel loop to compute distances
    for i in nb.prange(data_matrix.shape[0]):
        distances[i] = np.sum((data_matrix[i] - node_data)**2)

    return distances

def save_to_parquet(df, file_name):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_name)

class HNSWNode:
    def __init__(self, data, id):
        self.data = data
        self.id = id
        self.neighbors = {}  # Neighbors in the layer

class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) Index for efficient Approximate Nearest Neighbor (ANN) search.

    Kind of.
    """
    def __init__(self, dim, M=16, ef_construction=200, max_elements=10000):
        self.dim = dim  # Dimensionality of the data points.
        self.M = M  # Maximum number of connections per node in the graph.
        self.ef_construction = ef_construction  # Size of the dynamic candidate list during the construction phase.
        self.max_elements = max_elements  # Maximum capacity of the index.
        self.nodes = []  # Initializes an empty list to store the nodes.
        self.data_matrix = np.zeros((max_elements, dim))  # Pre-allocated if max_elements is a good estimate
        self.data_list = []  # Temporary storage for new data points
        self.current_size = 0  # Tracks the number of data points added

        # Set the maximum layer of the graph based on the maximum elements, using a logarithmic scale.
        self.max_layer = int(np.log2(max_elements)) if max_elements > 0 else 0

        self.enter_point = None  # Entry point for the graph
    
    def add_items(self, data):
        """
        Add stuff.
        """
        for d in tqdm(data):
            if self.current_size < self.max_elements:
                self._insert_node(np.array(d))

        # Update data_matrix in batch after all insertions
        if self.data_list:
            new_data = np.array(self.data_list)
            self.data_matrix[self.current_size:self.current_size + len(new_data)] = new_data
            self.data_list = []  # Clear the temporary list
            self.current_size += len(new_data)

    def _insert_node(self, data):
        """
        Insert a single data point into the graph,
        update the data matrix and assign neighbors to the new node.
        """
        node_id = len(self.nodes)
        node = HNSWNode(data, node_id)
        self.nodes.append(node)
        self.data_list.append(data)

        node_layer = min(self.max_layer, int(-np.log(np.random.random()) / np.log(2)))
        self._greedy_insert(node, node_layer)

    def _greedy_insert(self, node, node_layer):
        if self.enter_point is None:
            self.enter_point = node
            for i in range(self.max_layer + 1):
                node.neighbors[i] = []
        else:
            current_node = self.enter_point
            for layer in range(self.max_layer, -1, -1):
                current_node = self._search_layer(node, current_node, layer)
                if layer <= node_layer:
                    neighbors = self._select_neighbors(node, current_node, layer)
                    node.neighbors[layer] = neighbors
                    for neighbor_id in neighbors:
                        self.nodes[neighbor_id].neighbors[layer].append(node.id)

    def _search_layer(self, target_node, entry_node, layer):
        """
        Find the closest node to the target node in a given layer.
        """
        current_node = entry_node
        # set the cached distance for the first time
        cached_dist = _square_distance(self.nodes[current_node.id].data, target_node.data)

        while True:
            closest_node = current_node
            neighbor_ids = current_node.neighbors.get(layer, [])
    
            if neighbor_ids:
                neighbor_nodes = [self.nodes[neighbor_id] for neighbor_id in neighbor_ids]
                neighbor_data = np.array([node.data for node in neighbor_nodes])
    
                # Vectorized square distance computation
                # as we don't need to perform full distance calculations
                distances = _square_distances(neighbor_data, target_node.data)
                min_dist_index = np.argmin(distances)
    
                if distances[min_dist_index] < cached_dist:
                    closest_node = neighbor_nodes[min_dist_index]
                    # closest was updated, update the distance
                    cached_dist = _square_distance(self.nodes[closest_node.id].data, target_node.data)
    
            if closest_node == current_node:
                break
            current_node = closest_node
    
        return current_node

    def sort_candidates(self, candidates, M):
        return sorted(candidates, key=lambda x: x[0])[:M]
    
    def _select_neighbors(self, target_node, current_node, layer, M=None):
        """
        Selects the nearest neighbors for a given node at a specified layer.
        """
        if M is None:
            M = self.M

        visited = set()  # To keep track of visited nodes
        pq = []  # Priority queue (min-heap) for nearest neighbors
        candidates = []  # List to collect candidate neighbors

        # Initialize priority queue with the current node
        initial_distance = _square_distance(target_node.data, current_node.data)
        heapq.heappush(pq, (initial_distance, current_node.id))
        visited.add(current_node.id)
        candidates.append((initial_distance, current_node.id))

        while pq:
            _, node_id = heapq.heappop(pq)
            if len(visited) > self.ef_construction:  # Limit the number of evaluations
                break

            neighbor_ids = self.nodes[node_id].neighbors.get(layer, [])
            for neighbor_id in neighbor_ids:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_dist = _square_distance(target_node.data, self.nodes[neighbor_id].data)

                    if len(pq) < M or neighbor_dist < pq[0][0]:
                        candidates.append((neighbor_dist, neighbor_id))
                        heapq.heappush(pq, (neighbor_dist, neighbor_id))
                        if len(pq) > M:
                            heapq.heappop(pq)  # Keep the queue size to M

        # Sort the candidates by distance and select the top M
        sorted_candidates = self.sort_candidates(candidates, M)
        return [node_id for _, node_id in sorted_candidates]

        
    def knn_query(self, query_data, k, ef=10):
        """
        Layered search for the nearest neighbors of the given query point,
        starting from the top layer and gradually moving to the lower layers.

        We don't really need this as we're using client side, but useful for testing.
        """
        if ef is None:
            ef = max(k, self.ef_construction)  # Ensures that ef is at least as large as k.

        if not self.nodes:
            return [], []  # Returns empty lists if the index is empty.

        query_vector = np.array(query_data)
        init_node_id = np.random.randint(0, len(self.nodes) - 1)  # Randomly selects an initial node for the search.
        current_best = [(np.linalg.norm(query_vector - self.nodes[init_node_id].data), init_node_id)]

        # Search through layers starting from the top.
        for layer in range(self.max_layer, -1, -1):
            improved = True
            while improved:
                improved = False
                candidates = set([n_id for _, n_id in current_best])  # Uses a set to avoid duplicate candidates.
                new_candidates = set()

                # Explores the neighbors of each candidate node.
                for n_id in candidates:
                    for neighbor_id in self.nodes[n_id].neighbors.get(layer, []):
                        if neighbor_id not in candidates:
                            dist = np.linalg.norm(query_vector - self.nodes[neighbor_id].data)
                            if dist < current_best[-1][0]:
                                new_candidates.add((dist, neighbor_id))
                                improved = True

                current_best.extend(new_candidates)
                current_best = sorted(current_best, key=lambda x: x[0])[:ef]  # Keeps the closest ef candidates.

        sorted_results = sorted(current_best, key=lambda x: x[0])[:k]  # Sorts the final results to get the top k neighbors.
        distances, labels = zip(*sorted_results) if sorted_results else ([], [])
        return list(labels), list(distances)

def serialize_hnsw_to_tables_v2(hnsw_index):
    """
    Our serializable format
    """
    nodes_data = []
    edges_data = []

    for node in hnsw_index.nodes:
        node_layer = min(hnsw_index.max_layer, int(-np.log(np.random.random()) / np.log(2)))
        nodes_data.append([node.id, node.data])

        for layer, neighbors in node.neighbors.items():
            for neighbor_id in neighbors:
                # Calculate and store the distance
                edges_data.append([node.id, neighbor_id, layer])

    nodes_df = pd.DataFrame(nodes_data, columns=['node_id', 'data'])
    edges_df = pd.DataFrame(edges_data, columns=['source_node_id', 'target_node_id', 'layer'])
    
     # Ensure correct data types
    nodes_df['node_id'] = nodes_df['node_id'].astype('int32')

    edges_df['source_node_id'] = edges_df['source_node_id'].astype('int32')
    edges_df['target_node_id'] = edges_df['target_node_id'].astype('int32')
    edges_df['layer'] = edges_df['layer'].astype('int32')
    
    edges_df.sort_values(by=['source_node_id', 'layer'], inplace=True)
    edges_df.set_index(['source_node_id', 'target_node_id', 'layer'])

    nodes_df.set_index('node_id')
    
    return nodes_df, edges_df

def embed_documents(docs, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(docs, show_progress_bar=True)

def create_hnsw_index(embeddings, dim=384, ef=100, M=14):
    p = HNSWIndex(dim=dim, max_elements=len(embeddings), ef_construction=ef, M=M)
    p.add_items(embeddings)
    return p

def search_similar_documents(query, index, docs, top_k=1):
    query_embedding = embed_documents([query])[0]
    labels, distances = index.knn_query(query_embedding, k=top_k)
    return [docs[i] for i in labels]

def from_list(list, folder, max_chunk_chars=4000, precomputed_embeddings=None):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    embeddings = embed_documents(list) if precomputed_embeddings is None else precomputed_embeddings
    index = create_hnsw_index(embeddings)
    # Serialize the HNSW index to a table
    (nodes, edges) = serialize_hnsw_to_tables_v2(index)
    save_to_parquet(nodes, f"{folder}/nodes.parquet")
    save_to_parquet(edges, f"{folder}/edges.parquet")

    all_docs = pd.DataFrame([{ "id": i, "text": doc } for i, doc in enumerate(list) ])
    save_to_parquet(all_docs, f"{folder}/docs.parquet")
    return nodes, edges

def from_document(path, folder, max_chunk_chars=4000, precomputed_embeddings=None):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    chunks = [""]
    
    with open(path) as f:
        for line in f.readlines():
            if len(chunks[-1]) >= max_chunk_chars:
                chunks.append("")
            chunks[-1] += line

    return from_list(chunks, folder, max_chunk_chars=max_chunk_chars, precomputed_embeddings=precomputed_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single text document and create an HNSW index.")
    parser.add_argument("path", help="Path to a document")
    parser.add_argument("--folder", default="output", help="Output folder for Parquet files")

    args = parser.parse_args()
    from_document(args.path, args.folder)
