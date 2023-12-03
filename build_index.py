import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numba as nb
import os
import heapq

@nb.njit()
def _distance(data_matrix, node_data):
    """
    Compute distances using Numba for JIT compilation and parallel execution.
    """
    return np.linalg.norm(data_matrix - node_data)

@nb.njit(parallel=True)
def _distances(data_matrix, node_data):
    """
    Compute distances using Numba for JIT compilation and parallel execution.
    """
    # Pre-allocate an array for distances
    distances = np.empty(data_matrix.shape[0], dtype=np.float32)
    
    # Parallel loop to compute distances
    for i in nb.prange(data_matrix.shape[0]):
        distances[i] = _distance(data_matrix[i], node_data)

    return distances

def save_to_parquet(df, file_name):
    """
    Saves the DataFrame to a Parquet file.
    """
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_name)

class HNSWNode:
    """
    Represents a node in the HNSW graph.
    """
    def __init__(self, data, id):
        self.data = data
        self.id = id
        self.neighbors = {}  # Neighbors in the layer

class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) Index for efficient Approximate Nearest Neighbor (ANN) search.
    
    Attributes:
        dim (int): Dimensionality of the data points.
        M (int): Maximum number of edges per node (controls the graph connectivity).
        ef_construction (int): Size of the dynamic candidate list used during insertion (higher values lead to more accurate but slower construction).
        max_elements (int): Maximum number of elements the index can hold.
        nodes (list): List of nodes in the HNSW graph.
        data_matrix (numpy.ndarray): Matrix of data points added to the index.
        max_layer (int): Maximum number of layers in the HNSW graph. Logarithmically dependent on `max_elements`.
    
    The HNSW algorithm creates a layered graph structure for efficient search in high-dimensional spaces.
    """
    def __init__(self, dim, M=16, ef_construction=200, max_elements=10000):
        self.dim = dim  # Dimensionality of the data points.
        self.M = M  # Maximum number of connections per node in the graph.
        self.ef_construction = ef_construction  # Size of the dynamic candidate list during the construction phase.
        self.max_elements = max_elements  # Maximum capacity of the index.
        self.nodes = []  # Initializes an empty list to store the nodes.
        self.data_matrix = None  # Data matrix to store the vectors.
        # Set the maximum layer of the graph based on the maximum elements, using a logarithmic scale.
        self.max_layer = int(np.log2(max_elements)) if max_elements > 0 else 0

        self.enter_point = None  # Entry point for the graph

    def add_items(self, data):
        """
        Adds multiple items (data points) to the HNSW index.

        Args:
            data (iterable): An iterable of data points to be added to the index.
        
        This method iterates through the provided data points and inserts each into the index.
        """
        for d in tqdm(data):
            if len(self.nodes) < self.max_elements:
                self._insert_node(np.array(d))  # Inserts each data point into the index.

    def _insert_node(self, data):
        """
        Inserts a single data point into the HNSW index.

        Args:
            data (list or numpy.ndarray): A single data point to be inserted into the index.
        
        This private method handles the insertion of a single data point into the graph,
        updating the data matrix and assigning neighbors to the new node.
        """
        node_id = len(self.nodes)  # Assigns a unique ID to the new node, based on its position in the list.
        node = HNSWNode(data, node_id)  # Creates a new node instance.
        self.nodes.append(node)  # Adds the new node to the list of nodes.

        # Update the data matrix with the new data point.
        if self.data_matrix is None:
            self.data_matrix = np.array([data])
        else:
            self.data_matrix = np.vstack([self.data_matrix, data])

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
        Greedy search to find the closest node to the target node in a given layer.
    
        Args:
            target_node (HNSWNode): The target node we are finding neighbors for.
            entry_node (HNSWNode): The entry node from where the search starts.
            layer (int): The layer of the graph at which the search is conducted.
    
        Returns:
            HNSWNode: The closest node to the target node in this layer.
        """
        current_node = entry_node
        while True:
            closest_node = current_node
            neighbor_ids = current_node.neighbors.get(layer, [])
    
            if neighbor_ids:
                neighbor_nodes = [self.nodes[neighbor_id] for neighbor_id in neighbor_ids]
                neighbor_data = np.array([node.data for node in neighbor_nodes])
                target_data = np.array(target_node.data)
    
                # Vectorized distance computation
                distances = _distances(neighbor_data, target_data)
                min_dist_index = np.argmin(distances)
    
                if distances[min_dist_index] < _distance(self.nodes[closest_node.id].data, target_node.data):
                    closest_node = neighbor_nodes[min_dist_index]
    
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
        initial_distance = _distance(target_node.data, current_node.data)
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
                    neighbor_node = self.nodes[neighbor_id]
                    neighbor_dist = _distance(target_node.data, neighbor_node.data)

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
        Performs a k-NN (k-nearest neighbors) query on the HNSW index.

        Args:
            query_data (list or numpy.ndarray): The query data point.
            k (int): The number of nearest neighbors to find.
            ef (int, optional): The size of the dynamic candidate list during the search. Defaults to max(k, ef_construction).

        Returns:
            tuple: A tuple containing two lists - IDs of the k nearest neighbors and their corresponding distances.

        This method performs a layered search for the nearest neighbors of the given query point,
        starting from the top layer and gradually moving to the lower layers.
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
    
    # For 'data' column, ensure the data type is appropriate.
    # If it's a vector or complex type, you might need to handle it differently.
    edges_df['source_node_id'] = edges_df['source_node_id'].astype('int32')
    edges_df['target_node_id'] = edges_df['target_node_id'].astype('int32')
    edges_df['layer'] = edges_df['layer'].astype('int32')
    edges_df.sort_values(by=['source_node_id', 'layer'], inplace=True)

    nodes_df.set_index('node_id')
    edges_df.set_index(['source_node_id', 'target_node_id', 'layer'])
    
    return nodes_df, edges_df

def embed_documents(docs, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(docs, show_progress_bar=True)

def create_hnsw_index(embeddings, dim=384, ef=200, M=14):
    p = HNSWIndex(dim=dim, max_elements=len(embeddings), ef_construction=ef, M=M)
    p.add_items(embeddings)
    return p

def search_similar_documents(query, index, docs, top_k=1):
    query_embedding = embed_documents([query])[0]
    labels, distances = index.knn_query(query_embedding, k=top_k)
    return [docs[i] for i in labels]

def from_list(list, folder, max_chunk_chars=4000, precomputed_embeddings=precomputed_embeddings):
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
