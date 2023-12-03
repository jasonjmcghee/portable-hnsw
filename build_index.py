import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

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

    def add_items(self, data):
        """
        Adds multiple items (data points) to the HNSW index.

        Args:
            data (iterable): An iterable of data points to be added to the index.
        
        This method iterates through the provided data points and inserts each into the index.
        """
        for d in data:
            if len(self.nodes) < self.max_elements:
                self._insert_node(d)  # Inserts each data point into the index.

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

        # Assign layers to the new node, with a probabilistic approach based on the node's insertion order.
        node_layer = min(self.max_layer, int(-np.log(np.random.random()) / np.log(2)))
        for layer in range(node_layer + 1):
            node.neighbors[layer] = []  # Initializes an empty list for each layer of neighbors.
            if node_id > 0:
                neighbors = self._select_neighbors(node, layer)
                node.neighbors[layer] = neighbors[:self.M]  # Assigns the top M neighbors for this layer.
                # Update the neighbors' lists of existing nodes to include the new node.
                for neighbor_id in neighbors:
                    self.nodes[neighbor_id].neighbors.setdefault(layer, []).append(node_id)
                    if len(self.nodes[neighbor_id].neighbors[layer]) > self.M:
                        # Ensures that each node has at most M neighbors.
                        self.nodes[neighbor_id].neighbors[layer] = self.nodes[neighbor_id].neighbors[layer][:self.M]

    def _select_neighbors(self, node, layer, k=None):
        """
        Selects the nearest neighbors for a given node at a specified layer.

        Args:
            node (HNSWNode): The node for which neighbors are to be found.
            layer (int): The layer of the graph at which neighbors are being searched.
            k (int, optional): The number of nearest neighbors to find. Defaults to `ef_construction`.

        Returns:
            list: A list of IDs of the nearest neighbors.

        This method calculates the Euclidean distance of all nodes to the given node,
        and selects the closest neighbors, excluding the node itself.
        """
        if k is None:
            k = self.ef_construction  # Uses ef_construction as the default k value.
        node_data = np.array(node.data)
        distances = np.linalg.norm(self.data_matrix - node_data, axis=1)  # Compute distances from all nodes to the given node.
        neighbor_ids = np.argpartition(distances, min(len(distances) - 1, k+1))[:k+1]  # Selects the closest k+1 neighbors (including the node itself).
        return neighbor_ids[neighbor_ids != node.id].tolist()  # Excludes the node itself from its neighbors.

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
    
    nodes_df['node_id'] = nodes_df['node_id'].astype('int32')
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

def from_list(list, folder, max_chunk_chars=4000):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    embeddings = embed_documents(list)
    index = create_hnsw_index(embeddings)

    (nodes, edges) = serialize_hnsw_to_tables_v2(index)
    save_to_parquet(nodes, f"{folder}/nodes.parquet")
    save_to_parquet(edges, f"{folder}/edges.parquet")

    all_docs = pd.DataFrame([{ "id": i, "text": doc } for i, doc in enumerate(chunks) ])
    save_to_parquet(all_docs, f"{folder}/docs.parquet")

def from_document(path, folder, max_chunk_chars=4000):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    chunks = [""]
    
    with open(path) as f:
        for line in f.readlines():
            if len(chunks[-1]) >= max_chunk_chars:
                chunks.append("")
            chunks[-1] += line

    return from_list(chunks, folder, max_chunk_chars=max_chunk_chars)