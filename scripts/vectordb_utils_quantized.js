import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0";

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

console.log("triggering vectordb utils quantized;")

// Reference the elements that we will need
const status = document.getElementById("status");
const textInput = document.getElementById("text-input");
const blogs = document.getElementById("blogs");
const shakespeare = document.getElementById("shakespeare");
const frankenstein = document.getElementById("frankenstein");
const search = document.getElementById("search");
const instantMode = document.getElementById("instant-mode");
instantMode.checked = true;
const output = document.getElementById("output");
const path = document.getElementById("path");
let cachedPath = "";
let conn = null;

// Instantiate the embeddings (feature extraction) pipeline
status.textContent = "Loading embedding model...";
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);
status.textContent = "Ready for user!";

import * as duckdb from "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.28.0/+esm";

const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();

// Select a bundle based on browser checks
const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);

// Create a new web worker for database operations
const worker_url = URL.createObjectURL(
  new Blob([`importScripts("${bundle.mainWorker}");`], {
    type: "text/javascript",
  })
);

// Instantiate the asynchronus version of DuckDB-wasm
const worker = new Worker(worker_url);
const logger = new duckdb.ConsoleLogger();
const db = new duckdb.AsyncDuckDB(logger, worker);
window.db = db;
await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
URL.revokeObjectURL(worker_url);

// Function to calculate Euclidean distance squared between two vectors
function euclideanDistanceSquared(arr1, arr2, bitsPerDimension = 8) {
  const scale = 2 ** bitsPerDimension - 1;

  if (arr1.length !== arr2.length) {
    throw new Error("Arrays must be of the same length");
  }
  let sum = 0;
  for (let i = 0; i < arr1.length; i++) {
    // dequantize before calculating the distance
    sum += (arr1[i] / scale - arr2[i] / scale) ** 2;
  }
  return sum;
}

function quantizeVector(vector, bitsPerDimension = 8) {
  const maxVal = Math.max(...vector);
  const minVal = Math.min(...vector);
  const range = maxVal - minVal;
  const scale = 2 ** bitsPerDimension - 1;

  // appropriate typed array based on bitsPerDimension
  let TypedArrayConstructor;
  if (bitsPerDimension <= 8) {
    TypedArrayConstructor = Int8Array;
  } else if (bitsPerDimension <= 16) {
    TypedArrayConstructor = Int16Array;
  } else {
    TypedArrayConstructor = Int32Array;
  }

  const quantizedVector = new TypedArrayConstructor(vector.length);
  for (let i = 0; i < vector.length; i++) {
    // 0-1 normalization unsigned
    // let normalizedVector = vector[i] - minVal / range;

    // -1 -> 1 normalization signed
    let normalizedVector = 2 * ((vector[i] - minVal) / range) - 1;
    quantizedVector[i] = Math.floor(normalizedVector * scale);
  }
  return quantizedVector;
}

// Function to search with SQL queries using the DuckDB connection
async function searchWithSql(conn, queryData, k, path, ef = 10) {
  path = path.startsWith("http") ? path : `${window.location.href}/${path}`;

  status.textContent = "Searching HNSW Index...";

  await db.registerFileURL(
    "docs_quantized.parquet",
    `${path}/docs_quantized.parquet`,
    duckdb.DuckDBDataProtocol.HTTP,
    false
  );

  const cache = {};

  const countQuery = `SELECT COUNT(node_id) FROM nodes_quantized.parquet`;
  const countResult = await conn.query(countQuery);
  const countArrayed = countResult.toArray().map(([count, _]) => count[1]);
  const count = JSON.parse(countArrayed[0].toString());
  const quantizedQueryData = quantizeVector(queryData);

  const maxLayer = count > 0 ? Math.floor(Math.log2(count)) : 0;

  // query the database to randomly select one row from the nodes.parquet table
  const initNodeQuery = `
                SELECT n.node_id, n.data as node_data
                FROM nodes_quantized.parquet n
                ORDER BY RANDOM() LIMIT 1
            `;
  let currentBest = await conn.query(initNodeQuery);
  // compare the current embedding vector to the randomly selected row embedding
  currentBest = currentBest
    .toArray()
    .map(([nId, data]) => [
      nId[1],
      euclideanDistanceSquared(data[1].data[0].values, quantizedQueryData),
    ]);

  for (let layer = maxLayer; layer >= 0; layer--) {
    status.textContent = `Searching HNSW Index: ${
      layer + 1
    } layers remaining...`;

    let improved = true;
    while (improved) {
      improved = false;
      const currentNodeIds = currentBest;
      const candidates = new Set(currentNodeIds);
      const newCandidates = new Set();

      // maintain a visited Array
      const filteredCandidates = Array.from(candidates).filter(
        (nodeId) => !cache[nodeId] || !cache[nodeId][layer]
      );

      if (filteredCandidates.length === 0) {
        continue;
      }

      const sqlSafeFilteredCandidates = filteredCandidates.join(",");
      const sqlSafeCandidates = [...candidates, ...Object.keys(cache)].join(
        ","
      );

      // Create filtered_edges table
      const createFilteredEdgesQuery = `
                        DROP TABLE IF EXISTS filtered_edges;
                        CREATE TEMP TABLE filtered_edges AS
                        SELECT * FROM edges_quantized.parquet
                        WHERE source_node_id IN (${sqlSafeFilteredCandidates})
                        AND layer = ${layer}
                        AND target_node_id NOT IN (${sqlSafeCandidates});
                    `;
      await conn.query(createFilteredEdgesQuery);

      // Perform the join and fetch neighbors
      const fetchNeighborsQuery = `
                        SELECT 
                            e.target_node_id as node_id, 
                            n.data as node_data
                        FROM nodes_quantized.parquet n
                        INNER JOIN filtered_edges e ON n.node_id = e.target_node_id
                    `;
      const neighbors = (await conn.query(fetchNeighborsQuery)).toArray();

      if (neighbors.length === 0) {
        continue;
      }

      // Loop through neighbors
      for (const [neighborId_, node_data_] of neighbors) {
        const neighborId = neighborId_[1];
        if (newCandidates.has(neighborId)) {
          continue;
        }
        const data = node_data_[1].data[0].values;
        let dist;

        // get dist if neighbors exists in cache otherwise calculate fresh euclideanDistanceSquared
        if (cache[neighborId] && cache[neighborId].data) {
          dist = cache[neighborId].distance;
        } else {
          dist = euclideanDistanceSquared(data, quantizedQueryData);
          cache[neighborId] = { data, distance: dist };
        }

        // Include the neighbor in newCandidates if the difference in embeddingVectors is smaller
        if (
          currentBest.length < ef ||
          dist < currentBest[currentBest.length - 1][1]
        ) {
          currentBest.push([neighborId, dist]);
          newCandidates.add(neighborId);
          improved = true;
        }
      }

      currentBest.sort((a, b) => a[1] - b[1]);
      currentBest = currentBest.slice(0, ef);
    }
  }

  const ids = currentBest.slice(0, k).map(([nodeId, _]) => nodeId);

  status.textContent = `Found best candidates. Retrieving documents...`;

  const out = await Promise.all(
    ids.map(async (id) => {
      const result = await conn.query(
        `SELECT text FROM read_parquet('${path}/docs_quantized.parquet') OFFSET ${id} LIMIT 1`
      );
      return result.toArray().map(([item]) => item[1])[0];
    })
  );
  return out;
}

async function embed(text) {
  status.textContent = "Analysing...";
  const out = await extractor(text, { pooling: "mean", normalize: true });
  status.textContent = "Ready for user!";
  return out;
}

async function loadIndex() {
  debugger;
  if (path.value !== cachedPath) {
    cachedPath = path.value;
  }
  status.textContent = "Loading HNSW Index...";
  conn = await db.connect();
  const nodesRes = await fetch(`${cachedPath}/nodes_quantized.parquet`);
  // edit this Int8Array as per quantization
  await db.registerFileBuffer(
    "nodes_quantized.parquet",
    new Int8Array(await nodesRes.arrayBuffer())
  );
  const edgesRes = await fetch(`${cachedPath}/edges_quantized.parquet`);
  await db.registerFileBuffer(
    "edges_quantized.parquet",
    new Int8Array(await edgesRes.arrayBuffer())
  );
  status.textContent = "HNSW Index Loaded.";
  status.textContent = "Ready for user!";
}

async function performSearch() {
  const queryData = (await embed(textInput.value)).data;
  status.textContent = "Searching...";

  if (path.value !== cachedPath) {
    cachedPath = path.value;
    if (conn != null) {
      await conn.close();
    }
    await loadIndex();
  }

  if (conn == null) {
    conn = await db.connect();
  }

  const out = await searchWithSql(
    // DuckDB
    conn,
    // Embedding
    queryData,
    // K
    5,
    // Path to parquet indices
    cachedPath,
    // EF (the max number of candidate neighbors during search)
    20
  );

  status.textContent = "Ready for user!";
  output.innerHTML = out.map((t) => `<p>${t}</p>`).join("");
}

const debounce = (callback, wait) => {
  let timeoutId = null;
  return () => {
    window.clearTimeout(timeoutId);
    timeoutId = window.setTimeout(async () => {
      await callback();
    }, wait);
  };
};

const debouncedInstantSearch = debounce(async () => await performSearch(), 300);

textInput.addEventListener("input", async (e) => {
  if (instantMode.checked && !search.disabled) {
    debouncedInstantSearch();
  }
});

search.addEventListener("click", async (e) => {
  search.disabled = true;
  await performSearch();
  search.disabled = false;
});

blogs.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  path.value = "blogs";
  instantMode.checked = false;
  textInput.value = "falling in love";
  await loadIndex();
  status.textContent = "This dataset is bigger - hit search when you're ready.";
});

frankenstein.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  instantMode.checked = true;
  path.value = "frankenstein";
  textInput.value = "in the ocean";
  await performSearch();
  search.disabled = false;
});

shakespeare.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  path.value = "shakespeare";
  textInput.value = "got some blood on their hands";
  await performSearch();
});

await performSearch();
search.disabled = false;
