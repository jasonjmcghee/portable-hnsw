# Portable HNSW

So what's going on here?

Yeah - fair question.

So I had this idea. 

What if an HNSW index ([hierarchical navigable small world graphs](https://arxiv.org/abs/1603.09320))
was just a file, and you could serve it from a CDN, and search it directly in the browser?

And what if you didn't need to load the entire thing in memory, so you could search a massive index
without being RAM rich?

That would be cool.

A vector store without a server...

So yeah. Here's a proof of concept.

---

There's a Python file called `build_index.py` that builds an index using a custom hnsw algorithm that 
can be serialized to a couple of parquet files.

_There are very likely bugs and performance problems. But it's within an order of magnitude or two
of `hnswlib` which was fast enough that my development cycle wasn't impacted by repeatedly re-indexing
the same files while building the search and front-end bits. I welcome pull requests to fix the problems
and make it halfway reasonable._

Then I wrote a webpage that uses `transformers.js`, `duckdb` and some SQL to read the parquet files and 
search it (similar to HNSW approx nearest neighbor search) and then retrieve the associated text.

A big part of the original idea was how this could scale to massive indices.

So, I also tested using parquet range requests and only retrieving what we need from the parquet file,
which worked! But since the index is only like 100MB, and each range request added overhead, loading
it all into memory was about twice as fast. But, it means you could have a 1TB index and it would
(theoretically) still work, which is pretty crazy.

You can try this yourself by swapping out the `nodes.parquet` bits in the SQL for `read_parquet('${path}/nodes.parquet')`.
DuckDB takes care of the rest.

---

Anyway, would love feedback and welcome contributions.

It was a fun project!
