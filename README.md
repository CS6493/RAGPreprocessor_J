## File Structure

**Scripts:**
- `preprocess.py` - Loads HotpotQA → 4 chunk configs → BM25 + Contriever indexes

**BM25 Indexes (4 configs):**
- `bm25_size_256_overlap_25.pkl`
- `bm25_size_256_overlap_50.pkl`
- `bm25_size_512_overlap_25.pkl`
- `bm25_size_512_overlap_50.pkl`

**Chunk Data (4 configs):**
- `chunks_size_256_overlap_25.json`
- `chunks_size_256_overlap_50.json`
- `chunks_size_512_overlap_25.json`
- `chunks_size_512_overlap_50.json`

**Contriever Dense Index:**
- `contriever_default.index`
- `contriever_default_metadata.json`


