File Structure 
RAGPreprocessor_J/
├── 📄 preprocess.py                    # Main preprocessing script
|
├── 📊 BM25 Indexes (4 configurations)
│   ├── bm25_size_256_overlap_25.pkl
│   ├── bm25_size_256_overlap_50.pkl
│   ├── bm25_size_512_overlap_25.pkl
│   └── bm25_size_512_overlap_50.pkl
│
├── 📊 Chunked Data (4 configurations)
│   ├── chunks_size_256_overlap_25.json
│   ├── chunks_size_256_overlap_50.json
│   ├── chunks_size_512_overlap_25.json
│   └── chunks_size_512_overlap_50.json
│
└── 🔍 Contriever Retrieval Index
    ├── contriever_default.index
    └── contriever_default_metadata.json


preprocess.py - Core RAG pipeline: Loads HotpotQA → cleans text → 4x chunking configs (256/512 size, 25/50 overlap) → builds BM25 + Contriever/FAISS indexes

