import os
import re
import json
import pickle

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Set environment variable to handle Hugging Face connection issues
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class RAGPreprocessor:
    def __init__(self, dataset_name="hotpotqa/hotpot_qa", split_name="distractor"):
        """Initialize the RAG preprocessor"""
        self.dataset_name = dataset_name
        self.split_name = split_name

        self.raw_data = None
        self.raw_documents = []
        self.cleaned_documents = []
        self.chunk_results = {}
        self.final_chunks = []

        self.bm25 = None
        self.tokenized_corpus = None
        self.dense_model = None
        self.faiss_index = None

    def read_data(self, num_samples=50):
        """Step 1: Load dataset from Hugging Face and extract text"""
        print(f"Loading {self.dataset_name} dataset...")
        self.raw_documents = []

        ds = load_dataset(
            self.dataset_name,
            self.split_name,
            verification_mode="no_checks",
            trust_remote_code=True
        )

        self.raw_data = ds

        train_set = ds["train"]
        if num_samples is not None:
            num_samples = min(num_samples, len(train_set))
            train_set = train_set.select(range(num_samples))

        print(f"Successfully loaded! Extracting {len(train_set)} records.")

        # Extract text based on HotpotQA's specific context structure
        for idx, item in enumerate(train_set):
            titles = item["context"]["title"]
            sentences_lists = item["context"]["sentences"]

            full_text = ""
            for title, sentences in zip(titles, sentences_lists):
                full_text += f"Title: {title}\n"
                full_text += " ".join(sentences) + "\n\n"

            self.raw_documents.append(full_text)

        return self.raw_documents

    def clean_data(self, min_length=20):
        """Step 2: Clean the extracted documents"""
        print("\nStarting data cleaning process...")
        self.cleaned_documents = []
        removed_count = 0

        for text in self.raw_documents:
            # Replace 3 or more consecutive newlines with exactly 2 newlines
            cleaned_text = re.sub(r"\n{3,}", "\n\n", text)
            # Replace multiple spaces/tabs with a single space
            cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
            # Remove leading and trailing whitespaces
            cleaned_text = cleaned_text.strip()

            # Filter out invalid/empty paragraphs that are too short
            if len(cleaned_text) < min_length:
                removed_count += 1
                continue

            self.cleaned_documents.append(cleaned_text)

        print(f"Number of documents before cleaning: {len(self.raw_documents)}")
        print(f"Number of documents removed (too short): {removed_count}")
        print(f"Number of valid documents after cleaning: {len(self.cleaned_documents)}")

        return self.cleaned_documents

    def chunk_data(self, configs=None, default_key=None):
        """Step 3: Perform document chunking with multiple parameter combinations"""
        if not self.cleaned_documents:
            raise ValueError("Please run clean_data() before running chunk_data()")

        self.chunk_results = {}
        self.final_chunks = []

        # Add the 4 specific configurations for ablation study
        if configs is None:
            configs = [
                {"size": 256, "overlap": 25},
                {"size": 512, "overlap": 50},
                {"size": 256, "overlap": 50},
                {"size": 512, "overlap": 25}
            ]

        print("\nStarting chunking experiments...")

        for config in configs:
            size = config["size"]
            overlap = config["overlap"]
            key = f"size_{size}_overlap_{overlap}"

            print(f" -> Testing parameters - Chunk Size: {size}, Overlap: {overlap}")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ".", "!", "?", " ", ""]
            )

            chunks = []
            for doc_id, text in enumerate(self.cleaned_documents):
                # Create LangChain Document objects with metadata
                docs = splitter.create_documents(
                    [text],
                    metadatas=[{
                        "doc_id": doc_id,
                        "source_dataset": self.dataset_name,
                        "split_name": self.split_name,
                        "chunk_size": size,
                        "chunk_overlap": overlap
                    }]
                )

                # Assign a unique chunk ID to each generated chunk
                for chunk_idx, doc in enumerate(docs):
                    doc.metadata["chunk_id"] = f"{doc_id}_{chunk_idx}"

                chunks.extend(docs)

            self.chunk_results[key] = chunks
            print(f"    Result: Generated {len(chunks)} chunks.")

        # Set a default chunk result if not explicitly specified
        if default_key is None:
            default_key = "size_512_overlap_50"

        if default_key not in self.chunk_results:
            default_key = list(self.chunk_results.keys())[0]

        self.final_chunks = self.chunk_results[default_key]
        return self.final_chunks

    def save_all_chunks(self, output_dir="output"):
        """Step 4: Save all chunk combinations into separate JSON files"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if not self.chunk_results:
            raise ValueError("No chunks available to save. Please run chunk_data() first.")

        saved_files = []
        for key, chunks in self.chunk_results.items():
            records = []
            for doc in chunks:
                records.append({
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "doc_id": doc.metadata.get("doc_id"),
                    "source_dataset": doc.metadata.get("source_dataset"),
                    "split_name": doc.metadata.get("split_name"),
                    "chunk_size": doc.metadata.get("chunk_size"),
                    "chunk_overlap": doc.metadata.get("chunk_overlap"),
                    "text": doc.page_content
                })

            if output_dir:
                output_path = os.path.join(output_dir, f"chunks_{key}.json")
            else:
                output_path = f"chunks_{key}.json"
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
                
            print(f"Saved {len(chunks)} chunks to: {output_path}")
            saved_files.append(output_path)
            
        return saved_files

    def build_all_bm25(self, output_dir="output"):
        """Step 5a: Sparse Retrieval - Build BM25 indices for each parameter combination"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if not self.chunk_results:
            raise ValueError("No chunks available to build BM25. Please run chunk_data() first.")

        print("\nBuilding Sparse Retrieval Indices (BM25)...")
        saved_files = []
        for key, chunks in self.chunk_results.items():
            # Basic tokenization (lowercase and split by space)
            corpus = [doc.page_content for doc in chunks]
            tokenized_corpus = [text.lower().split() for text in corpus]

            # Build the BM25 model
            bm25_model = BM25Okapi(tokenized_corpus)

            chunk_records = []
            for doc in chunks:
                chunk_records.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

            if output_dir:
                save_path = os.path.join(output_dir, f"bm25_{key}.pkl")
            else:
                save_path = f"bm25_{key}.pkl"
            
            # Save the model and corpus using pickle
            with open(save_path, "wb") as f:
                pickle.dump({
                    "bm25": bm25_model,
                    "tokenized_corpus": tokenized_corpus,
                    "chunk_records": chunk_records
                }, f)

            print(f"BM25 index built and saved to: {save_path}")
            saved_files.append(save_path)
            
        return saved_files

    def build_dense_index(self, model_name="facebook/contriever-msmarco", output_dir="output"):
        """Step 5b: Dense Retrieval - Build Contriever + FAISS index for the default chunk configuration"""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
        except ImportError:
            print("❌ Missing dependencies for dense retrieval. Please install:")
            print("pip install sentence-transformers faiss-cpu numpy")
            return None

        if not self.final_chunks:
            raise ValueError("No default chunks available. Please run chunk_data() first.")

        print(f"\nBuilding Dense Retrieval Index using {model_name}...")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 1. Load the embedding model (Contriever is excellent for information retrieval)
        print("Loading embedding model...")
        self.dense_model = SentenceTransformer(model_name)

        # 2. Encode all chunks into high-dimensional vectors
        print("Encoding chunks into dense vectors... (This might take a moment)")
        corpus = [doc.page_content for doc in self.final_chunks]
        embeddings = self.dense_model.encode(corpus)
        embeddings = np.array(embeddings).astype('float32')

        # 3. Normalize vectors to compute Cosine Similarity using Inner Product
        faiss.normalize_L2(embeddings)

        # 4. Create the FAISS Index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension) # IndexFlatIP computes inner product
        self.faiss_index.add(embeddings)

        # 5. Save the FAISS index to disk
        if output_dir:
            index_path = os.path.join(output_dir, "contriever_default.index")
        else:
            index_path = "contriever_default.index"
            
        faiss.write_index(self.faiss_index, index_path)

        # 6. Save metadata mapper so we can retrieve the actual text later
        metadata_records = []
        for doc in self.final_chunks:
            metadata_records.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "text": doc.page_content
            })
            
        meta_path = index_path.replace('.index', '_metadata.json')
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_records, f, ensure_ascii=False, indent=2)

        print(f"✅ Dense FAISS index saved to: {index_path}")
        print(f"✅ Dense metadata saved to: {meta_path}")
        
        return self.faiss_index


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Initialize preprocessor
    preprocessor = RAGPreprocessor()

    # 2. Read first 50 samples (Change to None for full dataset)
    preprocessor.read_data(num_samples=50)

    # 3. Clean the raw documents
    preprocessor.clean_data()

    # 4. Perform chunking ablation study (4 combinations)
    preprocessor.chunk_data()

    print("\n--- Saving all experiment files ---")
    
    # 5. Save chunk JSONs and build Sparse Retrieval Indices (BM25)
    preprocessor.save_all_chunks(output_dir="")
    preprocessor.build_all_bm25(output_dir="")

    # 6. Build Dense Retrieval Index (Contriever + FAISS)
    preprocessor.build_dense_index(output_dir="")

    print("\n--- Preprocessing completed successfully! ---")