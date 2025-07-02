import chromadb
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorDatabase:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = None):
        logger.info("Initializing ChromaVectorDatabase...")

        # Load the embedding model
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

        # Initialize ChromaDB with or without persistence
        try:
            if persist_directory:
                logger.info(f"Using persistent directory: {persist_directory}")
                self.client = chromadb.Client(Settings(
                    is_persistent=True,
                    persist_directory=persist_directory,
                    allow_reset=True
                ))
            else:
                logger.info("Using in-memory mode")
                self.client = chromadb.Client(Settings(
                    is_persistent=False,
                    allow_reset=True
                ))

            self.collection = self.client.get_or_create_collection(name="document_embeddings")
            logger.info("ChromaVectorDatabase initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise RuntimeError(
                "ChromaDB initialization failed. If deploying on Streamlit Cloud, make sure sqlite3 >= 3.35.0 or use in-memory mode."
            )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def add_documents(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents...")
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")

            if not chunks:
                logger.warning("No chunks created")
                return

            texts = [chunk.page_content for chunk in chunks]
            metadata = [chunk.metadata for chunk in chunks]
            embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32).tolist()
            ids = [f"doc_{i}" for i in range(len(chunks))]

            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadata,
                ids=ids
            )
            logger.info(f"Added {len(chunks)} chunks to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.1) -> List[Document]:
        if not self.collection.count():
            logger.info("No documents in collection")
            return []

        logger.info(f"Searching for query: '{query[:50]}...' (k={k})")
        try:
            query_embedding = self.model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k
            )

            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            docs = []
            for i, (doc_content, meta, distance) in enumerate(zip(documents, metadatas, distances)):
                if distance < (1 - threshold):  # Convert similarity threshold to distance
                    meta_copy = meta.copy() if meta else {}
                    meta_copy["similarity_score"] = 1 - distance
                    docs.append(Document(page_content=doc_content, metadata=meta_copy))

            logger.info(f"Found {len(docs)} relevant documents")
            return docs
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            return []

    def clear_database(self):
        try:
            self.client.reset()
            self.collection = self.client.get_or_create_collection(name="document_embeddings")
            logger.info("Database cleared")
        except Exception as e:
            logger.error(f"Failed to clear database: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'total_documents': self.collection.count(),
            'has_embeddings': self.collection.count() > 0,
            'database_path': "in-memory or persistent"
        }
        return stats
