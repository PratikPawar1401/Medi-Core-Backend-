import os
import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime
import requests
import pypdf
from pypdf import PdfReader

import faiss
import numpy as np
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    UnstructuredExcelLoader
)
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from huggingface_hub import InferenceClient, AsyncInferenceClient




from sqlalchemy.orm import Session
from database import DocumentStore, get_db

logger = logging.getLogger(__name__)

load_dotenv()

class HFInferenceEmbeddings(Embeddings):
    def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2"):
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            raise EnvironmentError("HUGGINGFACE_API_TOKEN not found in environment variables.")
        self.client = InferenceClient(token=token)
        self.model_id = model_id

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            results = self.client.feature_extraction(texts, model=self.model_id)
            # The API returns a List[List[float]] compatible with FAISS
            return results
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return []
        
    def embed_query(self, text: str) -> List[float]:
        try:
            result = self.client.feature_extraction(text, model=self.model_id)
            return result
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return []

    def __call__(self, texts):
        """
        This makes the instance callable to be used by vectorstores like FAISS.
        It accepts either a single string or list of strings.
        """
        if isinstance(texts, str):
            # Single text example
            return self.embed_query(texts)
        elif isinstance(texts, list):
            # List of texts example
            return self.embed_documents(texts)
        else:
            raise ValueError("Input must be a string or list of strings")    

class MedicalRAGSystem:
    def __init__(self, vector_store_path: str = "medical_vector_store"):
        self.vector_store_path = vector_store_path
        self.embeddings = HFInferenceEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_store = self._load_or_create_vector_store()

    def _load_or_create_vector_store(self) -> FAISS:
        faiss_file = f"{self.vector_store_path}.faiss"
        if os.path.exists(faiss_file):
            try:
                return FAISS.load_local(self.vector_store_path, self.embeddings)
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")

        # create empty FAISS store
        sample_embed = self.embeddings.embed_query("sample")
        if sample_embed is None or (isinstance(sample_embed, np.ndarray) and sample_embed.size == 0):
            logger.warning("Empty embedding returned for sample query; creating zero-dim index.")
            dim = 384  # fallback dimension typical for MiniLM models
        else:
            dim = len(sample_embed)

        index = faiss.IndexFlatL2(dim)
        return FAISS(self.embeddings, index, InMemoryDocstore({}), {})


    async def process_uploaded_file(self, file_path: str, file_type: str, user_id: int, db: Session) -> dict:
        from langchain_community.document_loaders import (
            PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
        )

        loader_map = {
            "pdf": PyPDFLoader,
            "docx": Docx2txtLoader,
            "doc": Docx2txtLoader,
            "txt": TextLoader,
            "xlsx": UnstructuredExcelLoader,
            "xls": UnstructuredExcelLoader,
        }

        Loader = loader_map.get(file_type.lower())
        if not Loader:
            return {"status": "error", "message": f"Unsupported file type: {file_type}"}

        try:
            documents = await asyncio.to_thread(Loader(file_path).load)
        except Exception as e:
            logger.error(f"Failed loading document: {e}")
            return {"status": "error", "message": f"Failed to load document: {str(e)}"}

        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "user_id": user_id,
                "file_path": file_path,
                "file_type": file_type,
                "upload_timestamp": datetime.now().isoformat(),
                "chunk_index": i
            })

        self.vector_store.add_documents(chunks)
        self.vector_store.save_local(self.vector_store_path)

        doc_record = DocumentStore(
            user_id=user_id,
            filename=os.path.basename(file_path),
            file_type=file_type,
            file_path=file_path,
            chunk_count=len(chunks),
            processed_at=datetime.now()
        )
        db.add(doc_record)
        db.commit()

        return {
            "status": "success",
            "chunks_processed": len(chunks),
            "document_id": doc_record.id,
            "message": f"Processed {len(chunks)} chunks from {os.path.basename(file_path)}"
        }


    def search_knowledge_base(self, query: str, k: int = 5, user_id: int = None) -> List[Document]:
        fetch_k = k * 5 if user_id is not None else k
        results = self.vector_store.similarity_search(query, k=fetch_k)
        logger.info(f"Similarity search returned {len(results)} documents for query '{query}'")

        if user_id is not None:
            filtered = [doc for doc in results if doc.metadata.get("user_id") == user_id]
            logger.info(f"{len(filtered)} documents remain after filtering by user_id={user_id}")
            return filtered[:k]
    
        return results



    def get_context_for_query(self, query: str, user_id: int = None, max_context_length: int = 3000) -> str:
        docs = self.search_knowledge_base(query, k=5, user_id=user_id)
        context_parts = []
        total_length = 0
        for doc in docs:
            content = doc.page_content
            if total_length + len(content) > max_context_length:
                remaining = max_context_length - total_length
                context_parts.append(content[:remaining])
                break
            context_parts.append(content)
            total_length += len(content)
        return "\n\n".join(context_parts)

# Instantiate global rag_system
rag_system = MedicalRAGSystem()
