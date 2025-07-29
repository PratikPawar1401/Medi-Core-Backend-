import os
import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime

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

from sqlalchemy.orm import Session
from database import DocumentStore, get_db

logger = logging.getLogger(__name__)

class MedicalRAGSystem:
    def __init__(self, vector_store_path: str = "medical_vector_store"):
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_store = self._load_or_create_vector_store()
        
    def _load_or_create_vector_store(self) -> FAISS:
        """Load existing vector store or create new one."""
        vector_store_file = f"{self.vector_store_path}.faiss"
        
        if os.path.exists(vector_store_file):
            try:
                return FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")
        
        # Create new empty vector store
        sample_embedding = self.embeddings.embed_query("sample")
        embedding_size = len(sample_embedding)
        
        index = faiss.IndexFlatL2(embedding_size)
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}
        
        return FAISS(self.embeddings, index, docstore, index_to_docstore_id)
    
    async def process_uploaded_file(
        self, 
        file_path: str, 
        file_type: str,
        user_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """Process uploaded medical document and add to knowledge base."""
        try:
            # Load document based on file type
            documents = await self._load_document(file_path, file_type)
            
            if not documents:
                raise ValueError(f"No content extracted from {file_path}")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata.update({
                    "user_id": user_id,
                    "file_path": file_path,
                    "file_type": file_type,
                    "upload_timestamp": datetime.now().isoformat(),
                    "chunk_index": chunks.index(chunk)
                })
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            # Save updated vector store
            self.vector_store.save_local(self.vector_store_path)
            
            # Store document metadata in database
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
                "message": f"Successfully processed {len(chunks)} chunks from {os.path.basename(file_path)}"
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def _load_document(self, file_path: str, file_type: str) -> List[Document]:
        """Load document based on file type."""
        try:
            if file_type.lower() == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_type.lower() in ['docx', 'doc']:
                loader = Docx2txtLoader(file_path)
            elif file_type.lower() == 'txt':
                loader = TextLoader(file_path)
            elif file_type.lower() in ['xlsx', 'xls']:
                loader = UnstructuredExcelLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return await asyncio.to_thread(loader.load)
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []
    
    def search_knowledge_base(
        self, 
        query: str, 
        k: int = 5,
        user_id: Optional[int] = None
    ) -> List[Document]:
        """Search knowledge base for relevant documents."""
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=k)
            
            # Filter by user if specified
            if user_id:
                results = [
                    doc for doc in results 
                    if doc.metadata.get("user_id") == user_id
                ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def get_context_for_query(
        self, 
        query: str, 
        user_id: Optional[int] = None,
        max_context_length: int = 3000
    ) -> str:
        """Get relevant context for RAG query."""
        relevant_docs = self.search_knowledge_base(query, k=5, user_id=user_id)
        
        context_parts = []
        current_length = 0
        
        for doc in relevant_docs:
            content = doc.page_content
            if current_length + len(content) > max_context_length:
                remaining_space = max_context_length - current_length
                content = content[:remaining_space]
                context_parts.append(content)
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        return "\n\n".join(context_parts)

# Global RAG system instance
rag_system = MedicalRAGSystem()
