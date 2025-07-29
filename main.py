from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from typing import List
import os
import aiofiles
from rag_system import rag_system
from database import DocumentStore, get_db
import google.generativeai as genai

from database import create_tables, get_db, User as DBUser, ChatSession, ChatMessage
from chatbot import router as chatbot_router
from auth.auth_handler import (
    authenticate_user, create_access_token, get_current_active_user,
    create_user, update_user_profile, get_user_by_email,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from auth.models import UserCreate, UserUpdate, UserProfile, Token
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


load_dotenv()

app = FastAPI(title="FastAPI Medical Triage Backend with Database", version="1.0.0")

# Create database tables on startup
@app.on_event("startup")
def startup_event():
    create_tables()
    print("✅ Database tables created successfully")

# Include chatbot router
app.include_router(chatbot_router, prefix="", tags=["medical-chatbot"])

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI Medical Triage Backend with Database"}

@app.post("/register", response_model=dict)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """User registration endpoint with database storage."""
    existing_user = get_user_by_email(db, user.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = create_user(db, user)
    return {"message": "User registered successfully", "user_id": new_user.id}

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """User login endpoint."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/profile", response_model=UserProfile)
async def get_profile(current_user: DBUser = Depends(get_current_active_user)):
    """Get current user profile."""
    return current_user

@app.put("/profile", response_model=UserProfile)
async def update_profile(
    user_update: UserUpdate,
    current_user: DBUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user profile."""
    updated_user = update_user_profile(db, current_user.id, user_update)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    return updated_user

@app.get("/chat-history")
async def get_chat_history(
    current_user: DBUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's chat history."""
    sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).all()
    
    chat_history = []
    for session in sessions:
        messages = db.query(ChatMessage).filter(ChatMessage.session_id == session.session_id).order_by(ChatMessage.timestamp.asc()).all()
        chat_history.append({
            "session_id": session.session_id,
            "created_at": session.created_at,
            "completed": session.completed,
            "emergency_detected": session.emergency_detected,
            "final_assessment": session.final_assessment,
            "main_symptoms": session.main_symptoms,
            "messages_count": len(messages),
            "messages": [
                {
                    "message_type": msg.message_type,
                    "content": msg.content,
                    "stage": msg.stage,
                    "timestamp": msg.timestamp
                } for msg in messages
            ]
        })
    
    return {"chat_history": chat_history, "total_sessions": len(chat_history)}

@app.post("/upload-documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: DBUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload and process medical documents for knowledge base."""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create upload directory if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    results = []
    
    for file in files:
        # Validate file type
        allowed_types = ['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls']
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_types:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Unsupported file type: {file_extension}"
            })
            continue
        
        # Save file
        file_path = os.path.join(upload_dir, f"{current_user.id}_{file.filename}")
        
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process file
            result = await rag_system.process_uploaded_file(
                file_path=file_path,
                file_type=file_extension,
                user_id=current_user.id,
                db=db
            )
            
            result["filename"] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return {"results": results}

@app.get("/my-documents")
async def get_my_documents(
    current_user: DBUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's uploaded documents."""
    documents = db.query(DocumentStore).filter(
        DocumentStore.user_id == current_user.id
    ).order_by(DocumentStore.processed_at.desc()).all()
    
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "chunk_count": doc.chunk_count,
                "processed_at": doc.processed_at
            }
            for doc in documents
        ]
    }

@app.post("/query-documents")
async def query_documents(
    query: str = Form(...),
    use_personal_docs: bool = Form(True),
    current_user: DBUser = Depends(get_current_active_user)
):
    """Query the knowledge base with RAG."""
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get relevant context
        user_id = current_user.id if use_personal_docs else None
        context = rag_system.get_context_for_query(query, user_id=user_id)
        
        if not context:
            return {
                "query": query,
                "answer": "I couldn't find relevant information in your documents. Please try uploading medical documents first or rephrasing your query.",
                "sources": []
            }
        
        # Generate response using your existing Gemini model
        enhanced_prompt = f"""
        You are a medical AI assistant with access to uploaded medical documents. 
        Use the following context from the medical documents to answer the user's question accurately.
        
        Context from uploaded documents:
        {context}
        
        User Question: {query}
        
        Instructions:
        1. Base your answer primarily on the provided context
        2. If the context doesn't contain sufficient information, say so clearly
        3. Always remind users to consult healthcare professionals for medical decisions
        4. Be precise and avoid speculation
        
        Answer:
        """
        
        response = model.generate_content(enhanced_prompt)
        answer = response.text.strip()
        
        # Get source documents for transparency
        source_docs = rag_system.search_knowledge_base(query, k=3, user_id=user_id)
        sources = [
            {
                "filename": doc.metadata.get("file_path", "Unknown").split('/')[-1],
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "content_preview": doc.page_content[:200] + "..."
            }
            for doc in source_docs
        ]
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "context_used": len(context) > 0
        }
        
    except Exception as e:
        logger.error(f"Error in document query: {e}")
        raise HTTPException(status_code=500, detail="Error processing your query")

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: DBUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a document from user's knowledge base."""
    
    document = db.query(DocumentStore).filter(
        DocumentStore.id == document_id,
        DocumentStore.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove file from filesystem
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Remove from database
        db.delete(document)
        db.commit()
        
        # Note: Removing from FAISS vector store is complex
        # For now, we'll rebuild the vector store periodically
        # or implement a more sophisticated approach
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")
