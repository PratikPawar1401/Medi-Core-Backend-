from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from typing import List
import os
import aiofiles
from rag_system import rag_system
import google.generativeai as genai
from database import DocumentStore, get_db
import google.generativeai as genai
from fastapi import Depends, HTTPException, UploadFile, File, Form
from database import create_tables, get_db, User as DBUser, ChatSession, ChatMessage
from chatbot import router as chatbot_router
from auth.auth_handler import (
    authenticate_user, create_access_token, get_current_active_user,
    create_user, update_user_profile, get_user_by_email,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from auth.models import UserCreate, UserUpdate, UserProfile, Token
import logging
from chatbot import router as chatbot_router


load_dotenv()

app = FastAPI(title="FastAPI Medical Triage Backend with Database", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app.include_router(chatbot_router, prefix="", tags=["medical-chatbot"])

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
