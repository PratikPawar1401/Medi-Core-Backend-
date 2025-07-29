import os
import pickle
import logging
import re
from uuid import uuid4
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from database import get_db, ChatSession as DBChatSession, ChatMessage as DBChatMessage, User as DBUser
from auth.auth_handler import get_current_active_user
from fastapi import Depends
from rag_system import rag_system
from langchain_core.documents import Document
import aiofiles 

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

# Configure AI model
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Global vector store

VECTOR_STORE_PATH = "vector_store.pkl"
if os.path.exists(VECTOR_STORE_PATH):
    with open(VECTOR_STORE_PATH, "rb") as f:
        loaded_store = pickle.load(f)
    rag_system.vector_store = loaded_store

vector_store = rag_system.vector_store


# Medical triage stages
class TriageStage(Enum):
    GREETING = "greeting"
    CONSENT = "consent"
    DEMOGRAPHICS = "demographics"
    MEDICAL_HISTORY = "medical_history"
    MAIN_SYMPTOMS = "main_symptoms"
    SYMPTOM_DETAILS = "symptom_details"
    ASSOCIATED_SYMPTOMS = "associated_symptoms"
    SUMMARY_CONFIRMATION = "summary_confirmation"
    FINAL_ASSESSMENT = "final_assessment"
    COMPLETED = "completed"

class TriageSession:
    def __init__(self):
        self.stage = TriageStage.GREETING
        self.data = {}
        self.completed = False
        self.created_at = datetime.now()
        self.red_flags_detected = False

# In-memory session storage
triage_sessions: Dict[str, TriageSession] = {}

# Knowledge base integration
class MedicalKnowledgeBase:
    def __init__(self, vector_store, user_id: Optional[int] = None):
        self.vector_store = vector_store
        self.user_id = user_id

    def query(self, text: str, k: int = 3) -> List[Document]:
        if not self.vector_store:
            return []
        # similarity_search returns Documents
        docs = self.vector_store.similarity_search(text, k=k)
        # Optionally filter by metadata["user_id"]
        if self.user_id:
            docs = [d for d in docs if d.metadata.get("user_id")==self.user_id]
        return docs

    def get_context(self, text: str, k: int = 3) -> str:
        docs = self.query(text, k)
        return " ".join(d.page_content for d in docs)


#helper functions for formatting
def format_confirmation(item: str) -> str:
    """Standard confirmation format"""
    return f"✓ {item}"

def format_section_header(title: str) -> str:
    """Standard section header format"""
    return f"{title.upper()}:"

def get_concise_progress(stage_num: int, total: int) -> str:
    """Concise progress indicator"""
    return f"{stage_num}/{total}"

def clean_medical_assessment(text: str) -> str:
    """Clean and format medical assessment text."""
    import re
    
    # Remove standalone asterisks
    text = re.sub(r'(?<!\)\(?!\*)', '', text)
    
    # Clean up markdown bold formatting
    text = re.sub(r'\\(.?)\\*', r'\1', text)
    
    # Ensure proper spacing around headers
    text = re.sub(r'^([A-Z][A-Z\s]+:)', r'\n\1', text, flags=re.MULTILINE)
    
    # Clean up bullet points
    text = re.sub(r'•\s*', '• ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Ensure proper line breaks before major sections
    sections = ['CLINICAL SUMMARY:', 'DIFFERENTIAL DIAGNOSIS:', 'URGENCY ASSESSMENT:', 
                'CLINICAL RECOMMENDATIONS:', 'FOLLOW-UP RECOMMENDATIONS:', 'MEDICAL DISCLAIMER:']
    
    for section in sections:
        text = text.replace(section, f'\n{section}')
    
    return text.strip()



# Request/Response models
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    finished: bool
    stage: str
    progress: str
    extracted_info: Dict[str, Any] = {}
    

# Red flag symptoms for emergency detection
RED_FLAG_SYMPTOMS = [
    "worst headache of my life", "worst headache ever", "thunderclap headache",
    "loss of consciousness", "passed out", "fainted", "unconscious",
    "difficulty speaking", "slurred speech", "can't speak properly",
    "weakness on one side", "paralysis", "can't move arm", "can't move leg",
    "chest pain with shortness of breath", "crushing chest pain",
    "severe difficulty breathing", "can't breathe", "gasping for air",
    "severe abdominal pain", "worst stomach pain ever",
    "blood in vomit", "vomiting blood", "threw up blood",
    "blood in stool", "bloody stool", "rectal bleeding",
    "seizure", "convulsion", "fit", "shaking uncontrollably",
    "sudden vision loss", "blind", "can't see",
    "high fever with stiff neck", "neck stiffness with fever",
    "severe allergic reaction", "anaphylaxis", "throat closing"
]

@router.post("/triage", response_model=ChatResponse)
async def medical_triage(
    req: ChatRequest,
    current_user: DBUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Complete medical triage conversation following clinical protocols with database storage."""
    
    sess_id = req.session_id or str(uuid4())
    session = triage_sessions.get(sess_id, TriageSession())
    triage_sessions[sess_id] = session
    session.data["user_id"] = current_user.id

    # Get or create database session
    db_session = db.query(DBChatSession).filter(DBChatSession.session_id == sess_id).first()
    if not db_session:
        db_session = DBChatSession(
            session_id=sess_id,
            user_id=current_user.id,
            stage=TriageStage.GREETING.value
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
    
    
    message = req.message.strip()
    
    # Save user message to database
    user_message = DBChatMessage(
        session_id=sess_id,
        user_id=current_user.id,
        message_type="user",
        content=message,
        stage=session.stage.value
    )
    db.add(user_message)
    
    # Check for emergency keywords at any stage
    if check_emergency_keywords(message):
        session.red_flags_detected = True
        session.stage = TriageStage.COMPLETED
        session.completed = True
        reply = generate_emergency_response()
        
        # Update database session
        db_session.emergency_detected = True
        db_session.completed = True
        db_session.stage = TriageStage.COMPLETED.value
        db_session.completed_at = datetime.now()
        
    else:
        # Process based on current stage
        reply = await process_triage_stage(session, message)
        
        # Clean up the reply if it's a final assessment
        if session.stage == TriageStage.COMPLETED and "CLINICAL SUMMARY" in reply:
            reply = clean_medical_assessment(reply)

        # Update database session with collected data
        if session.data.get('age'):
            db_session.age = session.data['age']
            # Also update user profile if not set
            if not current_user.age:
                current_user.age = session.data['age']
                
        if session.data.get('sex'):
            db_session.sex = session.data['sex']
            if not current_user.sex:
                current_user.sex = session.data['sex']
                
        if session.data.get('medical_history'):
            db_session.medical_history = session.data['medical_history']
            if not current_user.medical_history:
                current_user.medical_history = session.data['medical_history']
        
        # Update other session data
        if session.data.get('main_symptoms'):
            db_session.main_symptoms = session.data['main_symptoms']
        if session.data.get('symptom_details'):
            db_session.symptom_details = session.data['symptom_details']
        if session.data.get('associated_symptoms'):
            db_session.associated_symptoms = session.data['associated_symptoms']
            
        # Update stage and completion status
        db_session.stage = session.stage.value
        db_session.completed = session.completed
        if session.completed:
            db_session.completed_at = datetime.now()
            if reply and "CLINICAL SUMMARY" in reply:
                db_session.final_assessment = reply
    
    # Save bot response to database
    bot_message = DBChatMessage(
        session_id=sess_id,
        user_id=current_user.id,
        message_type="bot",
        content=reply,
        stage=session.stage.value
    )
    db.add(bot_message)
    
    # Commit all changes
    db.commit()
    
    # Calculate progress
    total_stages = len(TriageStage) - 2  # Excluding COMPLETED
    current_stage_num = list(TriageStage).index(session.stage) + 1
    progress = f"Step {current_stage_num}/{total_stages}"
    
    return ChatResponse(
        session_id=sess_id,
        reply=reply,
        finished=session.completed,
        stage=session.stage.value,
        progress=progress,
        extracted_info={
            "data_collected": session.data,
            "red_flags": session.red_flags_detected,
            "timestamp": db_session.created_at.isoformat()
        }
    )

def check_emergency_keywords(message: str) -> bool:
    """Check for emergency/red flag symptoms."""
    message_lower = message.lower()
    return any(red_flag in message_lower for red_flag in RED_FLAG_SYMPTOMS)

def generate_emergency_response() -> str:
    """Generate emergency response for red flag symptoms."""
    return (
        "EMERGENCY ALERT\n\n"
        "Your symptoms require immediate medical attention.\n\n"
        "DO NOT DELAY:\n"
        "• Call 112 immediately\n"
        "• Go to nearest Emergency Room\n"
        "• Do not drive yourself\n\n"
        "This cannot wait."
    )

async def process_triage_stage(session: TriageSession, message: str) -> str:
    """Process message based on current triage stage."""
    
    if session.stage == TriageStage.GREETING:
        return handle_greeting(session, message)
    elif session.stage == TriageStage.CONSENT:
        return handle_consent(session, message)
    elif session.stage == TriageStage.DEMOGRAPHICS:
        return handle_demographics(session, message)
    elif session.stage == TriageStage.MEDICAL_HISTORY:
        return handle_medical_history(session, message)
    elif session.stage == TriageStage.MAIN_SYMPTOMS:
        return handle_main_symptoms(session, message)
    elif session.stage == TriageStage.SYMPTOM_DETAILS:
        return handle_symptom_details(session, message)
    elif session.stage == TriageStage.ASSOCIATED_SYMPTOMS:
        return handle_associated_symptoms(session, message)
    elif session.stage == TriageStage.SUMMARY_CONFIRMATION:
        return await handle_summary_confirmation(session, message)
    elif session.stage == TriageStage.FINAL_ASSESSMENT:
        return await generate_final_assessment(session)
    else:
        return "Thank you for using our medical triage system."

def handle_greeting(session: TriageSession, message: str) -> str:
    """Handle initial greeting and introduction."""
    session.stage = TriageStage.CONSENT
    return (
    "I will guide you through a structured assessment of your symptoms based on clinical protocols. This process typically takes 5-10 minutes.\n\n"
    "Important Disclaimers:\n"
    "• This is not a substitute for professional medical advice.\n"
    "• For emergencies, call 911 immediately.\n"
    "• This assessment is for informational purposes only.\n\n"
    "Do you consent to proceed with this medical assessment?"
)



def handle_consent(session: TriageSession, message: str) -> str:
    """Handle consent confirmation."""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["yes", "ok", "sure", "agree", "consent", "proceed"]):
        session.data["consent"] = True
        session.stage = TriageStage.DEMOGRAPHICS
        return "Please provide your age and biological sex (e.g., '32, female' or '45, male'):"
    elif any(word in message_lower for word in ["no", "don't", "refuse", "decline"]):
        session.completed = True
        return (
            "I understand. If you change your mind, feel free to start a new conversation.\n\n"
            "For immediate medical concerns, please contact your healthcare provider or call emergency services."
        )
    else:
        return (
            "Please respond with 'yes' if you consent to proceed with the medical assessment, "
            "or 'no' if you prefer not to continue."
        )


def handle_demographics(session: TriageSession, message: str) -> str:
    """Extract and validate demographic information."""
    
    # Extract age
    age_match = re.search(r'\b(\d{1,3})\b', message)
    
    # Extract sex with whole-word matching to avoid substring issues
    sex_patterns = {
        'male': [r'\bmale\b', r'\bman\b', r'\bm\b', r'\bboy\b'],  # \b ensures whole-word match
        'female': [r'\bfemale\b', r'\bwoman\b', r'\bf\b', r'\bgirl\b'],
        'other': [r'\bother\b', r'\bnon-binary\b', r'\bnonbinary\b', r'\btransgender\b']
    }
    
    detected_sex = None
    message_lower = message.lower()
    
    # Check patterns in a way that prefers exact matches (loop through all, but collect possibles and pick the best)
    possibles = []
    for sex, patterns in sex_patterns.items():
        for pattern in patterns:
            if re.search(pattern, message_lower):  # Use regex search for precision
                possibles.append(sex)
                break  # One match per category
    
    if possibles:
        # If multiple matches, prefer 'female' or 'other' over 'male' if explicit (or use first for simplicity)
        # For now, take the first, but you can add logic like prioritizing based on count
        detected_sex = possibles[0]
        # Optional: Count occurrences and pick the one with most mentions
        # counts = {s: sum(1 for p in sex_patterns[s] if re.search(p, message_lower)) for s in possibles}
        # detected_sex = max(counts, key=counts.get)
    
    if age_match and detected_sex:
        age = int(age_match.group(1))
        if 0 <= age <= 120:
            session.data["age"] = age
            session.data["sex"] = detected_sex
            session.stage = TriageStage.MEDICAL_HISTORY
            return (
                f"Recorded: {age} years old, {detected_sex}\n\n"
                "Medical History:\n"
                "Do you have any significant medical conditions, ongoing health issues, "
                "or take any medications regularly?\n\n"
                "Please include:\n"
                "• Chronic conditions (diabetes, heart disease, etc.)\n"
                "• Current medications\n"
                "• Known allergies\n"
                "• Recent surgeries or hospitalizations\n\n"
                "If none, simply say 'no' or 'none'."
            )
        else:
            return "Please provide a valid age between 0 and 120 years."
    else:
        return (
            "I need both your age and biological sex. Please provide both pieces of information.\n\n"
            "Examples:\n"
            "• 'I am 28 years old and female'\n"
            "• '35, male'\n"
            "• '42 year old woman'"
        )


def handle_medical_history(session: TriageSession, message: str) -> str:
    """Record medical history."""
    session.data["medical_history"] = message
    session.stage = TriageStage.MAIN_SYMPTOMS
    return (
    "✓ Medical history noted.\n\n"
    "Main Symptoms: Describe what you're experiencing, where, when it started, and severity (1-10):"
)


def handle_main_symptoms(session: TriageSession, message: str) -> str:
    """Record main symptoms and check for red flags."""
    session.data["main_symptoms"] = message
    
    # Check for red flags in the main symptoms
    if check_emergency_keywords(message):
        session.red_flags_detected = True
        session.completed = True
        return generate_emergency_response()
    
    session.stage = TriageStage.SYMPTOM_DETAILS
    return (
    "✓ Symptoms noted.\n\n"
    "Details: How long have you had this? Is it constant or intermittent? What makes it better/worse?"
)


def handle_symptom_details(session: TriageSession, message: str) -> str:
    """Record detailed symptom information."""
    session.data["symptom_details"] = message
    session.stage = TriageStage.ASSOCIATED_SYMPTOMS
    return (
    "✓ Details recorded.\n\n"
    "Other Symptoms: Any fever, nausea, dizziness, breathing issues, or other symptoms? (Say 'none' if not applicable)"
)


def handle_associated_symptoms(session: TriageSession, message: str) -> str:
    """Record associated symptoms and prepare summary."""
    session.data["associated_symptoms"] = message
    session.stage = TriageStage.SUMMARY_CONFIRMATION
    
    # Create summary for confirmation
    summary = create_patient_summary(session.data)
    
    return (
        "Associated symptoms recorded.\n\n"
        "Summary Confirmation:\n"
        "Let me confirm the information you've provided:\n\n"
        f"{summary}\n\n"
        "Is this information correct? Please respond with:\n"
        "• 'Yes' or 'Correct' if everything is accurate\n"
        "• 'No' or tell me what needs to be corrected"
    )

def create_patient_summary(data: Dict[str, Any]) -> str:
    """Create a formatted summary of patient information."""
    return f"""
Patient Summary:
• Age: {data.get('age', 'Not provided')} years
• Sex: {data.get('sex', 'Not provided')}
• Medical History: {data.get('medical_history', 'Not provided')}
• Main Symptoms: {data.get('main_symptoms', 'Not provided')}
• Symptom Details: {data.get('symptom_details', 'Not provided')}
• Associated Symptoms: {data.get('associated_symptoms', 'Not provided')}
"""

async def handle_summary_confirmation(session: TriageSession, message: str) -> str:
    """Handle summary confirmation and proceed to assessment."""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["yes", "correct", "accurate", "right", "confirm"]):
        session.stage = TriageStage.FINAL_ASSESSMENT
        return (
            "Information confirmed.\n\n"
            "Analyzing your symptoms...\n\n"
            "I'm now processing your information using clinical protocols and "
            "medical knowledge base to provide you with a comprehensive assessment.\n\n"
            "This may take a moment..."
            "Press any key to continue"
        )
    else:
        return (
            "Please let me know what information needs to be corrected, "
            "and I'll update your records accordingly."
        )

async def generate_final_assessment(session: TriageSession) -> str:
    symptoms_text = (
        f"{session.data.get('main_symptoms','')} "
        f"{session.data.get('associated_symptoms','')}"
    )
    kb = MedicalKnowledgeBase(vector_store, user_id=session.data.get("user_id"))
    kb_context = kb.get_context(symptoms_text, k=3)
    

    # comprehensive assessment prompt
    prompt = f"""
You are an AI medical assistant. Provide a concise clinical assessment.

PATIENT INFORMATION:
{create_patient_summary(session.data)}

RELEVANT MEDICAL KNOWLEDGE FROM DATABASE:
{kb_context}

Provide a structured clinical assessment following this exact format:

CLINICAL SUMMARY:
[Brief 2-3 sentence overview of the patient's presentation]

DIFFERENTIAL DIAGNOSIS:
1. Primary Consideration: [Most likely diagnosis] 
   - Clinical reasoning: [Why this is most likely]
   - Typical presentation: [How this condition usually presents]

2. Alternative Diagnosis: [Second possibility]
   - Clinical reasoning: [Why to consider this]
   - Key differentiating features: [What might distinguish this]

3. Less Likely Consideration: [Third possibility]
   - Brief rationale: [Why included in differential]

URGENCY ASSESSMENT:
Choose ONE and explain why:
- EMERGENT (Immediate care required - within minutes)
- URGENT (Same-day medical evaluation needed)
- SEMI-URGENT (Medical evaluation within 24-48 hours)
- NON-URGENT (Routine care appropriate)

CLINICAL RECOMMENDATIONS:

Immediate Actions:
• [Specific steps to take right now]
• [Any immediate symptom management]

Medical Care:
• [When to seek care and where]
• [What type of healthcare provider]
• [What to expect during the visit]

Self-Care Management:
• [Home care measures if appropriate]
• [Activity restrictions if any]
• [Symptom monitoring guidelines]

Red Flag Warning Signs:
• [Specific symptoms requiring immediate medical attention]
• [When to call 911 or go to ER]

FOLLOW-UP RECOMMENDATIONS:
• [When to return if symptoms persist]
• [Any specific tests or evaluations needed]

MEDICAL DISCLAIMER:
This assessment is for educational and informational purposes only. It does not constitute professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for definitive medical care. In case of emergency, call 911 immediately.

---
Assessment completed using clinical decision support tools and medical knowledge database.
"""
    
    try:
        response = model.generate_content(prompt)
        assessment = response.text.strip()
        
        # Mark session as completed
        session.completed = True
        session.stage = TriageStage.COMPLETED
        
        return assessment
        
    except Exception as e:
        logger.error(f"Error generating medical assessment: {e}")
        session.completed = True
        return (
            "I apologize, but I'm experiencing technical difficulties generating your assessment.\n\n"
            "Please consult with a healthcare professional immediately for proper evaluation.\n\n"
            "If you're experiencing concerning symptoms, contact:\n"
            "• Your primary care physician\n"
            "• Urgent care center\n"
            "• Emergency room (for severe symptoms)\n"
            "• Call 112 for emergencies"
        )

def extract_symptoms_from_text(text: str) -> List[str]:
    """Extract symptom keywords from text for knowledge base querying."""
    common_symptoms = [
        "headache", "fever", "cough", "pain", "nausea", "vomiting", "dizziness",
        "fatigue", "weakness", "shortness of breath", "chest pain", "abdominal pain",
        "diarrhea", "constipation", "rash", "swelling", "joint pain", "muscle pain"
    ]
    
    text_lower = text.lower()
    found_symptoms = [symptom for symptom in common_symptoms if symptom in text_lower]
    return found_symptoms

# Keep your existing endpoints
@router.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    current_user: DBUser = Depends(get_current_active_user)
):
    global vector_store
    docs = []
    for file in files:
        content = await file.read()
        path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        loader = PyPDFLoader(path)
        for doc in loader.load():
            # attach user metadata
            doc.metadata["user_id"] = current_user.id
            docs.append(doc)

    embeddings = HuggingFaceEmbeddings()
    if vector_store is None:
        vector_store = FAISS.from_documents(docs, embeddings)
    else:
        vector_store.add_documents(docs)

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)

    return {"message": "PDFs processed and knowledge base updated."}


@router.post("/ask", response_class=PlainTextResponse)
async def ask(
    prompt: str = Form(...),
    current_user: DBUser = Depends(get_current_active_user),
):
    if rag_system.vector_store is None:
        raise HTTPException(status_code=400, detail="Knowledge base is not ready. Upload documents first.")

    user_id = current_user.id
    kb = rag_system
    context = kb.get_context_for_query(prompt, user_id=user_id)

    if not context:
        return "I couldn't find relevant information in your uploaded documents. Please upload documents first or rephrase your question."

    prompt_template = f"""
You are a medical AI assistant with access to the user's medical documents:

Context:
{context}

User question:
{prompt}

Provide an accurate, concise answer based on the context above.
"""

    response = model.generate_content(prompt_template)

    return response.text.strip() or "No answer generated."



# Simple chat endpoint (existing functionality)
@router.post("/chat")
async def chat(
    files: Optional[List[UploadFile]] = File(None),
    query: Optional[str] = Form(None),
    current_user: DBUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    upload_results = []
    answer = None
    sources = []

    # Handle file uploads
    if files:
        uploads_folder = "uploads"
        os.makedirs(uploads_folder, exist_ok=True)

        for file in files:
            ext = file.filename.split(".")[-1].lower()
            if ext not in ('pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls'):
                upload_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Unsupported file type"
                })
                continue

            file_path = os.path.join(uploads_folder, f"{current_user.id}_{file.filename}")
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)

            try:
                res = await rag_system.process_uploaded_file(file_path, ext, current_user.id, db)
                res["filename"] = file.filename
                upload_results.append(res)
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                upload_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e)
                })

    # Handle query (if any)
    if query:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        user_id = current_user.id
        kb = rag_system
        context = kb.get_context_for_query(query, user_id=user_id)

        if not context:
            answer = "I couldn't find relevant information in your documents. Please upload documents or rephrase your question."
            sources = []
        else:
            prompt_template = f"""
You are a medical AI assistant with access to the user's uploaded medical documents below.

Context:
{context}

User question:
{query}

Please provide an accurate and concise answer based on the above context.
"""
            
            response = model.generate_content(prompt_template)
            answer = response.text.strip()

            # Provide source document snippets for transparency
            source_docs = kb.search_knowledge_base(query, k=3, user_id=user_id)
            sources = [
                {
                    "filename": d.metadata.get("file_path", "Unknown").split('/')[-1],
                    "chunk_index": d.metadata.get("chunk_index", 0),
                    "content_preview": d.page_content[:200] + "..."
                }
                for d in source_docs
            ]

    return {
        "upload_results": upload_results if upload_results else None,
        "query": query,
        "answer": answer,
        "sources": sources
    }
