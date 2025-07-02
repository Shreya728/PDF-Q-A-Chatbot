# app.py
import streamlit as st
import sys
import os
import time
from groq import Groq
from database import ChromaVectorDatabase
from utils import process_attachment, login_user_base64 as login_user, register_user_base64 as register_user, save_chat_history, get_chat_history, get_user_chats, log_user_activity, log_file_processing, init_database, delete_chat_history
from langchain.docstore.document import Document
from dotenv import load_dotenv
import logging
import psycopg2  # Added to fix import error

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/pdf_chatbot")
logger.info(f"Loaded DATABASE_URL: {DATABASE_URL}")  # Debug print for URL
if not GROQ_API_KEY:
    st.error("🔑 GROQ_API_KEY not found in .env file")
    st.stop()

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized successfully")
except Exception as e:
    st.error(f"❌ Failed to initialize Groq client: {str(e)}")
    logger.error(f"Groq initialization error: {str(e)}")
    st.stop()

# Initialize vector database
if "vector_db" not in st.session_state:
    try:
        st.session_state.vector_db = ChromaVectorDatabase(persist_directory="/data/chroma_db")  # Adjusted for Render's filesystem
        logger.info("Vector database initialized")
    except Exception as e:
        st.error(f"❌ Failed to initialize vector database: {str(e)}")
        logger.error(f"Vector database initialization error: {str(e)}")
        st.stop()

# Initialize database connection
if "db_connection" not in st.session_state:
    try:
        st.session_state.db_connection = psycopg2.connect(DATABASE_URL)
        logger.info("Database connection established and stored in session state")
    except Exception as e:
        st.error(f"❌ Failed to establish database connection: {str(e)}")
        logger.error(f"Database connection error: {str(e)}")
        st.stop()

# Function to get database connection from session state
def get_db_connection():
    return st.session_state.db_connection

# Initialize session state with default values
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = 1
if "current_files" not in st.session_state:
    st.session_state.current_files = []
if "current_files_id" not in st.session_state:
    st.session_state.current_files_id = None

# Database setup with error handling
try:
    init_database()  # Uses the session connection
    logger.info("Database initialized successfully")
except Exception as e:
    st.error(f"❌ Failed to initialize database: {str(e)}")
    logger.error(f"Database initialization error: {str(e)}")
    st.stop()

# Custom CSS with white background and black font
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: white;
        color: black;
        min-height: 100vh;
    }
    
    .chat-container {
        padding: 1rem;
        margin: 1rem 0;
        max-height: 60vh;
        overflow-y: auto;
        background: #f9f9f9;
        border-radius: 15px;
        border: 1px solid #ddd;
    }
    
    .user-message {
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-left: 20%;
        background: #e6e6fa;
        color: black;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .ai-message {
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-right: 20%;
        background: #f0f0f0;
        color: black;
        border: 1px solid #ccc;
        position: relative;
    }
    
    .message-timestamp {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 4px;
        text-align: right;
    }
    
    .source-info {
        font-size: 0.8rem;
        opacity: 0.8;
        font-style: italic;
        margin-top: 4px;
        padding-top: 4px;
        border-top: 1px solid #ccc;
    }
    
    .sidebar-container {
        padding: 1rem;
        margin: 1rem 0;
        background: #f9f9f9;
        border-radius: 15px;
        border: 1px solid #ddd;
    }
    
    .chat-selector {
        background: #f9f9f9;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #ccc;
    }
    
    .stats-card {
        background: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #ccc;
        text-align: center;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
    }
    
    .stButton > button {
        background: #e6e6fa;
        color: black;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #d0d0f0;
        border-color: #999;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stTextInput > div > div > input {
        background: #f9f9f9;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 0.5rem;
        color: black;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #666;
    }
    
    .stSelectbox > div > div > select {
        background: #f9f9f9;
        color: black;
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        margin: 10px 0;
        margin-right: 20%;
    }
    
    .typing-dots {
        padding: 15px;
        background: #f0f0f0;
        border-radius: 18px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #333;
        animation: typing 1.4s infinite ease;
    }
    
    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    .welcome-text {
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
        padding: 2rem;
        color: black;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 1.8rem; /* Reduced size */
        font-family: 'Comic Neue', cursive; /* Quirky font */
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        color: black;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .success-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: #e0ffe0;
        color: black;
        border: 1px solid #99cc99;
        animation: successPulse 0.6s ease-out;
    }
    
    @keyframes successPulse {
        0% { transform: scale(0.95); opacity: 0; }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .error-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: #ffe0e0;
        color: black;
        border: 1px solid #cc9999;
    }
    
    .info-badge {
        background: #fff3e0;
        color: #ff9900;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        margin-left: 8px;
        border: 1px solid #ffcc99;
    }
    
    .username-box {
        background: #e6e6fa;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ccc;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    @media (max-width: 768px) {
        .user-message, .ai-message {
            margin-left: 5%;
            margin-right: 5%;
        }
        .main-title { font-size: 1.5rem; }
        .welcome-text { font-size: 1.5rem; padding: 1rem; }
        .sidebar-container { padding: 0.5rem; }
    }
    
    .stChatInput {
        position: sticky;
        bottom: 0;
        background: #f9f9f9;
        border-top: 1px solid #ccc;
        padding: 10px;
        border-radius: 15px 15px 0 0;
    }
    
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def show_typing_indicator():
    """Show enhanced typing indicator."""
    st.markdown("""
    <div class="typing-indicator">
        <div class="typing-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def new_chat():
    """Start a new chat session."""
    st.session_state.messages = []
    st.session_state.chat_id = max(get_user_chats(st.session_state.user), default=0) + 1
    st.session_state.current_files = []
    st.session_state.current_files_id = None
    st.session_state.vector_db.clear_database()
    log_user_activity(st.session_state.user, "new_chat", f"chat_id: {st.session_state.chat_id}")
    st.rerun()

def load_selected_chat(selected_chat_id: int):
    """Load a previously selected chat."""
    if selected_chat_id != st.session_state.chat_id:
        st.session_state.chat_id = selected_chat_id
        st.session_state.messages = []
        history = get_chat_history(st.session_state.user, selected_chat_id)
        for entry in history:
            if entry['user_message']:
                st.session_state.messages.append({
                    "role": "user",
                    "content": entry['user_message'],
                    "timestamp": entry['timestamp']
                })
            if entry['bot_response']:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": entry['bot_response'],
                    "timestamp": entry['timestamp'],
                    "sources": entry.get('file_sources', [])
                })
        log_user_activity(st.session_state.user, "load_chat", f"chat_id: {selected_chat_id}")
        st.rerun()

def export_chat_history():
    """Export current chat history as a TXT file."""
    chat_content = f"Chat ID: {st.session_state.chat_id}\nUser: {st.session_state.user}\nExport Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    for msg in st.session_state.messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        timestamp = format_timestamp(msg.get("timestamp", ""))
        chat_content += f"{role}: {content} [{timestamp}]\n"
        if role == "Assistant" and msg.get("sources"):
            chat_content += f"Sources: {', '.join(msg['sources'])}\n"
        chat_content += "-" * 50 + "\n"
    st.download_button(
        label="📥 Export Chat History",
        data=chat_content,
        file_name=f"chat_export_{st.session_state.chat_id}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        key=f"export_chat_{st.session_state.chat_id}"
    )
    log_user_activity(st.session_state.user, "export_chat", f"chat_id: {st.session_state.chat_id}")

def delete_chat():
    """Delete the current chat."""
    if st.button("🗑️ Delete Chat", key="delete_chat_btn"):
        delete_chat_history(st.session_state.user, st.session_state.chat_id)
        st.session_state.messages = []
        st.session_state.chat_id = max(get_user_chats(st.session_state.user), default=0) + 1
        st.session_state.current_files = []
        st.session_state.current_files_id = None
        st.session_state.vector_db.clear_database()
        st.success("✅ Chat deleted successfully!")
        log_user_activity(st.session_state.user, "delete_chat", f"chat_id: {st.session_state.chat_id}")
        st.rerun()

def format_timestamp(timestamp_str):
    """Format timestamp for display."""
    try:
        if timestamp_str:
            return time.strftime("%I:%M %p", time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S"))
        return ""
    except:
        return ""

def detect_query_intent(query: str) -> dict:
    """Detect intent based on keywords."""
    query_lower = query.lower()
    intents = {
        "summarize": ["summarize", "summary", "overview", "brief"],
        "search": ["find", "search", "locate", "where"],
        "explain": ["explain", "what is", "how does", "clarify"],
        "compare": ["compare", "difference", "contrast", "vs"]
    }
    detected_intent = "general"
    for intent, keywords in intents.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_intent = intent
            break
    return {"intent": detected_intent, "query": query}

def create_dynamic_prompt(context: str, user_input: str, chat_history: list = None, file_sources: list = None) -> str:
    """Create dynamic prompt based on intent."""
    intent_data = detect_query_intent(user_input)
    intent = intent_data["intent"]
    history_context = "\n\nPrevious context:\n" + "\n".join([f"{msg['role'].capitalize()}: {msg['content'][:200]}..." for msg in chat_history[-6:]]) if chat_history else ""
    sources_context = f"\n\nSources: {', '.join(file_sources)}" if file_sources else ""
    templates = {
        "summarize": f"Summarize the following:\n{context}\nUser: {user_input}{history_context}{sources_context}\nSummary:",
        "search": f"Find information in:\n{context}\nUser: {user_input}{history_context}{sources_context}\nResult:",
        "explain": f"Explain based on:\n{context}\nUser: {user_input}{history_context}{sources_context}\nExplanation:",
        "compare": f"Compare using:\n{context}\nUser: {user_input}{history_context}{sources_context}\nComparison:",
        "general": f"Answer using:\n{context}\nUser: {user_input}{history_context}{sources_context}\nResponse:"
    }
    return templates.get(intent, templates["general"])

def get_relevant_context(user_input: str, k: int = 5) -> tuple:
    """Get relevant context using vector similarity."""
    try:
        if st.session_state.current_files:
            top_docs = st.session_state.vector_db.similarity_search(user_input, k=k)
            if top_docs:
                context_parts = []
                sources = []
                for doc in top_docs:
                    filename = doc.metadata.get("filename", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    context_parts.append(f"[Source: {filename}, Page: {page}]\n{doc.page_content}")
                    if filename not in sources:
                        sources.append(filename)
                return "\n\n".join(context_parts), sources
            return "No relevant context found.", []
        return "No files uploaded.", []
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        return f"Error: {str(e)}", []

def generate_response(user_input: str) -> tuple:
    """Generate LLM response with context and memory."""
    if not st.session_state.current_files:
        return "Please upload a file to enable chatting.", []
    try:
        context, sources = get_relevant_context(user_input)
        recent_history = st.session_state.messages[-6:] if st.session_state.messages else []
        prompt = create_dynamic_prompt(context, user_input, recent_history, sources)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1500
        ).choices[0].message.content
        return response, sources
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}", []

def display_chat_message(message: dict):
    """Display a chat message with timestamp and sources."""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")
    sources = message.get("sources", [])
    formatted_time = format_timestamp(timestamp)
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            {content}
            <div class="message-timestamp">{formatted_time}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        source_info = f'<div class="source-info">📎 Sources: {", ".join(sources)}</div>' if sources else ""
        st.markdown(f"""
        <div class="ai-message">
            {content}
            {source_info}
            <div class="message-timestamp">{formatted_time}</div>
        </div>
        """, unsafe_allow_html=True)

def get_user_analytics(username: str) -> dict:
    """Get user analytics using PostgreSQL."""
    try:
        conn = get_db_connection()  # Use the session connection
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM user_activity WHERE username = %s", (username,))
        total_activities = c.fetchone()[0]
        c.execute("SELECT COUNT(DISTINCT chat_id) FROM chat_history WHERE username = %s", (username,))
        total_chats = c.fetchone()[0]
        conn.commit()  # Ensure changes are committed if any
        return {"total_activities": total_activities, "total_chats": total_chats}
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        return {"total_activities": 0, "total_chats": 0}

def main_chat_page():
    """Main chat page with all enhancements."""
    load_css()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="main-title">Welcome to PDF Chatbot</div>', unsafe_allow_html=True)
    with col2:
        st.write("")  # Removed "👤 Pradeep"
    with col3:
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.user = None
            st.session_state.page = "login"
            st.rerun()

    with st.sidebar:
        st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
        if st.session_state.user:
            st.markdown(f'<div class="username-box">👤 Logged in as: {st.session_state.user}</div>', unsafe_allow_html=True)

        st.markdown("### 📁 Upload Files")
        uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, key="file_uploader")  # Changed to accept_multiple_files=True
        if uploaded_files:
            if any(file.size > 10 * 1024 * 1024 for file in uploaded_files):  # 10MB limit per file
                st.error("One or more files exceed 10MB limit")
            else:
                current_files_id = hash(tuple(file.name for file in uploaded_files))  # Unique ID based on all filenames
                if st.session_state.current_files_id != current_files_id:
                    st.session_state.current_files = [file.name for file in uploaded_files]
                    st.session_state.current_files_id = current_files_id
                    st.session_state.vector_db.clear_database()
                    all_docs = []
                    for uploaded_file in uploaded_files:
                        docs = process_attachment(uploaded_file)
                        if docs:
                            all_docs.extend(docs)
                            log_file_processing(st.session_state.user, uploaded_file.name, uploaded_file.size, "success")
                    if all_docs:
                        st.session_state.vector_db.add_documents(all_docs)
                    st.success(f"✅ {len(uploaded_files)} file(s) processed!")
                    log_user_activity(st.session_state.user, "file_upload", f"files: {len(uploaded_files)}")

        st.markdown("### 💬 Chat Management")
        chats = get_user_chats(st.session_state.user) if st.session_state.user else []
        if chats:
            options = [f"Chat {chat_id}" for chat_id in chats[:10]]
            selected_chat = st.selectbox("Select Chat", options, index=None, key="chat_selector")
            if selected_chat and st.button("📂 Load", key="load_chat_btn"):
                load_selected_chat(int(selected_chat.split()[1]))
        if st.button("🆕 New Chat", key="new_chat_btn"):
            new_chat()
        if st.button("📤 Export", key="export_chat_btn"):
            export_chat_history()
        delete_chat()

        st.markdown("### 📊 Analytics")
        analytics = get_user_analytics(st.session_state.user) if st.session_state.user else {"total_activities": 0, "total_chats": 0}
        st.markdown(f'<div class="stats-card"><div class="stats-number">{analytics["total_chats"]}</div>Total Chats</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stats-card"><div class="stats-number">{analytics["total_activities"]}</div>Total Activities</div>', unsafe_allow_html=True)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        display_chat_message(message)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.current_files:
        user_input = st.chat_input("💬 Ask about the file...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
            display_chat_message(st.session_state.messages[-1])
            typing_placeholder = st.empty()
            with typing_placeholder:
                show_typing_indicator()
            response, sources = generate_response(user_input)
            typing_placeholder.empty()
            assistant_message = {"role": "assistant", "content": response, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": sources}
            st.session_state.messages.append(assistant_message)
            display_chat_message(assistant_message)
            save_chat_history(st.session_state.user, user_input, response, st.session_state.chat_id, sources)
            log_user_activity(st.session_state.user, "successful_query", f"chat_id: {st.session_state.chat_id}")
            st.rerun()
    else:
        st.warning("⚠️ Please upload a file to start chatting.")

def login_page():
    """Login page."""
    load_css()
    st.markdown('<div class="main-title">Welcome to PDF Q&A BOT</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("👤 Username", key="login_username")
        password = st.text_input("🔒 Password", type="password", key="login_password")
        if st.button("🚀 Login", key="login_submit"):
            if username and password:
                if login_user(username, password):
                    st.session_state.user = username
                    st.session_state.page = "main"
                    st.session_state.messages = []  # Clear messages on login
                    st.session_state.chat_id = 1  # Reset to default chat ID
                    log_user_activity(username, "login", "successful")
                    st.success("✅ Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
                    log_user_activity(username, "login", "failed")
            else:
                st.warning("⚠️ Fill all fields")
        if st.button("📝 Register", key="go_to_register"):
            st.session_state.page = "register"
            st.rerun()

def register_page():
    """Register page."""
    load_css()
    st.markdown('<div class="main-title">📝 Register</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("👤 Username", key="register_username")
        password = st.text_input("🔒 Password", type="password", key="register_password")
        confirm_password = st.text_input("🔒 Confirm Password", type="password", key="confirm_password")
        if st.button("🎉 Create Account", key="register_submit"):
            if username and password and confirm_password:
                if password == confirm_password:
                    if register_user(username, password):
                        st.success("✅ Account created!")
                        log_user_activity(username, "register", "successful")
                        time.sleep(1)
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.error("❌ Username exists")
                        log_user_activity(username, "register", "failed")
                else:
                    st.error("❌ Passwords mismatch")
            else:
                st.warning("⚠️ Fill all fields")
        if st.button("🔙 Back", key="back_to_login"):
            st.session_state.page = "login"
            st.rerun()

# Page routing
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "main" and st.session_state.user:
    main_chat_page()
else:
    st.session_state.page = "login"  # Default to login page if not set
    st.rerun()
