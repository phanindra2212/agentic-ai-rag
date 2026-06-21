import os
import uuid

# --- Startup Dependency Validation ---
try:
    from validate_environment import validate_env
    validate_env()
except Exception as e:
    import streamlit as st
    st.set_page_config(page_title="Dependency Error", page_icon="⚠️", layout="wide")
    st.error("## ⚠️ Startup Dependency Validation Failed")
    st.write("We detected critical environmental or package incompatibilities:")
    st.error(str(e))
    st.info("To fix this permanently, please ensure your environment matches the pinned versions in requirements.txt.")
    st.stop()

import streamlit as st

# Export Streamlit secrets to process environment variables for external libraries
try:
    for key in st.secrets:
        os.environ[key] = str(st.secrets[key])
except Exception:
    pass

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from agents.workflow import execute_agentic_rag  # noqa: E402
from ui.streamlit_components import (  # noqa: E402
    render_sidebar_uploader,
    render_sidebar_collection_stats,
    render_confidence_badge,
    render_query_expansions,
    render_citations,
    render_eval_metrics,
    render_analytics_dashboard,
    render_history_table,
    generate_txt_chat,
    generate_pdf_chat
)
from utils.logger import logger  # noqa: E402
from config.settings import cleanup_orphaned_sessions  # noqa: E402

# --- Page Config ---
st.set_page_config(
    page_title="Agentic Multi-Doc RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query_metrics" not in st.session_state:
    st.session_state.last_query_metrics = {}
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "light"

# Run background cleanup of inactive sessions
cleanup_orphaned_sessions()

# --- Sidebar Theme Controller ---
st.sidebar.markdown("### 🎨 Visual Theme")
theme_toggle = st.sidebar.toggle(
    "🌓 Enable Dark Theme",
    value=(st.session_state.app_theme == "dark"),
    help="Toggle between Light and Dark mode for the app."
)
new_theme = "dark" if theme_toggle else "light"
if new_theme != st.session_state.app_theme:
    st.session_state.app_theme = new_theme
    st.rerun()

# Apply Dynamic CSS based on theme
if st.session_state.app_theme == "dark":
    css_style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Main page layout */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #0F172A !important;
            color: #F8FAFC !important;
        }
        
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        /* Sidebar layout */
        [data-testid="stSidebar"], section[data-testid="stSidebar"], .stSidebar {
            background-color: #1E293B !important;
            border-right: 1px solid #334155 !important;
        }
        
        /* Sidebar text */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] li {
            color: #CBD5E1 !important;
        }
        
        /* Markdown / normal text styling */
        div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li, span, label, .stSubheader {
            color: #F8FAFC !important;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #F8FAFC !important;
            font-family: 'Inter', sans-serif;
        }
        
        /* Chat Messages Dark Mode styling */
        .user-msg {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border-left: 4px solid #3B82F6 !important;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid #334155;
        }
        
        .assistant-msg {
            background-color: #0F172A !important;
            color: #F8FAFC !important;
            border-left: 4px solid #10B981 !important;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3);
            border: 1px solid #334155;
        }
        
        /* Header Gradient Dark Theme */
        .header-container {
            background: linear-gradient(135deg, #1E1B4B 0%, #1E3A8A 100%) !important;
            padding: 2.5rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px -2px rgba(30, 58, 138, 0.4);
            border: 1px solid #1E40AF;
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.025em;
            color: white !important;
        }
        
        .header-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-top: 0.5rem;
            font-weight: 300;
            color: white !important;
        }
        
        /* Input elements, Selectboxes, Textareas */
        .stTextInput input, .stTextArea textarea, .stSelectbox [role="combobox"], [data-baseweb="select"] > div {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
        }
        
        div[role="listbox"], [data-baseweb="popover"] {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
        }
        
        /* Multiselect selections */
        .stMultiSelect div[role="button"] {
            background-color: #334155 !important;
            color: #F8FAFC !important;
        }
        
        /* File Uploader styling */
        [data-testid="stFileUploader"] {
            background-color: #1E293B !important;
            border: 1px dashed #334155 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid #334155 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94A3B8 !important;
            background-color: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            color: #3B82F6 !important;
            border-bottom-color: #3B82F6 !important;
        }
        
        /* Buttons styling */
        .stButton button, [role="button"], button[kind="secondary"] {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
        }
        .stButton button:hover, button[kind="secondary"]:hover {
            border-color: #3B82F6 !important;
            color: #3B82F6 !important;
        }
        
        /* Expanders styling */
        .st-emotion-cache-1h9z78m, .streamlit-expanderHeader {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 4px;
        }
        
        .system-metrics-footer {
            border-top: 1px solid #1E293B;
            padding: 1rem 0;
            margin-top: 3rem;
            font-size: 0.85rem;
            color: #94A3B8;
            text-align: center;
        }
    </style>
    """
else:
    css_style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Main page layout */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        /* Sidebar layout */
        [data-testid="stSidebar"], section[data-testid="stSidebar"], .stSidebar {
            background-color: #F1F5F9 !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        
        /* Sidebar text */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] li {
            color: #334155 !important;
        }
        
        /* Markdown / normal text styling */
        div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li, span, label, .stSubheader {
            color: #0F172A !important;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important;
            font-family: 'Inter', sans-serif;
        }
        
        /* Chat Messages Light Mode styling */
        .user-msg {
            background-color: #F1F5F9 !important;
            color: #0F172A !important;
            border-left: 4px solid #3B82F6 !important;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid #E2E8F0;
        }
        
        .assistant-msg {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-left: 4px solid #10B981 !important;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            border: 1px solid #E2E8F0;
        }
        
        /* Header Gradient Light Theme */
        .header-container {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important;
            padding: 2.5rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px -2px rgba(59, 130, 246, 0.3);
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.025em;
            color: white !important;
        }
        
        .header-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-top: 0.5rem;
            font-weight: 300;
            color: white !important;
        }
        
        /* Input elements, Selectboxes, Textareas */
        .stTextInput input, .stTextArea textarea, .stSelectbox [role="combobox"], [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
        }
        
        div[role="listbox"], [data-baseweb="popover"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
        }
        
        /* Multiselect selections */
        .stMultiSelect div[role="button"] {
            background-color: #E2E8F0 !important;
            color: #0F172A !important;
        }
        
        /* File Uploader styling */
        [data-testid="stFileUploader"] {
            background-color: #F8FAFC !important;
            border: 1px dashed #E2E8F0 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid #E2E8F0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #64748B !important;
            background-color: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            color: #2563EB !important;
            border-bottom-color: #2563EB !important;
        }
        
        /* Buttons styling */
        .stButton button, [role="button"], button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
        }
        .stButton button:hover, button[kind="secondary"]:hover {
            border-color: #2563EB !important;
            color: #2563EB !important;
        }
        
        /* Expanders styling */
        .st-emotion-cache-1h9z78m, .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 4px;
        }
        
        .system-metrics-footer {
            border-top: 1px solid #E2E8F0;
            padding: 1rem 0;
            margin-top: 3rem;
            font-size: 0.85rem;
            color: #64748B;
            text-align: center;
        }
    </style>
    """

st.markdown(css_style, unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🤖 Agentic Multi-Document RAG Assistant</h1>
    <p class="header-subtitle">Analyze files, extract citations, and query intelligence using LangGraph self-corrective agents and Gemini 2.5 Flash.</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar UI components ---
st.sidebar.markdown("# ⚙️ System Controls")

st.sidebar.markdown("### 🔑 API Key Status")

# Check if environment key exists
has_env_key = bool(os.getenv("GEMINI_API_KEY"))
has_custom_key = bool(st.session_state.get("custom_gemini_api_key"))

# Initialize state variables
if "show_key_input" not in st.session_state:
    st.session_state.show_key_input = not has_env_key

if has_custom_key:
    # Custom key is active
    st.sidebar.success("🛡️ Custom Override Active")
    if st.sidebar.button("🗑️ Clear Override", use_container_width=True):
        st.session_state.custom_gemini_api_key = ""
        st.session_state.show_key_input = not has_env_key
        st.rerun()
elif has_env_key:
    # System environment key is active
    st.sidebar.info("🔒 Configured via Environment")
    if not st.session_state.show_key_input:
        if st.sidebar.button("✏️ Override Key", use_container_width=True):
            st.session_state.show_key_input = True
            st.rerun()
else:
    # No key configured
    st.sidebar.warning("⚠️ No API Key Configured")

# Render input field if requested or if no key is present at all
if st.session_state.show_key_input:
    custom_key = st.sidebar.text_input(
        "Enter Custom Gemini API Key",
        value="",
        type="password",
        placeholder="Enter API Key",
        help="Paste a custom Gemini API key to override the system key. This is processed strictly on the server and is never exposed."
    )
    if custom_key:
        st.session_state.custom_gemini_api_key = custom_key.strip()
        st.session_state.show_key_input = False
        st.rerun()
        
    if has_env_key:
        if st.sidebar.button("Cancel Override", use_container_width=True):
            st.session_state.show_key_input = False
            st.rerun()

render_sidebar_uploader()
filters = render_sidebar_collection_stats()

# --- Main App Layout ---
# Use Streamlit tabs for Chat, Analytics, and logs
tabs = st.tabs(["💬 Chat Assistant", "📊 Performance Analytics", "📜 History Log"])

# --- Tab 1: Chat Assistant ---
with tabs[0]:
    # Check if API Key is set
    from config.settings import get_gemini_api_key
    if not get_gemini_api_key():
        st.warning("⚠️ Google Gemini API Key is missing. Please enter your Gemini API Key in the sidebar or ensure the GEMINI_API_KEY environment variable is configured.")
        
    # Render chat history
    chat_container = st.container()
    with chat_container:
        for idx, turn in enumerate(st.session_state.chat_history):
            if turn["role"] == "user":
                st.markdown(f'<div class="user-msg"><b>👤 You:</b><br/>{turn["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="assistant-msg"><b>🤖 Assistant:</b><br/>{turn["content"]}</div>', 
                    unsafe_allow_html=True
                )
                
                # Render metadata badges: Confidence and Complexity
                col_badge1, col_badge2, _ = st.columns([1, 1, 4])
                with col_badge1:
                    render_confidence_badge(turn.get("confidence_score", "Low Confidence"))
                with col_badge2:
                    st.caption(f"🧠 Complexity: **{turn.get('complexity', 'simple').upper()}**")
                    
                # Render Query Expansions if they exist
                render_query_expansions(turn.get("generated_queries", []))
                
                # Render Citations (sources)
                render_citations(turn.get("citations", []))
                
                # Render In-Line Evaluations
                render_eval_metrics(turn.get("eval_metrics", {}))
                
                st.markdown("<hr style='margin: 1.5rem 0; border: 0; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    # Chat Input block
    st.markdown("### 💬 Ask a Question")
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("Enter your question based on the uploaded documents:")
        submit_button = st.form_submit_button("Send Query", use_container_width=True)
        
    if submit_button and user_question:
        if not user_question.strip():
            st.error("Please enter a valid question.")
        else:
            # 1. Append User Turn to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # 2. Execute RAG Workflow via LangGraph
            with st.spinner("🤖 Executing Agentic self-correcting workflow (Query Expansion -> Retrieval -> Optimization -> Gemini -> Validation)..."):
                try:
                    # Pass the turn list
                    # Format history to pass to Graph State
                    history_list = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history[:-1]
                    ]
                    
                    results = execute_agentic_rag(
                        question=user_question,
                        chat_history=history_list,
                        filters=filters
                    )
                    
                    # 3. Append Assistant response and metrics
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": results["answer"],
                        "citations": results["citations"],
                        "confidence_score": results["confidence_score"],
                        "confidence_val": results["confidence_val"],
                        "complexity": results["complexity"],
                        "generated_queries": results["generated_queries"],
                        "eval_metrics": results["eval_metrics"]
                    })
                    
                    # Store latest metrics for footer display
                    st.session_state.last_query_metrics = {
                        "total_time": results["total_time"],
                        "retrieval_time": results["retrieval_time"],
                        "response_time": results["response_time"],
                        "chunks_count": results["chunks_count"]
                    }
                    
                except Exception as e:
                    st.error(f"Failed to execute RAG pipeline: {e}")
                    logger.error(f"Error in UI query form submission: {e}", exc_info=True)
                    
            st.rerun()

    # Chat export buttons
    if st.session_state.chat_history:
        st.markdown("### 📤 Download Chat Transcript")
        col_txt, col_pdf, _ = st.columns([1, 1, 4])
        
        # 1. TXT Export
        txt_content = generate_txt_chat(st.session_state.chat_history)
        col_txt.download_button(
            label="📥 Download TXT",
            data=txt_content,
            file_name=f"rag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        # 2. PDF Export
        pdf_bytes = generate_pdf_chat(st.session_state.chat_history)
        col_pdf.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=f"rag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

# --- Tab 2: Analytics Dashboard ---
with tabs[1]:
    render_analytics_dashboard()

# --- Tab 3: History Log ---
with tabs[2]:
    render_history_table()

# --- Footer Section ---
metrics = st.session_state.last_query_metrics
if metrics:
    metrics_str = (
        f"Last Query Performance Metrics: "
        f"⏱️ Total Latency: {metrics['total_time']:.3f}s | "
        f"🔍 DB Retrieval: {metrics['retrieval_time']:.3f}s | "
        f"✍️ Gemini Generation: {metrics['response_time']:.2f}s | "
        f"📄 Retrieved Chunks: {metrics['chunks_count']}"
    )
else:
    metrics_str = "No active query metrics logged yet. Submit a query to see performance metrics."

st.markdown(f"""
<div class="system-metrics-footer">
    <p>{metrics_str}</p>
    <p>Agentic Multi-Document RAG Knowledge Assistant • Tech Stack: Streamlit, LangChain, LangGraph, Gemini 2.5 Flash, ChromaDB</p>
</div>
""", unsafe_allow_html=True)
