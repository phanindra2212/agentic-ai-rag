import os
import streamlit as st
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from agents.workflow import execute_agentic_rag
from ui.streamlit_components import (
    render_sidebar_api_key,
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
from utils.logger import logger

# --- Page Config & Styling ---
st.set_page_config(
    page_title="Agentic Multi-Doc RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (HSL tailoring, slate backgrounds, clean typography)
st.markdown("""
<style>
    /* Primary Colors & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient */
    .header-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
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
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Chat Messages styling */
    .user-msg {
        background-color: #F1F5F9;
        color: #0F172A;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .assistant-msg {
        background-color: #FFFFFF;
        color: #0F172A;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* System Metrics Footer */
    .system-metrics-footer {
        border-top: 1px solid #E2E8F0;
        padding: 1rem 0;
        margin-top: 3rem;
        font-size: 0.85rem;
        color: #64748B;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query_metrics" not in st.session_state:
    st.session_state.last_query_metrics = {}

# --- Header Section ---
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🤖 Agentic Multi-Document RAG Assistant</h1>
    <p class="header-subtitle">Analyze files, extract citations, and query intelligence using LangGraph self-corrective agents and Gemini 2.5 Flash.</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar UI components ---
st.sidebar.markdown("# ⚙️ System Controls")
render_sidebar_api_key()
render_sidebar_uploader()
filters = render_sidebar_collection_stats()

# --- Main App Layout ---
# Use Streamlit tabs for Chat, Analytics, and logs
tabs = st.tabs(["💬 Chat Assistant", "📊 Performance Analytics", "📜 History Log"])

# --- Tab 1: Chat Assistant ---
with tabs[0]:
    # Check if API Key is set
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("⚠️ Google Gemini API Key is missing. Please enter it in the sidebar to initiate the assistant.")
        
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
