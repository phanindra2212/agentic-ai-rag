import io
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from rag.ingest import get_collection_statistics, remove_document_from_db, process_and_index_file
from utils.metrics import get_analytics, clear_all_metrics
from utils.logger import logger

# Helper to generate TXT chat export
def generate_txt_chat(chat_history: List[Dict[str, Any]]) -> str:
    """Formats chat history into a downloadable plain text document."""
    lines = []
    lines.append("="*50)
    lines.append(" RAG KNOWLEDGE ASSISTANT - CHAT TRANSCRIPT")
    lines.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("="*50)
    lines.append("")
    
    for turn in chat_history:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "")
        lines.append(f"[{role}]")
        lines.append(content)
        
        if role == "ASSISTANT" and turn.get("citations"):
            lines.append("\nSources:")
            for cite in turn["citations"]:
                lines.append(f"- {cite['file_name']} (Page {cite['page_number']})")
        lines.append("-"*50)
        lines.append("")
        
    return "\n".join(lines)

# Helper to generate PDF chat export using ReportLab
def generate_pdf_chat(chat_history: List[Dict[str, Any]]) -> bytes:
    """Formats chat history into a clean, downloadable PDF."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=54, 
            leftMargin=54,
            topMargin=54, 
            bottomMargin=54
        )
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            spaceAfter=15,
            textColor=colors.HexColor('#0F172A') # slate-900
        )
        
        user_q_style = ParagraphStyle(
            'UserQuestion',
            parent=styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=11,
            spaceBefore=12,
            spaceAfter=4,
            textColor=colors.HexColor('#2563EB') # blue-600
        )
        
        assistant_a_style = ParagraphStyle(
            'AssistantAnswer',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=14,
            spaceAfter=8,
            textColor=colors.HexColor('#334155') # slate-700
        )
        
        citation_style = ParagraphStyle(
            'Citations',
            parent=styles['Italic'],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#64748B') # slate-500
        )
        
        story = [
            Paragraph("RAG Knowledge Assistant - Chat History", title_style),
            Paragraph(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
            Spacer(1, 15)
        ]
        
        for turn in chat_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            if role == "user":
                story.append(Paragraph(f"<b>User Question:</b> {content}", user_q_style))
                story.append(Spacer(1, 4))
            else:
                # Replace newlines with linebreaks for PDF layout
                cleaned_content = content.replace("\n", "<br/>")
                story.append(Paragraph(f"<b>Assistant:</b> {cleaned_content}", assistant_a_style))
                
                citations = turn.get("citations", [])
                if citations:
                    citation_lines = ["<b>Sources:</b>"]
                    for cite in citations:
                        citation_lines.append(f"• {cite['file_name']} (Page {cite['page_number']})")
                    story.append(Paragraph("<br/>".join(citation_lines), citation_style))
                story.append(Spacer(1, 10))
                
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error compiling PDF: {e}")
        # Return fallback text in bytes
        fallback_txt = f"PDF compilation failed: {e}\n\n" + generate_txt_chat(chat_history)
        return fallback_txt.encode("utf-8")

def render_sidebar_api_key() -> None:
    """Renders API Key configuration section in sidebar."""
    st.sidebar.markdown("### 🔑 API Authentication")
    api_key = os.getenv("GEMINI_API_KEY", "")
    
    # Text input for API Key
    user_key = st.sidebar.text_input(
        "Gemini API Key",
        value=api_key,
        type="password",
        help="Input your Gemini API key. If already set as an environment variable, you can leave this blank."
    )
    
    if user_key:
        os.environ["GEMINI_API_KEY"] = user_key

def render_sidebar_uploader() -> None:
    """Renders multi-file uploader component in sidebar."""
    st.sidebar.markdown("### 📂 Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents (Max 10)",
        type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, PPTX, TXT. Up to 10 files simultaneously."
    )
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.sidebar.error("You can only upload a maximum of 10 files at once.")
            uploaded_files = uploaded_files[:10]
            
        temp_dir = Path("data/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Ingestion button
        if st.sidebar.button("⚙️ Process Documents", use_container_width=True):
            status_box = st.sidebar.empty()
            progress_bar = st.sidebar.progress(0.0)
            
            successful_uploads = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_box.markdown(f"*Processing {uploaded_file.name}...*")
                
                # Write to temp uploads folder
                file_path = temp_dir / uploaded_file.name
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # Process and index
                    chunks_count, pages_count = process_and_index_file(str(file_path))
                    successful_uploads.append({
                        "name": uploaded_file.name,
                        "chunks": chunks_count,
                        "pages": pages_count
                    })
                except Exception as e:
                    st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")
                    logger.error(f"Error during file upload processing: {e}", exc_info=True)
                    
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_box.empty()
            progress_bar.empty()
            
            if successful_uploads:
                st.sidebar.success(f"Successfully processed {len(successful_uploads)} file(s)!")
                # Show details of ingested files
                for f_info in successful_uploads:
                    st.sidebar.caption(f"✓ **{f_info['name']}** ({f_info['pages']} pages, {f_info['chunks']} chunks)")
                st.rerun()

def render_sidebar_collection_stats() -> Dict[str, Any]:
    """Renders details of documents currently stored in Chroma vector store.
    
    Returns:
        Dict representing filtering states.
    """
    st.sidebar.markdown("### 🗄️ Database Collections")
    
    stats = get_collection_statistics()
    st.sidebar.metric("Total Files Indexed", stats["total_documents"])
    st.sidebar.metric("Total Context Chunks", stats["total_chunks"])
    
    doc_list = stats["document_list"]
    
    # Filters setup
    selected_files = []
    selected_types = []
    
    if doc_list:
        with st.sidebar.expander("Manage Stored Files"):
            for doc in doc_list:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"**{doc['file_name']}**\n*{doc['file_type']} • {doc['pages']} pgs • {doc['chunks']} chks*")
                
                # Trash button to delete individual document
                if col2.button("🗑️", key=f"del_{doc['file_name']}"):
                    try:
                        remove_document_from_db(doc["file_name"])
                        # Delete temp file if exists
                        temp_path = Path("data/uploads") / doc["file_name"]
                        if temp_path.exists():
                            os.remove(temp_path)
                        st.success(f"Deleted {doc['file_name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Deletion failed: {e}")
                        
        # Filtering parameters for retrieval
        st.sidebar.markdown("### 🔍 Search Filtering")
        
        # File selector filter
        file_names_available = [doc["file_name"] for doc in doc_list]
        selected_files = st.sidebar.multiselect(
            "Search within files",
            options=file_names_available,
            default=[],
            help="Leave blank to search across all documents."
        )
        
        # File type filter
        file_types_available = list(set(doc["file_type"] for doc in doc_list))
        selected_types = st.sidebar.multiselect(
            "Search by file type",
            options=file_types_available,
            default=[],
            help="Filter retrieval results by file extension."
        )
        
    # Top K configuration
    top_k = st.sidebar.slider(
        "Top-K Retrieve Chunks",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        help="Number of relevant text chunks retrieved for generation."
    )
    
    return {
        "file_names": selected_files,
        "file_types": selected_types,
        "top_k": top_k
    }

def render_confidence_badge(confidence: str) -> None:
    """Renders confidence pill with specific color code matching classification."""
    if confidence == "High Confidence":
        st.markdown(
            '<span style="background-color: #DEF7EC; color: #03543F; font-size: 0.8rem; '
            'font-weight: 600; padding: 4px 10px; border-radius: 12px; border: 1px solid #BCF0DA;">'
            '🟢 High Confidence</span>',
            unsafe_allow_html=True
        )
    elif confidence == "Medium Confidence":
        st.markdown(
            '<span style="background-color: #FEF08A; color: #713F12; font-size: 0.8rem; '
            'font-weight: 600; padding: 4px 10px; border-radius: 12px; border: 1px solid #FDE047;">'
            '🟡 Medium Confidence</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span style="background-color: #FDE8E8; color: #9B1C1C; font-size: 0.8rem; '
            'font-weight: 600; padding: 4px 10px; border-radius: 12px; border: 1px solid #FBD5D5;">'
            '🔴 Low Confidence</span>',
            unsafe_allow_html=True
        )

def render_query_expansions(queries: List[str]) -> None:
    """Renders the query rewrites/expansions generated by the Query Agent."""
    if queries:
        with st.expander("🤖 Query Understanding Agent Expansions"):
            st.markdown("The Query Understanding Agent rewrote and expanded your prompt into the following searches:")
            for idx, q in enumerate(queries):
                st.markdown(f"{idx+1}. `\"{q}\"`")

def render_citations(citations: List[Dict[str, Any]]) -> None:
    """Renders citation accordions referencing source documents and contents."""
    if citations:
        st.markdown("#### 📚 Source Attribution")
        for idx, cite in enumerate(citations):
            with st.expander(f"[{idx+1}] {cite['file_name']} (Page {cite['page_number']})"):
                st.markdown(f"**Snippet:**\n> {cite['content_snippet']}")
                st.markdown(f"*Document Type: {cite['file_type']} | Source Path: `{cite['source']}`*")

def render_eval_metrics(metrics: Dict[str, float]) -> None:
    """Displays the RAG evaluation scores in an horizontal columns layout."""
    if metrics:
        st.markdown("#### 📊 RAG In-Line Evaluation")
        col1, col2, col3, col4 = st.columns(4)
        
        # Display each score (0.0 to 1.0) as percentages
        col1.metric("Context Precision", f"{metrics.get('context_precision', 0.0)*100:.0f}%", 
                    help="How relevant are the retrieved chunks to the question?")
        col2.metric("Context Recall", f"{metrics.get('context_recall', 0.0)*100:.0f}%", 
                    help="Are all facts needed to answer the question contained in the context?")
        col3.metric("Faithfulness", f"{metrics.get('faithfulness', 0.0)*100:.0f}%", 
                    help="Is the answer derived strictly from the context, without hallucinating?")
        col4.metric("Answer Relevance", f"{metrics.get('answer_relevance', 0.0)*100:.0f}%", 
                    help="How directly and fully does the answer address the question?")

def render_analytics_dashboard() -> None:
    """Renders the advanced metrics and evaluation history dashboard."""
    st.markdown("## 📊 System Performance & Quality Analytics")
    
    analytics = get_analytics()
    
    if analytics["total_queries"] == 0:
        st.info("No query logs available yet. Submit questions in the chat window to populate analytics.")
        return
        
    # High-level aggregate cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Queries", analytics["total_queries"])
    col2.metric("Avg Retrieval Time", f"{analytics['average_retrieval_time']:.3f}s")
    col3.metric("Avg Response Time", f"{analytics['average_response_time']:.2f}s")
    col4.metric("Avg Faithfulness", f"{analytics['average_faithfulness']*100:.0f}%")
    col5.metric("Avg Relevance", f"{analytics['average_answer_relevance']*100:.0f}%")
    
    history = analytics["search_history"]
    if not history:
        return
        
    # Build dataframe for timeseries plots
    df_data = []
    for idx, entry in enumerate(history):
        evals = entry.get("evaluation", {})
        df_data.append({
            "Query ID": idx + 1,
            "Question": entry["question"][:30] + "..." if len(entry["question"]) > 30 else entry["question"],
            "Retrieval Time (s)": entry["retrieval_time"],
            "Response Time (s)": entry["response_time"],
            "Context Precision": evals.get("context_precision", 0.0),
            "Context Recall": evals.get("context_recall", 0.0),
            "Faithfulness": evals.get("faithfulness", 0.0),
            "Answer Relevance": evals.get("answer_relevance", 0.0),
            "Timestamp": entry.get("timestamp", "")
        })
    df = pd.DataFrame(df_data)
    
    # 1. Latency Plot
    st.markdown("### ⏱️ Latency Analysis")
    fig_latency = px.line(
        df, 
        x="Query ID", 
        y=["Retrieval Time (s)", "Response Time (s)"],
        title="Execution Latency Over Successive Queries",
        markers=True,
        color_discrete_sequence=["#3B82F6", "#10B981"] # Blue and Green
    )
    fig_latency.update_layout(xaxis_title="Query Sequence Number", yaxis_title="Time (Seconds)")
    st.plotly_chart(fig_latency, use_container_width=True)
    
    # 2. Evaluation Scores History
    st.markdown("### 📈 RAG Alignment Scores Trend")
    fig_eval = px.line(
        df,
        x="Query ID",
        y=["Context Precision", "Context Recall", "Faithfulness", "Answer Relevance"],
        title="Quality Scores Trends (LLM-Grounded Evaluation)",
        markers=True,
        color_discrete_sequence=["#EC4899", "#8B5CF6", "#F59E0B", "#14B8A6"] # Pink, Purple, Orange, Teal
    )
    fig_eval.update_layout(xaxis_title="Query Sequence Number", yaxis_title="Score (0.0 to 1.0)")
    st.plotly_chart(fig_eval, use_container_width=True)
    
    # 3. Export/Reset Section
    st.markdown("### 🛠️ Analytics Operations")
    col_a, col_b = st.columns([1, 4])
    if col_a.button("🗑️ Reset Stats", help="Clear all metrics history"):
        clear_all_metrics()
        st.success("Successfully cleared statistics.")
        st.rerun()

def render_history_table() -> None:
    """Renders searchable past query history with CSV/JSON exports."""
    st.markdown("## 📜 Query & Response History Log")
    analytics = get_analytics()
    history = analytics["search_history"]
    
    if not history:
        st.info("Query history is empty.")
        return
        
    df_history = pd.DataFrame([
        {
            "Timestamp": entry["timestamp"],
            "Question": entry["question"],
            "Answer": entry["answer"],
            "Retrieval (s)": f"{entry['retrieval_time']:.3f}",
            "Response (s)": f"{entry['response_time']:.2f}",
            "Precision": f"{entry.get('evaluation', {}).get('context_precision', 0.0)*100:.0f}%",
            "Recall": f"{entry.get('evaluation', {}).get('context_recall', 0.0)*100:.0f}%",
            "Faithfulness": f"{entry.get('evaluation', {}).get('faithfulness', 0.0)*100:.0f}%",
            "Relevance": f"{entry.get('evaluation', {}).get('answer_relevance', 0.0)*100:.0f}%"
        } for entry in history
    ])
    
    # Search box for history
    search = st.text_input("🔍 Search History Log", "")
    if search:
        df_history = df_history[
            df_history["Question"].str.contains(search, case=False) |
            df_history["Answer"].str.contains(search, case=False)
        ]
        
    st.dataframe(df_history, use_container_width=True)
    
    # Export options
    col1, col2, _ = st.columns([1, 1, 4])
    
    csv = df_history.to_csv(index=False).encode('utf-8')
    col1.download_button(
        label="📥 Export CSV",
        data=csv,
        file_name="rag_search_history.csv",
        mime="text/csv"
    )
    
    json_str = json.dumps(history, indent=4)
    col2.download_button(
        label="📥 Export JSON",
        data=json_str,
        file_name="rag_search_history.json",
        mime="application/json"
    )
