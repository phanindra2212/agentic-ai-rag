import time
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END

# Import nodes
from agents.query_agent import query_understanding_node, AgentState
from agents.retrieval_agent import retrieval_agent_node
from agents.context_agent import context_optimization_node
from agents.response_agent import response_generation_node
from agents.validation_agent import validation_agent_node
from utils.logger import logger
from utils.metrics import record_query
from rag.evaluator import evaluate_rag

# Compile LangGraph Workflow
def create_rag_graph():
    # 1. Initialize StateGraph with the AgentState definition
    workflow = StateGraph(AgentState)
    
    # 2. Add all agent nodes
    workflow.add_node("query_understanding", query_understanding_node)
    workflow.add_node("retrieval", retrieval_agent_node)
    workflow.add_node("context_optimization", context_optimization_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("validation", validation_agent_node)
    
    # 3. Define linear execution path
    workflow.set_entry_point("query_understanding")
    workflow.add_edge("query_understanding", "retrieval")
    workflow.add_edge("retrieval", "context_optimization")
    workflow.add_edge("context_optimization", "response_generation")
    workflow.add_edge("response_generation", "validation")
    
    # 4. Define conditional router edge from Validation Agent
    def route_validation_output(state: AgentState):
        if state.get("is_hallucinated", False):
            logger.info("Validation Agent flagged hallucination. Routing back to Response Generation.")
            return "response_generation"
        logger.info("Validation Agent approved response. Routing to END.")
        return END
        
    workflow.add_conditional_edges(
        "validation",
        route_validation_output,
        {
            "response_generation": "response_generation",
            END: END
        }
    )
    
    # 5. Compile the graph
    return workflow.compile()

# Instantiate the graph
rag_workflow_graph = create_rag_graph()

def execute_agentic_rag(
    question: str,
    chat_history: List[Dict[str, str]],
    filters: Dict[str, Any]
) -> Dict[str, Any]:
    """Executes the complete compiled LangGraph workflow for a user question.
    
    Records metrics and runs post-generation evaluations.
    """
    logger.info(f"Initiating Agentic RAG execution for: '{question}'")
    start_time = time.time()
    
    # Initial state configuration
    initial_state: AgentState = {
        "question": question,
        "current_query": question,
        "complexity": "simple",
        "generated_queries": [],
        "retrieved_documents": [],
        "optimized_context": [],
        "response": "",
        "citations": [],
        "confidence_score": "Low Confidence",
        "confidence_val": 0.0,
        "is_hallucinated": False,
        "validation_feedback": "",
        "chat_history": chat_history,
        "filters": filters,
        "retries": 0
    }
    
    # Run the graph
    # Measure retrieval vs response times
    # Note: retrieval completes after context_optimization node, and response completes at the end
    retrieval_end_time = start_time
    
    state = initial_state
    try:
        # Step through the graph to measure individual phase times
        for output in rag_workflow_graph.stream(initial_state):
            # Capture state updates
            for node_name, state_update in output.items():
                state.update(state_update)
                if node_name == "context_optimization":
                    retrieval_end_time = time.time()
    except Exception as e:
        logger.error(f"Error executing LangGraph RAG: {e}", exc_info=True)
        state["response"] = f"An execution error occurred in the agentic workflow: {e}"
        
    end_time = time.time()
    
    # Calculations
    retrieval_time = retrieval_end_time - start_time
    response_time = end_time - retrieval_end_time
    total_time = end_time - start_time
    
    logger.info(f"Agentic workflow execution completed in {total_time:.4f}s.")
    
    # Run Evaluation Module
    retrieved_texts = [doc.page_content for doc in state.get("optimized_context", [])]
    eval_results = {}
    if retrieved_texts and state.get("response"):
        eval_results = evaluate_rag(
            question=question,
            retrieved_chunks=retrieved_texts,
            generated_answer=state["response"]
        )
        
    # Record to session history/metrics state
    record_query(
        question=question,
        answer=state["response"],
        retrieval_time=retrieval_time,
        response_time=response_time,
        evaluation=eval_results,
        confidence_val=state.get("confidence_val", 0.0),
        citations=state.get("citations", []),
        context_chunks=retrieved_texts
    )
    
    # Return aggregated result details for UI consumption
    return {
        "question": question,
        "answer": state.get("response", ""),
        "citations": state.get("citations", []),
        "confidence_score": state.get("confidence_score", "Low Confidence"),
        "confidence_val": state.get("confidence_val", 0.0),
        "complexity": state.get("complexity", "simple"),
        "generated_queries": state.get("generated_queries", []),
        "retrieval_time": retrieval_time,
        "response_time": response_time,
        "total_time": total_time,
        "eval_metrics": eval_results,
        "chunks_count": len(state.get("retrieved_documents", []))
    }
