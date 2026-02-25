"""
Hierarchical Agent Evaluation Framework - Minimal Agent Implementation
Based on technical_design.md using LangChain and LangGraph
"""

import os
import re
import json
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
load_dotenv()

# Import system prompt and database utilities
from system_prompt import main_prompt
from init_db import (
    get_vectorstore, get_db_connection, get_database_schema,
    DB_PATH, CHROMA_PATH, init_all_databases
)


# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    """State for the agent workflow"""
    messages: Annotated[Sequence[BaseMessage], add]
    original_query: str
    sub_queries: Dict[str, List[str]]  # tool_name -> list of sub-queries
    tool_outputs: Dict[str, Any]
    final_answer: str


# ============================================================================
# LLM Setup
# ============================================================================

def get_llm():
    """Initialize LLM - uses OpenAI by default, can be swapped for local models"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


# ============================================================================
# Node Functions
# ============================================================================

def decompose_and_route(state: AgentState) -> AgentState:
    """
    Decomposition & Routing Node
    - Analyzes user query
    - Decomposes into sub-queries if needed
    - Routes to appropriate tools
    """
    llm = get_llm()
    query = state["original_query"]
    
    # Create messages directly without template parsing
    messages = [
        ("system", main_prompt),
        ("human", query)
    ]
    
    # Get routing decision from LLM
    response = llm.invoke(messages).content
    
    # Parse the structured output
    parsed = parse_router_output(response)
    
    # Update state with parsed results
    state["messages"] = state["messages"] + [
        AIMessage(content=f"Routing Decision:\n{response}")
    ]
    state["sub_queries"] = parsed.get("sub_queries", {})
    
    return state


def parse_router_output(output: str) -> Dict[str, Any]:
    """
    Parse the router's structured output into a dictionary
    Expected format from main_prompt:
    INTENT: ...
    TOOLS: [...]
    CONFIDENCE: {...}
    SUB_QUERIES: {...}
    REASON: ...
    """
    result = {
        "intent": "",
        "tools": [],
        "confidence": {},
        "sub_queries": {},
        "reason": ""
    }
    
    # Extract INTENT
    intent_match = re.search(r'INTENT:\s*(.+?)(?=\n\w+:|$)', output, re.DOTALL)
    if intent_match:
        result["intent"] = intent_match.group(1).strip()
    
    # Extract TOOLS
    tools_match = re.search(r'TOOLS:\s*(\[.*?\])', output, re.DOTALL)
    if tools_match:
        tools_str = tools_match.group(1)
        # Parse JSON-like array
        result["tools"] = [t.strip().strip('"\'') for t in tools_str.strip('[]').split(',') if t.strip()]
    
    # Extract SUB_QUERIES
    sub_queries_match = re.search(r'SUB_QUERIES:\s*(\{.*?\})(?=\n\w+:|$)', output, re.DOTALL)
    if sub_queries_match:
        sub_queries_str = sub_queries_match.group(1)
        # Parse the sub-queries dictionary
        result["sub_queries"] = parse_sub_queries(sub_queries_str)
    
    # Extract REASON
    reason_match = re.search(r'REASON:\s*(.+?)(?=\n\w+:|$)', output, re.DOTALL)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()
    
    return result


def parse_sub_queries(sub_queries_str: str) -> Dict[str, List[str]]:
    """Parse the SUB_QUERIES dictionary from the router output"""
    sub_queries = {}
    
    # Simple regex-based parsing for the nested structure
    # Format: {"tool_name": ["query1", "query2"], ...}
    tool_pattern = r'"(\w+)":\s*\[(.*?)\]'
    matches = re.findall(tool_pattern, sub_queries_str, re.DOTALL)
    
    for tool_name, queries_str in matches:
        # Extract individual queries
        queries = re.findall(r'"([^"]+)"', queries_str)
        sub_queries[tool_name] = queries
    
    return sub_queries


def execute_rag_tool(state: AgentState) -> AgentState:
    """
    RAG Tool Execution Node
    - Executes RAG retrieval for sub-queries routed to rag_tool
    - Uses ChromaDB for document retrieval
    """
    sub_queries = state.get("sub_queries", {})
    rag_queries = sub_queries.get("rag_tool", [])
    
    outputs = []
    for query in rag_queries:
        output = execute_rag_query(query)
        outputs.append({"query": query, "result": output})
    
    state["tool_outputs"] = state.get("tool_outputs", {})
    state["tool_outputs"]["rag_tool"] = outputs
    
    return state


def execute_nl2sql_tool(state: AgentState) -> AgentState:
    """
    NL2SQL Tool Execution Node
    - Executes SQL generation and execution for sub-queries
    - Uses LLM to generate SQL from natural language
    - Executes against SQLite database
    """
    sub_queries = state.get("sub_queries", {})
    sql_queries = sub_queries.get("nl2sql_tool", [])
    
    outputs = []
    for query in sql_queries:
        output = execute_nl2sql_query(query)
        outputs.append({"query": query, "result": output})
    
    state["tool_outputs"] = state.get("tool_outputs", {})
    state["tool_outputs"]["nl2sql_tool"] = outputs
    
    return state


def synthesize_answer(state: AgentState) -> AgentState:
    """
    Synthesis Node
    - Combines tool outputs into final answer
    - Ensures faithfulness to tool outputs
    """
    llm = get_llm()
    original_query = state["original_query"]
    tool_outputs = state.get("tool_outputs", {})
    
    # Build synthesis prompt
    synthesis_prompt = f"""
You are a helpful assistant that synthesizes information from multiple sources into a coherent answer.

Original User Query: {original_query}

Tool Outputs:
{format_tool_outputs(tool_outputs)}

Instructions:
1. Synthesize the information from all tool outputs into a clear, comprehensive answer
2. Do NOT introduce information not present in the tool outputs
3. If the tools returned no relevant information, state that clearly
4. Be concise but complete

Provide your final answer:"""
    
    messages = [
        HumanMessage(content=synthesis_prompt)
    ]
    
    response = llm.invoke(messages)
    state["final_answer"] = response.content
    state["messages"] = state["messages"] + [
        AIMessage(content=f"Final Answer:\n{response.content}")
    ]
    
    return state


# ============================================================================
# Tool Implementations
# ============================================================================

def execute_rag_query(query: str) -> str:
    """
    Execute RAG query using ChromaDB
    - Retrieves relevant documents
    - Returns summarized context
    """
    try:
        vectorstore = get_vectorstore()
        
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant documents found in the knowledge base."
        
        # Format results
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content[:500]  # Truncate long content
            results.append(f"[{i}] Source: {source}\nContent: {content}...")
        
        return "\n\n".join(results)
    
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


def execute_nl2sql_query(query: str) -> str:
    """
    Execute NL2SQL query using LLM to generate SQL
    - Generates SQL from natural language
    - Executes against SQLite database
    - Returns results
    """
    import sqlite3
    
    try:
        # Get schema information
        schema = get_database_schema()
        
        # Generate SQL using LLM
        llm = get_llm()
        
        sql_prompt = f"""You are an expert SQL generator for banking data.

Database Schema:
{schema}

Convert the following natural language query into a valid SQLite SQL query.
Only return the SQL query, no explanation.

User Query: {query}

SQL Query:"""
        
        sql_response = llm.invoke([HumanMessage(content=sql_prompt)])
        sql_query = sql_response.content.strip()
        
        # Clean up SQL (remove markdown code blocks if present)
        sql_query = re.sub(r'```sql\s*', '', sql_query)
        sql_query = re.sub(r'```\s*', '', sql_query)
        sql_query = sql_query.strip()
        
        # Execute SQL
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            
            if not rows:
                result = "Query executed successfully. No results found."
            else:
                # Format results as JSON-like string
                columns = [description[0] for description in cursor.description]
                results = []
                for row in rows[:10]:  # Limit to 10 rows
                    results.append(dict(zip(columns, row)))
                
                result = f"SQL: {sql_query}\n\nResults ({len(rows)} rows):\n{json.dumps(results, indent=2, default=str)}"
            
            conn.close()
            return result
            
        except sqlite3.Error as e:
            conn.close()
            return f"SQL Error: {str(e)}\nGenerated SQL: {sql_query}"
    
    except Exception as e:
        return f"Error executing query: {str(e)}"


def format_tool_outputs(tool_outputs: Dict[str, Any]) -> str:
    """Format tool outputs for synthesis prompt"""
    formatted = []
    
    for tool_name, outputs in tool_outputs.items():
        formatted.append(f"\n=== {tool_name.upper()} ===")
        for output in outputs:
            formatted.append(f"Query: {output['query']}")
            formatted.append(f"Result: {output['result']}")
    
    return "\n".join(formatted)


# ============================================================================
# Routing Logic
# ============================================================================

def route_after_decomposition(state: AgentState) -> List[str]:
    """
    Determine which tool nodes to execute based on routing decision
    Returns list of node names to execute
    """
    sub_queries = state.get("sub_queries", {})
    targets = []
    
    if "rag_tool" in sub_queries and sub_queries["rag_tool"]:
        targets.append("rag_tool")
    if "nl2sql_tool" in sub_queries and sub_queries["nl2sql_tool"]:
        targets.append("nl2sql_tool")
    
    return targets if targets else ["synthesize"]


def should_continue_to_synthesize(state: AgentState) -> str:
    """Determine if we should proceed to synthesis"""
    # In a parallel execution model, we'd check if all tools completed
    # For simplicity, we always proceed to synthesize after tools
    return "synthesize"


# ============================================================================
# Graph Construction
# ============================================================================

def build_agent_graph() -> StateGraph:
    """Build the LangGraph workflow for the agent"""
    
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("decompose_route", decompose_and_route)
    workflow.add_node("rag_tool", execute_rag_tool)
    workflow.add_node("nl2sql_tool", execute_nl2sql_tool)
    workflow.add_node("synthesize", synthesize_answer)
    
    # Define edges
    workflow.set_entry_point("decompose_route")
    
    # Conditional routing from decomposition
    workflow.add_conditional_edges(
        "decompose_route",
        lambda state: route_after_decomposition(state),
        {
            "rag_tool": "rag_tool",
            "nl2sql_tool": "nl2sql_tool",
            "synthesize": "synthesize"
        }
    )
    
    # From tools to synthesis
    workflow.add_edge("rag_tool", "synthesize")
    workflow.add_edge("nl2sql_tool", "synthesize")
    
    # End at synthesis
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# ============================================================================
# Main Entry Point
# ============================================================================

def run_agent(query: str) -> Dict[str, Any]:
    """
    Main entry point to run the agent on a user query
    
    Args:
        query: The user's natural language query
        
    Returns:
        Dictionary containing the final answer and intermediate results
    """
    # Initialize state
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        original_query=query,
        sub_queries={},
        tool_outputs={},
        final_answer=""
    )
    
    # Build and run graph
    graph = build_agent_graph()
    final_state = graph.invoke(initial_state)
    
    return {
        "original_query": final_state["original_query"],
        "sub_queries": final_state["sub_queries"],
        "tool_outputs": final_state["tool_outputs"],
        "final_answer": final_state["final_answer"],
        "message_history": [m.content for m in final_state["messages"]]
    }


# ============================================================================
# CLI Interface
# ============================================================================

def init_databases():
    """Initialize both ChromaDB and SQLite databases"""
    print("Initializing databases...")
    init_all_databases()
    print("Database initialization complete!\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the hierarchical agent")
    parser.add_argument("--query", "-q", type=str, help="User query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--init", action="store_true", help="Initialize/reset databases")
    
    args = parser.parse_args()
    
    # Initialize databases on first run or if --init flag
    if args.init or not os.path.exists(DB_PATH) or not os.path.exists(CHROMA_PATH):
        init_databases()
    
    if args.interactive:
        print("=" * 60)
        print("Hierarchical Agent - Interactive Mode")
        print("Type 'exit' or 'quit' to stop")
        print("=" * 60)
        
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue
                
            result = run_agent(query)
            
            print("\n--- Routing Decision ---")
            for tool, subqs in result["sub_queries"].items():
                print(f"  {tool}: {subqs}")
            
            print("\n--- Tool Outputs ---")
            for tool, outputs in result["tool_outputs"].items():
                print(f"  {tool}:")
                for out in outputs:
                    print(f"    - {out['result'][:500]}...")  # Truncate long outputs
            
            print("\n--- Final Answer ---")
            print(result["final_answer"])
    
    elif args.query:
        result = run_agent(args.query)
        
        print("=" * 60)
        print("Query:", result["original_query"])
        print("=" * 60)
        
        print("\n--- Sub-queries ---")
        for tool, subqs in result["sub_queries"].items():
            print(f"  {tool}: {subqs}")
        
        print("\n--- Tool Outputs ---")
        for tool, outputs in result["tool_outputs"].items():
            print(f"  {tool}:")
            for out in outputs:
                print(f"    - {out['result'][:500]}...")  # Truncate long outputs
        
        print("\n--- Final Answer ---")
        print(result["final_answer"])
    
    else:
        # Demo with sample queries
        demo_queries = [
            "What is ABC Corp's current loan exposure?",
            "What are XYZ Inc's ESG commitments and revenue YTD?",
            "How many deals does DEF Ltd have in the pipeline?",
            "What was discussed in the last meeting with GHI Group?"
        ]
        
        for query in demo_queries:
            print("=" * 60)
            print("Query:", query)
            print("=" * 60)
            
            result = run_agent(query)
            
            print("\n--- Sub-queries ---")
            for tool, subqs in result["sub_queries"].items():
                print(f"  {tool}: {subqs}")
            
            print("\n--- Tool Outputs ---")
            for tool, outputs in result["tool_outputs"].items():
                print(f"  {tool}:")
                for out in outputs:
                    print(f"    - {out['result'][:500]}...")  # Truncate long outputs
            
            print("\n--- Final Answer ---")
            print(result["final_answer"])
            print("\n")
