"""
Local exploration of PageIndex functionality
Loads API key from outside the repo for security
"""
import os
import sys
import json
import time
import asyncio

# Add PageIndex to path
sys.path.insert(0, '/Users/xiongyuyu/Documents/projects/agent-eval/agent-evaluation/PageIndex')

# Load API key from outside the repo (parent directory)
from dotenv import load_dotenv
load_dotenv('/Users/xiongyuyu/Documents/projects/agent-eval/.env')

# Set the key that PageIndex expects
os.environ['CHATGPT_API_KEY'] = os.getenv('OPENAI_API_KEY')

# ============ CUSTOM LLM IMPLEMENTATION ============
# Option 1: Monkey-patch to use your own LLM
# Replace these functions with your custom LLM calls

def custom_llm_call(model, prompt, temperature=0):
    """
    Custom LLM implementation - replace with your own
    Examples: local LLM (Ollama, vLLM), Azure OpenAI, Anthropic, etc.
    """
    # Example using OpenAI-compatible API (default)
    import openai
    client = openai.OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        # base_url="http://localhost:11434/v1"  # For Ollama
        # base_url="https://your-custom-endpoint.com/v1"  # For custom endpoint
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

async def custom_llm_call_async(model, prompt, temperature=0):
    """Async version of custom LLM call"""
    import openai
    async with openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY')) as client:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

# Monkey-patch PageIndex to use custom LLM
import pageindex.utils as utils

def custom_chatgpt_api(model, prompt, api_key=None, chat_history=None):
    """Wrapper that calls custom LLM instead of OpenAI"""
    max_retries = 3
    for i in range(max_retries):
        try:
            return custom_llm_call(model, prompt, temperature=0)
        except Exception as e:
            print(f'************* Retrying ({i+1}/{max_retries}) *************')
            if i < max_retries - 1:
                time.sleep(1)
            else:
                return "Error"

async def custom_chatgpt_api_async(model, prompt, api_key=None):
    """Async wrapper for custom LLM"""
    max_retries = 3
    for i in range(max_retries):
        try:
            return await custom_llm_call_async(model, prompt, temperature=0)
        except Exception as e:
            print(f'************* Retrying ({i+1}/{max_retries}) *************')
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                return "Error"

# Apply monkey patches
def custom_chatgpt_api_with_finish_reason(model, prompt, api_key=None, chat_history=None):
    """Wrapper that calls custom LLM and returns (content, finish_reason)"""
    max_retries = 3
    for i in range(max_retries):
        try:
            result = custom_llm_call(model, prompt, temperature=0)
            return result, "finished"
        except Exception as e:
            print(f'************* Retrying ({i+1}/{max_retries}) *************')
            if i < max_retries - 1:
                time.sleep(1)
            else:
                return "Error", "error"

utils.ChatGPT_API = custom_chatgpt_api
utils.ChatGPT_API_async = custom_chatgpt_api_async
utils.ChatGPT_API_with_finish_reason = custom_chatgpt_api_with_finish_reason

print("=" * 60)
print("PageIndex Exploration")
print("=" * 60)
print(f"\nAPI Key loaded: {os.getenv('OPENAI_API_KEY')[:20]}...")
print(f"API Key length: {len(os.getenv('OPENAI_API_KEY'))} characters")
print("✓ Custom LLM implementation patched")

# Step 1: Import and explore PageIndex modules
print("\n" + "=" * 60)
print("Step 1: Importing PageIndex modules")
print("=" * 60)

try:
    from pageindex import page_index_main
    from pageindex.utils import ConfigLoader, extract_json
    # Note: ChatGPT_API is already monkey-patched above, don't reimport
    print("✓ Successfully imported PageIndex modules")
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Check PDF file
print("\n" + "=" * 60)
print("Step 2: Checking PDF file")
print("=" * 60)

pdf_path = "/Users/xiongyuyu/Documents/projects/agent-eval/agent-evaluation/pdf/JD - Vice President, AI Data Scientist (Singapore).pdf"

if os.path.exists(pdf_path):
    print(f"✓ PDF found: {pdf_path}")
    print(f"  File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
else:
    print(f"✗ PDF not found: {pdf_path}")
    sys.exit(1)

# Step 3: Explore PDF structure with PyMuPDF
print("\n" + "=" * 60)
print("Step 3: Exploring PDF structure (PyMuPDF)")
print("=" * 60)

try:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    print(f"✓ PDF opened successfully")
    print(f"  Total pages: {len(doc)}")
    
    # Show first few pages content
    for i in range(min(3, len(doc))):
        page = doc[i]
        text = page.get_text()
        print(f"\n--- Page {i+1} (first 300 chars) ---")
        print(text[:300].replace('\n', ' '))
    doc.close()
except Exception as e:
    print(f"✗ Error reading PDF: {e}")

# Step 4: Build PageIndex tree structure
print("\n" + "=" * 60)
print("Step 4: Building PageIndex tree structure")
print("=" * 60)

# Configure options using ConfigLoader
config_loader = ConfigLoader()
user_opt = {
    'model': 'gpt-4o-mini',
    'toc_check_page_num': 20,
    'max_page_num_each_node': 10,
    'max_token_num_each_node': 20000,
    'if_add_node_id': 'yes',
    'if_add_node_summary': 'yes',
    'if_add_doc_description': 'no',
    'if_add_node_text': 'no'
}
opt = config_loader.load(user_opt)

print(f"Configuration:")
print(f"  - model: {opt.model}")
print(f"  - toc_check_page_num: {opt.toc_check_page_num}")
print(f"  - max_page_num_each_node: {opt.max_page_num_each_node}")
print(f"  - max_token_num_each_node: {opt.max_token_num_each_node}")
print(f"  - if_add_node_id: {opt.if_add_node_id}")
print(f"  - if_add_node_summary: {opt.if_add_node_summary}")

print("\nRunning page_index_main()...")
print("(This will call OpenAI API and may take 30-60 seconds)")

try:
    result = page_index_main(pdf_path, opt)
    print("\n✓ PageIndex tree built successfully!")
    
    # Display results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Document name: {result.get('doc_name', 'N/A')}")
    
    structure = result.get('structure', [])
    print(f"Total top-level sections: {len(structure)}")
    
    # Print tree structure
    def print_tree(nodes, indent=0):
        for node in nodes:
            node_id = node.get('node_id', 'N/A')
            title = node.get('title', 'N/A')
            start = node.get('start_index', 'N/A')
            end = node.get('end_index', 'N/A')
            summary = node.get('summary', '')
            print(f"{'  ' * indent}[{node_id}] {title} (pages {start}-{end})")
            if summary and indent < 2:
                print(f"{'  ' * (indent+1)}Summary: {summary[:100]}...")
            if 'nodes' in node and node['nodes']:
                print_tree(node['nodes'], indent + 1)
    
    print("\nTree Structure:")
    print_tree(structure)
    
    # Save to file
    output_dir = '/Users/xiongyuyu/Documents/projects/agent-eval/agent-evaluation/results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'jd_structure.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
except Exception as e:
    print(f"\n✗ Error building PageIndex: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Retrieval - Tree Search
print("\n" + "=" * 60)
print("Step 5: Retrieval - Tree Search")
print("=" * 60)

class PageIndexRetriever:
    """Simple retriever that uses the PageIndex tree for reasoning-based retrieval"""
    
    def __init__(self, structure, pdf_path):
        self.structure = structure
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
    def get_page_text(self, page_num):
        """Get text from a specific page (1-indexed)"""
        if 1 <= page_num <= len(self.doc):
            return self.doc[page_num - 1].get_text()
        return ""
    
    def get_node_text(self, node):
        """Get full text for a node (all pages in its range)"""
        start = node.get('start_index', 1)
        end = node.get('end_index', start)
        texts = []
        for page_num in range(start, end + 1):
            texts.append(self.get_page_text(page_num))
        return "\n".join(texts)
    
    def flatten_tree(self, nodes=None):
        """Flatten tree to list of all nodes"""
        if nodes is None:
            nodes = self.structure
        result = []
        for node in nodes:
            result.append(node)
            if 'nodes' in node and node['nodes']:
                result.extend(self.flatten_tree(node['nodes']))
        return result
    
    def tree_search(self, query, model='gpt-4o-mini'):
        """
        Reasoning-based retrieval using the tree structure.
        Simulates how humans navigate a document - start from top-level, 
        reason which section is most relevant, then drill down.
        """
        
        all_nodes = self.flatten_tree()
        
        # Build tree context for the LLM
        tree_context = []
        for node in all_nodes:
            node_id = node.get('node_id', 'N/A')
            title = node.get('title', '')
            summary = node.get('summary', '')
            start = node.get('start_index', '')
            end = node.get('end_index', '')
            tree_context.append(f"[{node_id}] {title} (pages {start}-{end}): {summary}")
        
        # Step 1: Reason which nodes are relevant
        prompt = f"""You are given a document's table of contents and a user query.
Your task is to identify which sections (by node_id) are most relevant to answer the query.

Document Structure:
{chr(10).join(tree_context)}

User Query: {query}

Instructions:
1. Analyze which sections would contain information relevant to the query
2. Consider both the section titles and summaries
3. Return the node_ids of the most relevant sections (up to 3)
4. If no sections seem relevant, return an empty list

Response format (JSON):
{{
    "reasoning": "Explain your thinking about which sections are relevant",
    "relevant_node_ids": ["0001", "0003", ...],
    "confidence": "high/medium/low"
}}

Return only the JSON, no other text."""
        
        response = utils.ChatGPT_API(model=model, prompt=prompt)
        result = extract_json(response)
        
        relevant_ids = result.get('relevant_node_ids', [])
        reasoning = result.get('reasoning', '')
        
        print(f"\nTree Search Reasoning: {reasoning[:200]}...")
        print(f"Relevant nodes: {relevant_ids}")
        
        # Step 2: Retrieve text from relevant nodes
        retrieved_contexts = []
        for node in all_nodes:
            if node.get('node_id') in relevant_ids:
                node_text = self.get_node_text(node)
                retrieved_contexts.append({
                    'node_id': node.get('node_id'),
                    'title': node.get('title'),
                    'pages': f"{node.get('start_index')}-{node.get('end_index')}",
                    'text': node_text[:3000]  # Limit text length
                })
        
        return retrieved_contexts, reasoning

# Initialize retriever
print("\nInitializing PageIndex Retriever...")
retriever = PageIndexRetriever(structure, pdf_path)
print(f"✓ Loaded {len(retriever.flatten_tree())} nodes from tree")

# Test queries
test_queries = [
    "What are the main responsibilities of this role?",
    "What qualifications are required?",
    "What technical skills are mentioned?"
]

for query in test_queries:
    print("\n" + "-" * 60)
    print(f"Query: {query}")
    print("-" * 60)
    
    contexts, reasoning = retriever.tree_search(query)
    
    print(f"\nRetrieved {len(contexts)} relevant sections:")
    for ctx in contexts:
        print(f"  [{ctx['node_id']}] {ctx['title']} (pages {ctx['pages']})")

# Step 6: Generation - RAG with PageIndex
print("\n" + "=" * 60)
print("Step 6: Generation - RAG with PageIndex")
print("=" * 60)

def generate_answer(query, retriever, model='gpt-4o-mini'):
    """
    Full RAG pipeline: retrieve relevant sections, then generate answer
    """
    
    # Retrieve
    contexts, reasoning = retriever.tree_search(query, model)
    
    if not contexts:
        return "No relevant information found in the document."
    
    # Build context string
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        context_parts.append(f"""Source {i}: [{ctx['node_id']}] {ctx['title']} (pages {ctx['pages']})
{ctx['text']}
""")
    
    full_context = "\n---\n".join(context_parts)
    
    # Generate answer
    prompt = f"""You are a helpful assistant answering questions about a job description.
Use the provided context to answer the user's question accurately.
If the context doesn't contain enough information, say so.

Context:
{full_context}

User Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Cite the source sections (e.g., "According to [0001]...")
3. Be concise but complete
4. If information is missing, acknowledge it

Answer:"""
    
    answer = utils.ChatGPT_API(model=model, prompt=prompt)
    return answer, contexts

# Test generation with one query
print("\nTesting full RAG pipeline...")
test_query = "What are the key responsibilities of the Vice President, AI Data Scientist role?"
print(f"\nQuery: {test_query}")

answer, sources = generate_answer(test_query, retriever)

print("\n" + "-" * 60)
print("Generated Answer:")
print("-" * 60)
print(answer)

print("\n" + "-" * 60)
print("Sources Used:")
print("-" * 60)
for src in sources:
    print(f"  [{src['node_id']}] {src['title']} - pages {src['pages']}")

# Cleanup
retriever.doc.close()

print("\n" + "=" * 60)
print("PageIndex RAG Exploration Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("1. PageIndex builds a hierarchical tree structure from PDFs")
print("2. Retrieval uses LLM reasoning to navigate the tree (not vectors)")
print("3. Generation uses retrieved sections as context for RAG")
print("4. All page references are traceable and explainable")
