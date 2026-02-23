"""
PageIndex + LlamaIndex Integration
Clean implementation without monkey patching
"""
import os
import sys
import json
from typing import List

# Add PageIndex to path
sys.path.insert(0, '/Users/xiongyuyu/Documents/projects/agent-eval/agent-evaluation/PageIndex')

# Load API key
from dotenv import load_dotenv
load_dotenv('/Users/xiongyuyu/Documents/projects/agent-eval/.env')
os.environ['CHATGPT_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Imports
from pageindex import page_index_main
from pageindex.utils import ConfigLoader, ChatGPT_API, extract_json
import fitz

# LlamaIndex imports
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    from llama_index.llms.openai import OpenAI
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("LlamaIndex not installed. Run: pip install llama-index")


class PageIndexLlamaRetriever(BaseRetriever):
    """
    LlamaIndex retriever that uses PageIndex tree for reasoning-based retrieval.
    """
    
    def __init__(self, pageindex_structure: list, pdf_path: str, llm_model: str = 'gpt-4o-mini'):
        """
        Args:
            pageindex_structure: The tree structure from page_index_main()
            pdf_path: Path to the original PDF
            llm_model: LLM model for reasoning
        """
        self.structure = pageindex_structure
        self.pdf_path = pdf_path
        self.llm_model = llm_model
        self.doc = fitz.open(pdf_path)
        self.all_nodes = self._flatten_tree(pageindex_structure)
        super().__init__()
    
    def _flatten_tree(self, nodes):
        """Flatten tree to list of all nodes"""
        result = []
        for node in nodes:
            result.append(node)
            if 'nodes' in node and node['nodes']:
                result.extend(self._flatten_tree(node['nodes']))
        return result
    
    def _get_page_text(self, page_num: int) -> str:
        """Get text from a specific page (1-indexed)"""
        if 1 <= page_num <= len(self.doc):
            return self.doc[page_num - 1].get_text()
        return ""
    
    def _get_node_text(self, node: dict) -> str:
        """Get full text for a node (all pages in its range)"""
        start = node.get('start_index', 1)
        end = node.get('end_index', start)
        texts = []
        for page_num in range(start, end + 1):
            texts.append(self._get_page_text(page_num))
        return "\n".join(texts)
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes using PageIndex tree reasoning.
        This is the core method called by LlamaIndex.
        """
        query = query_bundle.query_str
        
        # Build tree context for reasoning
        tree_context = []
        for node in self.all_nodes:
            node_id = node.get('node_id', 'N/A')
            title = node.get('title', '')
            summary = node.get('summary', '')
            start = node.get('start_index', '')
            end = node.get('end_index', '')
            tree_context.append(f"[{node_id}] {title} (pages {start}-{end}): {summary}")
        
        # Use LLM to reason which nodes are relevant
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
    "reasoning": "Explain your thinking",
    "relevant_node_ids": ["0001", "0003"],
    "confidence": "high/medium/low"
}}

Return only the JSON."""
        
        response = ChatGPT_API(model=self.llm_model, prompt=prompt)
        result = extract_json(response)
        
        relevant_ids = result.get('relevant_node_ids', [])
        
        # Create LlamaIndex nodes from relevant PageIndex nodes
        retrieved_nodes = []
        for node in self.all_nodes:
            if node.get('node_id') in relevant_ids:
                node_text = self._get_node_text(node)
                
                # Create LlamaIndex TextNode
                llama_node = TextNode(
                    text=node_text,
                    metadata={
                        'node_id': node.get('node_id'),
                        'title': node.get('title'),
                        'page_range': f"{node.get('start_index')}-{node.get('end_index')}",
                        'source': 'pageindex'
                    }
                )
                
                # Score based on position in relevant_ids (higher = more relevant)
                try:
                    score = 1.0 - (relevant_ids.index(node.get('node_id')) * 0.1)
                except ValueError:
                    score = 0.5
                
                retrieved_nodes.append(NodeWithScore(node=llama_node, score=score))
        
        return retrieved_nodes
    
    def close(self):
        """Close the PDF document"""
        self.doc.close()


def main():
    """Main execution flow"""
    print("=" * 60)
    print("PageIndex + LlamaIndex Integration")
    print("=" * 60)
    
    pdf_path = "/Users/xiongyuyu/Documents/projects/agent-eval/agent-evaluation/pdf/JD - Vice President, AI Data Scientist (Singapore).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return
    
    # Step 1: Build PageIndex tree
    print("\nStep 1: Building PageIndex tree...")
    config_loader = ConfigLoader()
    opt = config_loader.load({
        'model': 'gpt-4o-mini',
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'yes',
    })
    
    result = page_index_main(pdf_path, opt)
    structure = result.get('structure', [])
    print(f"✓ Built tree with {len(structure)} top-level sections")
    
    # Step 2: Create LlamaIndex retriever
    print("\nStep 2: Creating LlamaIndex retriever...")
    retriever = PageIndexLlamaRetriever(structure, pdf_path)
    print(f"✓ Loaded {len(retriever.all_nodes)} nodes")
    
    # Step 3: Use with LlamaIndex (if available)
    if LLAMAINDEX_AVAILABLE:
        print("\nStep 3: Testing with LlamaIndex...")
        
        # Set up LlamaIndex with OpenAI
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create index with our custom retriever
        # Note: We create empty documents since our retriever handles everything
        dummy_doc = Document(text="PageIndex document")
        index = VectorStoreIndex.from_documents([dummy_doc])
        
        # Replace default retriever with our PageIndex retriever
        query_engine = index.as_query_engine(retriever=retriever)
        
        # Test query
        query = "What are the key responsibilities of this role?"
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        
        print(f"\nAnswer: {response}")
        print(f"\nSources: {response.source_nodes}")
    else:
        print("\nStep 3: Testing standalone retrieval...")
        query = "What are the key responsibilities of this role?"
        print(f"\nQuery: {query}")
        
        from llama_index.core.schema import QueryBundle
        nodes = retriever._retrieve(QueryBundle(query_str=query))
        
        print(f"\nRetrieved {len(nodes)} nodes:")
        for node_with_score in nodes:
            node = node_with_score.node
            print(f"  [{node.metadata['node_id']}] {node.metadata['title']} (score: {node_with_score.score:.2f})")
    
    retriever.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
