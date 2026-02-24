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


class PageIndexDocument:
    """Holds PageIndex tree and PDF for a single document"""
    def __init__(self, doc_id: str, structure: list, pdf_path: str):
        self.doc_id = doc_id
        self.structure = structure
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.all_nodes = self._flatten_tree(structure)
    
    def _flatten_tree(self, nodes):
        result = []
        for node in nodes:
            result.append(node)
            if 'nodes' in node and node['nodes']:
                result.extend(self._flatten_tree(node['nodes']))
        return result
    
    def close(self):
        self.doc.close()


class PageIndexLlamaRetriever(BaseRetriever):
    """
    LlamaIndex retriever that uses PageIndex tree for reasoning-based retrieval.
    Supports multiple documents.
    """
    
    def __init__(self, documents: List[PageIndexDocument], llm_model: str = 'gpt-4o-mini'):
        """
        Args:
            documents: List of PageIndexDocument objects
            llm_model: LLM model for reasoning
        """
        self.documents = documents
        self.llm_model = llm_model
        super().__init__()
    
    def _get_all_nodes(self):
        """Get all nodes from all documents with doc_id prefix"""
        all_nodes = []
        for doc in self.documents:
            for node in doc.all_nodes:
                # Add doc_id to node_id for uniqueness
                node_copy = node.copy()
                node_copy['doc_id'] = doc.doc_id
                node_copy['unique_id'] = f"{doc.doc_id}:{node.get('node_id', 'N/A')}"
                all_nodes.append((doc, node_copy))
        return all_nodes
    
    def _get_node_text(self, doc: PageIndexDocument, node: dict) -> str:
        """Get full text for a node from a specific document"""
        start = node.get('start_index', 1)
        end = node.get('end_index', start)
        texts = []
        for page_num in range(start, end + 1):
            if 1 <= page_num <= len(doc.doc):
                texts.append(doc.doc[page_num - 1].get_text())
        return "\n".join(texts)
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes using PageIndex tree reasoning across all documents.
        """
        query = query_bundle.query_str
        all_nodes = self._get_all_nodes()
        
        # Build tree context for all documents
        tree_context = []
        for doc, node in all_nodes:
            unique_id = node.get('unique_id')
            title = node.get('title', '')
            summary = node.get('summary', '')
            start = node.get('start_index', '')
            end = node.get('end_index', '')
            tree_context.append(f"[{unique_id}] {title} (pages {start}-{end}): {summary}")
        
        # Use LLM to reason which nodes are relevant
        prompt = f"""You are given multiple document tables of contents and a user query.
        Your task is to identify which sections (by unique_id) are most relevant to answer the query.

        Documents Structure:
        {chr(10).join(tree_context)}

        User Query: {query}

        Instructions:
        1. Analyze which sections from which documents are relevant
        2. Consider both section titles and summaries
        3. Return the unique_ids of most relevant sections (up to 5 across all docs)
        4. Format: "doc_id:node_id" (e.g., "doc1:0001", "doc2:0003")

        Response format (JSON):
        {{
            "reasoning": "Explain your thinking",
            "relevant_unique_ids": ["doc1:0001", "doc2:0003"],
            "confidence": "high/medium/low"
        }}

        Return only the JSON."""
        
        response = ChatGPT_API(model=self.llm_model, prompt=prompt)
        result = extract_json(response)
        
        relevant_ids = result.get('relevant_unique_ids', [])
        
        # Create LlamaIndex nodes from relevant PageIndex nodes
        retrieved_nodes = []
        for doc, node in all_nodes:
            if node.get('unique_id') in relevant_ids:
                node_text = self._get_node_text(doc, node)
                
                llama_node = TextNode(
                    text=node_text,
                    metadata={
                        'doc_id': node.get('doc_id'),
                        'node_id': node.get('node_id'),
                        'title': node.get('title'),
                        'page_range': f"{node.get('start_index')}-{node.get('end_index')}",
                        'source': 'pageindex'
                    }
                )
                
                try:
                    score = 1.0 - (relevant_ids.index(node.get('unique_id')) * 0.1)
                except ValueError:
                    score = 0.5
                
                retrieved_nodes.append(NodeWithScore(node=llama_node, score=score))
        
        return retrieved_nodes
    
    def close(self):
        """Close all PDF documents"""
        for doc in self.documents:
            doc.close()


def main():
    """Main execution flow with multiple PDFs"""
    print("=" * 60)
    print("PageIndex + LlamaIndex (Multi-Document)")
    print("=" * 60)
    
    # Define PDFs to process
    pdf_files = [
        ("jd1", "/Users/xiongyuyu/Documents/projects/agent-eval/agent-evaluation/pdf/JD - Vice President, AI Data Scientist (Singapore).pdf"),
        # Add more PDFs here:
        # ("doc2", "/path/to/another.pdf"),
        # ("doc3", "/path/to/third.pdf"),
    ]
    
    # Filter existing files
    pdf_files = [(doc_id, path) for doc_id, path in pdf_files if os.path.exists(path)]
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    print(f"\nProcessing {len(pdf_files)} PDF(s)...")
    
    # Step 1: Build PageIndex trees for all documents
    print("\nStep 1: Building PageIndex trees...")
    config_loader = ConfigLoader()
    opt = config_loader.load({
        'model': 'gpt-4o-mini',
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'yes',
    })
    
    documents = []
    for doc_id, pdf_path in pdf_files:
        print(f"  Processing {doc_id}...")
        result = page_index_main(pdf_path, opt)
        structure = result.get('structure', [])
        
        doc = PageIndexDocument(doc_id, structure, pdf_path)
        documents.append(doc)
        print(f"    ✓ {len(structure)} top-level sections, {len(doc.all_nodes)} total nodes")
    
    # Step 2: Create multi-document retriever
    print("\nStep 2: Creating multi-document retriever...")
    retriever = PageIndexLlamaRetriever(documents)
    total_nodes = sum(len(d.all_nodes) for d in documents)
    print(f"✓ Loaded {total_nodes} nodes from {len(documents)} documents")
    
    # Step 3: Query across all documents with custom answer generation
    print("\nStep 3: Testing retrieval and answer generation...")
    
    if LLAMAINDEX_AVAILABLE:
        from llama_index.core.schema import QueryBundle
        
        # Set up LlamaIndex LLM
        llm = OpenAI(model="gpt-4o-mini", temperature=0)
        
        def generate_answer(query: str, context_str: str, llm) -> str:
            """Custom answer generation with query and context string"""
            prompt = f"""You are a helpful assistant answering questions based on provided document context.

Context:
{context_str}

User Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Be concise but complete
3. If information is missing, say "I don't have enough information to answer this"
4. Cite document sources when possible (e.g., "According to [doc_id:node_id]...")

Answer:"""
            
            response = llm.complete(prompt)
            return response.text
        
        queries = [
            "What are the key responsibilities?",
            "What qualifications are required?",
            "What technical skills are mentioned?"
        ]
        
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            # Step 1: Retrieve relevant nodes
            print("\n[Retrieval]")
            nodes = retriever._retrieve(QueryBundle(query_str=query))
            print(f"Retrieved {len(nodes)} nodes:")
            for node_with_score in nodes:
                node = node_with_score.node
                doc_id = node.metadata['doc_id']
                title = node.metadata['title']
                print(f"  [{doc_id}:{node.metadata['node_id']}] {title} (score: {node_with_score.score:.2f})")
            
            # Step 2: Build context string from retrieved nodes
            print("\n[Context Building]")
            context_parts = []
            for i, node_with_score in enumerate(nodes, 1):
                node = node_with_score.node
                meta = node.metadata
                context_parts.append(f"""Source {i}: [{meta['doc_id']}:{meta['node_id']}] {meta['title']} (pages {meta['page_range']})
{node.text[:2000]}
""")
            
            context_str = "\n---\n".join(context_parts)
            print(f"Context length: {len(context_str)} characters")
            
            # Step 3: Generate answer with custom prompt
            print("\n[Answer Generation]")
            answer = generate_answer(query, context_str, llm)
            
            print(f"Answer: {answer}")
            print(f"\nSources used: {len(nodes)}")
            for i, node_with_score in enumerate(nodes, 1):
                meta = node_with_score.node.metadata
                print(f"  {i}. [{meta['doc_id']}:{meta['node_id']}] {meta['title']}")
    else:
        print("LlamaIndex not available. Install with: pip install llama-index")
    
    retriever.close()
    print("\n" + "="*60)
    print("✓ Done!")


if __name__ == "__main__":
    main()
