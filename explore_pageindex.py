"""
Local exploration of PageIndex functionality
Loads API key from outside the repo for security
"""
import os
import sys
import json

# Add PageIndex to path
sys.path.insert(0, '/Users/xiongyuyu/Documents/projects/agent-eval/agent-evaluation/PageIndex')

# Load API key from outside the repo (parent directory)
from dotenv import load_dotenv
load_dotenv('/Users/xiongyuyu/Documents/projects/agent-eval/.env')

# Set the key that PageIndex expects
os.environ['CHATGPT_API_KEY'] = os.getenv('OPENAI_API_KEY')

print("=" * 60)
print("PageIndex Exploration")
print("=" * 60)
print(f"\nAPI Key loaded: {os.environ['CHATGPT_API_KEY'][:20]}...")
print(f"API Key length: {len(os.environ['CHATGPT_API_KEY'])} characters")

# Step 1: Import and explore PageIndex modules
print("\n" + "=" * 60)
print("Step 1: Importing PageIndex modules")
print("=" * 60)

try:
    from pageindex import page_index_main
    from pageindex.utils import ConfigLoader
    print("✓ Successfully imported PageIndex modules")
except Exception as e:
    print(f"✗ Import error: {e}")
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
