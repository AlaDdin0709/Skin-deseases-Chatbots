"""
Build FAISS Index
Script pour construire l'index FAISS depuis les abstracts médicaux.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag_service import RAGService
from src.utils.helpers import load_config, setup_logging

setup_logging("INFO")

if __name__ == "__main__":
    print("="*80)
    print("🔨 Building FAISS Index for LLM-Bot")
    print("="*80)
    
    # Charger config
    config = load_config()
    rag_config = config['rag']
    
    # Créer service
    rag_service = RAGService(rag_config)
    
    # Build index
    index_path = rag_config.get('index_path', 'data/processed/faiss_index')
    rag_service.build_index(save_path=index_path)
    
    print("\n" + "="*80)
    print("✅ Index built successfully!")
    print(f"   Location: {index_path}")
    print("="*80)
    
    # Test search
    print("\n🧪 Running test search...")
    test_query = "melanoma skin cancer diagnosis"
    results = rag_service.search(test_query, top_k=3)
    
    print(f"\n   Query: '{test_query}'")
    print(f"   Results: {len(results)}\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"   [{i}] Score: {score:.4f}")
        print(f"       Preview: {doc.page_content[:150]}...")
        print()
    
    print("="*80)
    print("🎉 Done! You can now run: python src/app.py")
    print("="*80)
