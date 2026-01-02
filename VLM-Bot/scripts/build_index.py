"""
Script pour construire l'index FAISS RAG.
Usage: python scripts/build_index.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag_service import RAGService
from src.utils.helpers import load_config, setup_logging, load_environment
import logging

logger = logging.getLogger(__name__)


def main():
    """Construit et sauvegarde l'index FAISS."""
    # Setup
    setup_logging("INFO")
    load_environment()
    config = load_config()
    
    logger.info("="*80)
    logger.info("🏗️  Construction de l'index RAG FAISS")
    logger.info("="*80)
    
    # Initialiser le service RAG
    rag_config = config['rag']
    rag_service = RAGService(rag_config)
    
    # Construire l'index
    index_path = rag_config.get('index_path', 'data/processed/faiss_index')
    rag_service.build_index(save_path=index_path)
    
    logger.info("="*80)
    logger.info(f"✅ Index sauvegardé: {index_path}")
    logger.info("="*80)
    
    # Test de recherche
    logger.info("\n🔍 Test de recherche...")
    test_query = "melanoma diagnostic criteria ABCDE asymmetry"
    results = rag_service.search(test_query, top_k=3)
    
    logger.info(f"\nRésultats pour: '{test_query}'")
    for i, (doc, score) in enumerate(results, 1):
        logger.info(f"\n[{i}] Score: {score:.4f}")
        logger.info(f"    {doc.page_content[:200]}...")
    
    logger.info("\n✅ Construction terminée avec succès!")


if __name__ == "__main__":
    main()
