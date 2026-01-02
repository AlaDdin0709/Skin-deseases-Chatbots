"""
Prepare Medical Dataset
Script pour créer un dataset médical depuis les abstracts.
"""

import sys
from pathlib import Path
import json
from datasets import load_dataset
import logging
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.helpers import setup_logging, load_config

setup_logging("INFO")
logger = logging.getLogger(__name__)


def extract_qa_from_abstract(abstract: str, category: str) -> List[Dict[str, str]]:
    """
    Génère des paires question-réponse depuis un abstract.
    
    Args:
        abstract: Texte de l'abstract médical
        category: Catégorie (skin_cancer, benign_lesions, etc.)
        
    Returns:
        Liste de paires {"input": question, "output": answer}
    """
    qa_pairs = []
    
    # Question 1: Résumé général
    qa_pairs.append({
        "input": f"What does medical research say about {category.replace('_', ' ')}?",
        "output": abstract
    })
    
    # Question 2: Détails spécifiques (extraction de phrases clés)
    sentences = abstract.split('. ')
    if len(sentences) >= 2:
        # Utiliser les 2 premières phrases comme contexte
        key_info = '. '.join(sentences[:2])
        qa_pairs.append({
            "input": f"Explain key findings about {category.replace('_', ' ')}",
            "output": key_info
        })
    
    # Question 3: Question diagnostique
    if "melanoma" in abstract.lower():
        qa_pairs.append({
            "input": "How is melanoma diagnosed according to medical literature?",
            "output": abstract
        })
    elif "psoriasis" in abstract.lower():
        qa_pairs.append({
            "input": "What are the clinical features of psoriasis?",
            "output": abstract
        })
    elif "eczema" in abstract.lower() or "dermatitis" in abstract.lower():
        qa_pairs.append({
            "input": "Describe the characteristics and treatment of eczema/dermatitis",
            "output": abstract
        })
    
    return qa_pairs


def prepare_medical_dataset(
    output_path: str = "data/dermatology_qa.json",
    max_samples: int = 1000
):
    """
    Prépare un dataset médical pour entraînement LoRA.
    
    Args:
        output_path: Chemin de sortie JSON
        max_samples: Nombre max de paires QA
    """
    logger.info("="*80)
    logger.info("📊 Preparing Medical Dataset")
    logger.info("="*80)
    
    # 1. Charger config pour keywords
    config = load_config()
    keywords = config['rag']['keywords']
    
    # 2. Charger dataset medical abstracts
    logger.info("📥 Loading TimSchopf/medical_abstracts...")
    dataset = load_dataset("TimSchopf/medical_abstracts", split='train')
    logger.info(f"   Total abstracts: {len(dataset)}")
    
    # 3. Filtrer et convertir en QA pairs
    logger.info("🔍 Filtering by keywords and generating QA pairs...")
    
    qa_pairs = []
    category_counts = {}
    
    for item in dataset:
        abstract = item.get('abstract', '')
        if not abstract or len(abstract) < 100:
            continue
        
        abstract_lower = abstract.lower()
        
        # Classifier l'abstract
        matched_category = None
        for category, category_keywords in keywords.items():
            for keyword in category_keywords:
                if keyword.lower() in abstract_lower:
                    matched_category = category
                    break
            if matched_category:
                break
        
        if matched_category:
            # Générer QA pairs
            new_pairs = extract_qa_from_abstract(abstract, matched_category)
            qa_pairs.extend(new_pairs)
            
            # Compter
            category_counts[matched_category] = category_counts.get(matched_category, 0) + 1
            
            # Limite
            if len(qa_pairs) >= max_samples:
                break
    
    logger.info(f"   Generated QA pairs: {len(qa_pairs)}")
    logger.info("   Category distribution:")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"      {cat}: {count}")
    
    # 4. Sauvegarder
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Dataset saved to: {output_path}")
    logger.info(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # 5. Afficher exemples
    logger.info("\n📝 Sample QA pairs:")
    for i, pair in enumerate(qa_pairs[:3], 1):
        logger.info(f"\n   [{i}] Input: {pair['input'][:100]}...")
        logger.info(f"       Output: {pair['output'][:150]}...")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Dataset preparation complete!")
    logger.info("\n   Next steps:")
    logger.info("   1. Review dataset: cat data/dermatology_qa.json")
    logger.info("   2. Train adapter: python scripts/train_lora_adapter.py")
    logger.info("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare medical dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/dermatology_qa.json",
        help="Output path for dataset JSON"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of QA pairs"
    )
    
    args = parser.parse_args()
    
    prepare_medical_dataset(
        output_path=args.output,
        max_samples=args.max_samples
    )
