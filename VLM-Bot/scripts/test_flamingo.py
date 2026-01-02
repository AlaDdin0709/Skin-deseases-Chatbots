"""
Test Med-Flamingo Access
Vérifie si le modèle Med-Flamingo est accessible et charge.
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import model_info, list_repo_files
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_flamingo_access():
    """Test l'accès au modèle Med-Flamingo."""
    
    model_id = "med-flamingo/med-flamingo"
    
    logger.info("="*80)
    logger.info(f"🔍 Test d'accès: {model_id}")
    logger.info("="*80)
    
    # 1. Vérifier si le modèle existe
    try:
        logger.info("\n📋 Récupération des infos du modèle...")
        info = model_info(model_id)
        logger.info(f"   ✅ Modèle trouvé!")
        logger.info(f"   Downloads: {info.downloads}")
        logger.info(f"   Likes: {info.likes}")
        logger.info(f"   Tags: {info.tags}")
        
        # Lister les fichiers
        logger.info("\n📁 Fichiers du repository:")
        files = list_repo_files(model_id)
        for f in files[:20]:  # Premiers 20 fichiers
            logger.info(f"   - {f}")
        if len(files) > 20:
            logger.info(f"   ... et {len(files)-20} autres fichiers")
            
    except Exception as e:
        logger.error(f"   ❌ Modèle introuvable: {e}")
        logger.info("\n💡 Le modèle peut nécessiter:")
        logger.info("   - Un token HuggingFace avec accès spécial")
        logger.info("   - Une demande d'accès au propriétaire")
        logger.info("   - Le modèle peut être privé ou restreint")
        return False
    
    # 2. Essayer de charger le tokenizer/processor
    try:
        logger.info("\n📥 Test de chargement du processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        logger.info("   ✅ Processor chargé!")
        
    except Exception as e:
        logger.warning(f"   ⚠️  Processor non disponible: {e}")
        logger.info("   Tentative avec AutoTokenizer...")
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            logger.info("   ✅ Tokenizer chargé!")
        except Exception as e2:
            logger.error(f"   ❌ Tokenizer aussi inaccessible: {e2}")
            return False
    
    # 3. Essayer de charger le modèle (config seulement)
    try:
        logger.info("\n⚙️  Test de chargement de la config...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id)
        logger.info(f"   ✅ Config chargée!")
        logger.info(f"   Architecture: {config.model_type}")
        logger.info(f"   Architectures: {getattr(config, 'architectures', 'N/A')}")
        
    except Exception as e:
        logger.error(f"   ❌ Config inaccessible: {e}")
        return False
    
    # 4. Test de chargement du modèle (petit test)
    try:
        logger.info("\n🧪 Test de chargement du modèle (config only)...")
        
        # Essayer AutoModelForVision2Seq (architecture Flamingo typique)
        logger.info("   Tentative avec AutoModelForVision2Seq...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True  # Important pour modèles custom
        )
        logger.info("   ✅ Modèle chargé avec AutoModelForVision2Seq!")
        
        # Infos sur le modèle
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Paramètres totaux: {total_params / 1e9:.2f}B")
        
        logger.info("\n" + "="*80)
        logger.info("✅ SUCCESS! Med-Flamingo est accessible et chargeable!")
        logger.info("="*80)
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Chargement échoué: {e}")
        logger.info("\n💡 Essayez:")
        logger.info("   1. Vérifier votre token HuggingFace dans .env")
        logger.info("   2. Demander accès au modèle sur HuggingFace")
        logger.info("   3. Utiliser trust_remote_code=True si modèle custom")
        
        logger.info("\n🔄 Alternative recommandée:")
        logger.info("   - microsoft/Phi-3-vision-128k-instruct (4.2B, publique)")
        logger.info("   - llava-hf/llava-1.5-7b-hf (standard)")
        
        return False

if __name__ == "__main__":
    success = test_flamingo_access()
    
    if not success:
        logger.info("\n" + "="*80)
        logger.info("❌ Med-Flamingo non accessible")
        logger.info("="*80)
        logger.info("\nOptions:")
        logger.info("1. Demander accès: https://huggingface.co/med-flamingo/med-flamingo")
        logger.info("2. Utiliser alternative: Phi-3-Vision ou Llava")
