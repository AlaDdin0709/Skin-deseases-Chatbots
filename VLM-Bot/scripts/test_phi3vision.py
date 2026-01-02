"""
Test Phi-3-Vision Access & Loading
Vérifie si Phi-3-Vision (4.2B) charge correctement sur RTX 3050 4GB.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_gpu_memory():
    """Affiche la mémoire GPU."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def test_phi3_vision():
    """Test le chargement de Phi-3-Vision."""
    
    model_id = "microsoft/Phi-3-vision-128k-instruct"
    
    logger.info("="*80)
    logger.info(f"🔍 Test: {model_id}")
    logger.info("="*80)
    
    # 1. Test processor
    try:
        logger.info("\n📥 Chargement du processor...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        logger.info("   ✅ Processor chargé!")
        
    except Exception as e:
        logger.error(f"   ❌ Processor failed: {e}")
        return False
    
    # 2. Test modèle avec 4-bit quantization
    try:
        logger.info("\n🧪 Chargement du modèle (4-bit quantization)...")
        logger.info("   Stratégie: device_map='auto', 4-bit NF4")
        
        from transformers import BitsAndBytesConfig, AutoConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Vérifier si flash_attn est installé (FlashAttention2)
        has_flash = False
        try:
            import flash_attn  # type: ignore
            has_flash = True
        except Exception:
            has_flash = False

        if not has_flash:
            logger.warning("⚠️ FlashAttention2 non installé - utilisation de l'implémentation eager (standard).")

        log_gpu_memory()

        # Charger la config et forcer attn_implementation
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if not has_flash:
            config._attn_implementation = "eager"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            max_memory={0: "3.5GiB", "cpu": "12GiB"},
            offload_folder="offload_phi3"
        )
        
        logger.info("   ✅ Modèle chargé!")
        log_gpu_memory()
        
        # Device map info
        logger.info(f"\n📍 Device Map:")
        if hasattr(model, 'hf_device_map'):
            for k, v in list(model.hf_device_map.items())[:10]:
                logger.info(f"   {k}: {v}")
            if len(model.hf_device_map) > 10:
                logger.info(f"   ... et {len(model.hf_device_map)-10} autres layers")
        
        # Test inference simple
        logger.info("\n🧪 Test d'inférence...")
        prompt = "<|user|>\n<|image_1|>\nDescribe this image.<|end|>\n<|assistant|>\n"
        
        # Créer image dummy (RGB 224x224)
        from PIL import Image
        import numpy as np
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        inputs = processor(prompt, [dummy_image], return_tensors="pt")
        
        # Move inputs to same device as model
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        logger.info("   Génération en cours...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                use_cache=False  # Désactiver cache pour compatibilité
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"   ✅ Output: {response[:100]}...")
        
        log_gpu_memory()
        
        logger.info("\n" + "="*80)
        logger.info("✅ SUCCESS! Phi-3-Vision fonctionne sur 4GB VRAM!")
        logger.info("="*80)
        
        # Cleanup
        del model, processor, inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Échec du chargement: {e}")
        logger.error(f"\nTraceback:", exc_info=True)
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        return False

if __name__ == "__main__":
    success = test_phi3_vision()
    
    if success:
        logger.info("\n✅ Phi-3-Vision est prêt pour VLM-Bot!")
        logger.info("   Lancez: python src/app.py")
    else:
        logger.info("\n❌ Test échoué")
        logger.info("   Options:")
        logger.info("   1. Vérifier CUDA/GPU setup")
        logger.info("   2. Augmenter max_memory GPU")
        logger.info("   3. Utiliser LLM-Bot (text-only) en fallback")
