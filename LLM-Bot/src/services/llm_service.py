"""
LLM Service - Flan-T5-XL
Service pour génération de texte basée sur Flan-T5-XL avec quantisation 8-bit.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMService:
    """Service de génération LLM avec Flan-T5-XL."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le service LLM.
        
        Args:
            config: Configuration du modèle
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔧 LLMService initialized (device: {self.device})")
    
    def load_model(self) -> None:
        """Charge le modèle Flan-T5-XL avec quantisation 8-bit."""
        try:
            model_name = self.config.get('name', 'google/flan-t5-xl')
            logger.info(f"📥 Loading {model_name}...")
            
            # Configuration de quantisation
            quant_config = self.config.get('quantization', {})
            
            if quant_config.get('load_in_8bit', True):
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_enable_fp32_cpu_offload=False  # Forcer GPU-only (risque OOM)
                )
                logger.info("⚙️  Using 8-bit quantization (GPU-only, no CPU offload)")
            else:
                bnb_config = None
            
            # Charger tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Charger modèle
            max_memory = self.config.get('max_memory', {"0": "4GB", "cpu": "16GB"})
            
            # Forcer device_map sur GPU uniquement
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="cuda",
                max_memory={0: "4GiB"},
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "   CPU mode")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Génère une réponse basée sur le prompt.
        
        Args:
            prompt: Texte d'entrée
            max_new_tokens: Nombre maximum de tokens à générer
            temperature: Température de sampling
            top_p: Nucleus sampling
            do_sample: Activer le sampling
            
        Returns:
            Texte généré
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            raise
    
    def unload_model(self) -> None:
        """Libère la mémoire GPU."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🗑️  Model unloaded")
