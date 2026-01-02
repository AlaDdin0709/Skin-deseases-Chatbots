"""
LLM Service - Flan-T5-XL
Service pour génération de texte basée sur Flan-T5-XL avec quantisation 8-bit.
Support optionnel pour adaptateurs médicaux LoRA/PEFT.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMService:
    """Service de génération LLM avec Flan-T5-XL + adaptateur médical optionnel."""
    
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
        self.use_adapter = config.get('use_medical_adapter', False)
        self.adapter_path = config.get('adapter_path', None)
        logger.info(f"🔧 LLMService initialized (device: {self.device})")
        if self.use_adapter:
            logger.info(f"🧬 Medical adapter enabled: {self.adapter_path}")
    
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
                )
                logger.info("⚙️  Using 8-bit quantization")
            else:
                bnb_config = None
            
            # Charger tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Charger modèle
            max_memory = self.config.get('max_memory', {"0": "4GB", "cpu": "16GB"})
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=self.config.get('device_map', 'auto'),
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Charger adaptateur médical si disponible
            if self.use_adapter and self.adapter_path:
                self._load_medical_adapter()
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "   CPU mode")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def _load_medical_adapter(self) -> None:
        """Charge l'adaptateur médical LoRA/PEFT."""
        try:
            from peft import PeftModel
            
            adapter_path = Path(self.adapter_path)
            if not adapter_path.exists():
                logger.warning(f"⚠️  Adapter path not found: {adapter_path}")
                logger.warning("   Skipping adapter loading. Model will run without specialization.")
                return
            
            logger.info(f"🧬 Loading medical adapter from: {adapter_path}")
            
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path)
            )
            
            logger.info("✅ Medical adapter loaded successfully")
            logger.info("   Model is now specialized for dermatology!")
            
        except ImportError:
            logger.error("❌ PEFT library not installed. Install with: pip install peft")
            logger.warning("   Running without medical adapter...")
        except Exception as e:
            logger.error(f"❌ Error loading adapter: {e}")
            logger.warning("   Running without medical adapter...")
    
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
