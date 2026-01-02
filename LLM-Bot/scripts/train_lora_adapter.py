"""
Train Medical LoRA Adapter
Script pour entraîner un adaptateur LoRA spécialisé en dermatologie.
"""

import sys
from pathlib import Path
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import logging

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.helpers import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def prepare_dataset(tokenizer, dataset_path: str, max_length: int = 512):
    """
    Prépare le dataset pour l'entraînement.
    
    Args:
        tokenizer: Tokenizer T5
        dataset_path: Chemin vers fichier JSON
        max_length: Longueur max des séquences
        
    Returns:
        Dataset tokenizé
    """
    # Charger dataset
    dataset = load_dataset("json", data_files=dataset_path)
    
    def preprocess_function(examples):
        # Format: "input" -> "output"
        inputs = examples["input"]
        targets = examples["output"]
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize targets
        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset


def train_lora_adapter(
    model_name: str = "google/flan-t5-xl",
    dataset_path: str = "data/dermatology_qa.json",
    output_dir: str = "data/models/lora-dermatology",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 3e-4,
    lora_r: int = 8,
    lora_alpha: int = 32
):
    """
    Entraîne un adaptateur LoRA médical.
    
    Args:
        model_name: Nom du modèle de base
        dataset_path: Chemin vers dataset d'entraînement
        output_dir: Dossier de sortie
        epochs: Nombre d'époques
        batch_size: Taille batch par device
        learning_rate: Learning rate
        lora_r: Rang LoRA
        lora_alpha: Alpha LoRA
    """
    logger.info("="*80)
    logger.info("🧬 Training Medical LoRA Adapter")
    logger.info("="*80)
    
    # 1. Charger tokenizer
    logger.info(f"📥 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Charger modèle avec quantisation 8-bit
    logger.info(f"📥 Loading base model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 3. Configuration LoRA
    logger.info("⚙️  Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q", "v"],  # Attention query/value matrices
        bias="none",
        inference_mode=False
    )
    
    # 4. Créer modèle PEFT
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 5. Préparer dataset
    logger.info(f"📊 Loading dataset: {dataset_path}")
    
    if not Path(dataset_path).exists():
        logger.error(f"❌ Dataset not found: {dataset_path}")
        logger.info("💡 Create a JSON file with format:")
        logger.info('   [{"input": "question", "output": "answer"}, ...]')
        return
    
    tokenized_dataset = prepare_dataset(tokenizer, dataset_path)
    train_dataset = tokenized_dataset["train"]
    
    logger.info(f"   Training samples: {len(train_dataset)}")
    
    # 6. Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )
    
    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none"
    )
    
    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # 9. Entraînement
    logger.info("🚀 Starting training...")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   LoRA rank: {lora_r}")
    
    trainer.train()
    
    # 10. Sauvegarder adaptateur
    logger.info(f"💾 Saving adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("="*80)
    logger.info("✅ Training complete!")
    logger.info(f"   Adapter saved to: {output_dir}")
    logger.info(f"   Adapter size: ~10-50MB")
    logger.info("="*80)
    
    # 11. Test inference
    logger.info("\n🧪 Testing adapter...")
    test_prompt = "What are the ABCDE criteria for melanoma diagnosis?"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logger.info(f"   Prompt: {test_prompt}")
    logger.info(f"   Response: {response}")
    
    logger.info("\n🎉 Next steps:")
    logger.info("   1. Update config.yaml:")
    logger.info(f"      use_medical_adapter: true")
    logger.info(f"      adapter_path: {output_dir}")
    logger.info("   2. Restart LLM-Bot: python src/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical LoRA Adapter")
    
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-xl",
        help="Base model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dermatology_qa.json",
        help="Path to training dataset (JSON)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/lora-dermatology",
        help="Output directory for adapter"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    
    args = parser.parse_args()
    
    train_lora_adapter(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
