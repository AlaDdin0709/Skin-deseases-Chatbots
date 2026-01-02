"""
Utilities - Helper Functions
Fonctions utilitaires pour configuration, logging, prompts.
"""

import yaml
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Charge la configuration depuis YAML.
    
    Args:
        config_path: Chemin vers config.yaml
        
    Returns:
        Dictionnaire de configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")


def setup_logging(level: str = "INFO") -> None:
    """
    Configure le logging.
    
    Args:
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_environment() -> None:
    """Charge les variables d'environnement depuis .env"""
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logging.warning("⚠️  .env file not found. Using .env.example as template.")


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Crée les répertoires nécessaires.
    
    Args:
        config: Configuration
    """
    directories = [
        'data/raw',
        'data/processed',
        config.get('rag', {}).get('index_path', 'data/processed/faiss_index')
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def format_prompt(
    question: str,
    retrieved_context: str,
    mode: str = "default"
) -> str:
    """
    Formate le prompt pour le LLM.
    
    Args:
        question: Question de l'utilisateur
        retrieved_context: Contexte récupéré par RAG
        mode: Mode de prompt
        
    Returns:
        Prompt formaté
    """
    if mode == "with_context":
        prompt = f"""You are an expert medical assistant specializing in dermatology. 
Based on the following medical literature and the user's question, provide a detailed and evidence-based response.

MEDICAL LITERATURE:
{retrieved_context}

USER QUESTION:
{question}

Provide a comprehensive answer citing specific evidence from the literature above. Use medical terminology when appropriate but ensure the explanation is clear."""

    elif mode == "direct":
        prompt = f"""You are an expert medical assistant specializing in dermatology.

USER QUESTION:
{question}

Provide a detailed and evidence-based response using your medical knowledge."""

    else:
        # Default mode
        prompt = f"""Based on the following context, answer the question:

CONTEXT:
{retrieved_context}

QUESTION:
{question}

ANSWER:"""
    
    return prompt


def format_medical_prompt(
    symptoms: str,
    duration: str,
    retrieved_context: str
) -> str:
    """
    Formate un prompt médical structuré.
    
    Args:
        symptoms: Description des symptômes
        duration: Durée des symptômes
        retrieved_context: Contexte médical récupéré
        
    Returns:
        Prompt formaté
    """
    prompt = f"""You are an expert dermatologist. Based on the medical literature and patient information below, provide a preliminary assessment.

MEDICAL LITERATURE:
{retrieved_context}

PATIENT INFORMATION:
- Symptoms: {symptoms}
- Duration: {duration}

Please provide:
1. Possible differential diagnoses (with evidence from literature)
2. Recommended investigations
3. General management approach

IMPORTANT: This is for educational purposes only. Always consult a qualified healthcare professional for medical advice.

ASSESSMENT:"""
    
    return prompt
