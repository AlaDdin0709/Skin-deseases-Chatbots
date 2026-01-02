"""Utilitaires partagés pour le projet VLM-Bot."""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging
from dotenv import load_dotenv
import os


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Charge la configuration depuis config.yaml.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Configuration sous forme de dictionnaire
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(level: str = "INFO") -> None:
    """
    Configure le système de logging.
    
    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_environment() -> None:
    """Charge les variables d'environnement depuis .env."""
    load_dotenv()
    
    # Vérifier les variables requises
    required_vars = ['HF_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Variables d'environnement manquantes: {', '.join(missing_vars)}\n"
            f"Copiez .env.example vers .env et remplissez les valeurs."
        )


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Crée les répertoires nécessaires s'ils n'existent pas.
    
    Args:
        config: Configuration du projet
    """
    dirs_to_create = [
        'data/raw',
        'data/processed',
        'models',
        'logs'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def format_prompt(
    opencv_description: str,
    rag_context: str,
    prompt_type: str = "with_opencv"
) -> str:
    """
    Construit le prompt pour le VLM.
    
    Args:
        opencv_description: Description textuelle depuis OpenCV
        rag_context: Contexte médical depuis RAG
        prompt_type: Type de prompt ('with_opencv' ou 'direct')
        
    Returns:
        Prompt formaté
    """
    if prompt_type == "with_opencv":
        return f"""
{opencv_description}

================================================================================
RELEVANT MEDICAL LITERATURE:
================================================================================
{rag_context}

================================================================================
EVIDENCE-BASED ANALYSIS INSTRUCTIONS:
================================================================================

Based on the quantitative measurements provided above AND the medical literature,
provide a comprehensive structured diagnosis.

**MANDATORY CITATION RULE**: Cite sources using [Source 1], [Source 2], etc.

Structure your response with these EXACT sections:

1. DIFFERENTIAL DIAGNOSIS (Ranked by Likelihood):
   - AT LEAST 5 diagnoses with likelihood levels (Most Likely / Likely / Possible / Less Likely)
   - Reference the specific measurements (asymmetry score, color zones, size, irregularity score)
   - Cite literature for each diagnosis

2. CONCERNING FEATURES WITH EVIDENCE:
   - List specific quantitative features from the measurements
   - Explain clinical significance with citations

3. COMPARISON TO LITERATURE PATTERNS:
   - How these measurements compare to literature patterns
   - Statistical context if available from sources

4. CLINICAL RECOMMENDATIONS:
   - Urgency level (Immediate/Urgent/Routine) with justification from sources
   - Specific next steps (biopsy type, excision, monitoring)
   - Follow-up timeline

5. PATIENT COMMUNICATION GUIDANCE:
   - Clear, compassionate explanation of findings
   - What to expect in next steps

Remember: Cite [Source X] for every clinical claim.
"""
    else:
        return f"""
COMPREHENSIVE DERMATOLOGICAL LESION ANALYSIS:

Analyze this skin lesion image in detail using:
- Morphological features
- Color characteristics
- Border quality
- Symmetry assessment
- ABCDE criteria

================================================================================
RELEVANT MEDICAL LITERATURE:
================================================================================
{rag_context}

================================================================================
EVIDENCE-BASED ANALYSIS INSTRUCTIONS:
================================================================================

Based on your visual analysis AND the medical literature, provide a comprehensive
structured diagnosis with citations to sources.

Remember: Cite [Source X] for every clinical claim.
"""
