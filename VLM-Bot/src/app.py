"""
Application Gradio - VLM-Bot
Interface web pour l'analyse dermatologique.
"""

import gradio as gr
from PIL import Image
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from services.vlm_service import VLMService
from services.rag_service import RAGService
from services.opencv_service import OpenCVService
from utils.helpers import (
    load_config,
    setup_logging,
    load_environment,
    ensure_directories,
    format_prompt
)

# Setup
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Charger configuration et environnement
try:
    load_environment()
    config = load_config()
    ensure_directories(config)
except Exception as e:
    logger.error(f"❌ Erreur de configuration: {e}")
    raise

# Initialiser les services globaux
vlm_service = None
rag_service = None
opencv_service = OpenCVService(config['opencv'])


def initialize_services():
    """Initialise les services VLM et RAG (lazy loading)."""
    global vlm_service, rag_service
    
    if vlm_service is None:
        logger.info("🔄 Initialisation du VLM...")
        vlm_service = VLMService(config['models']['vlm'])
        vlm_service.load_model()
    
    if rag_service is None:
        logger.info("🔄 Chargement de l'index RAG...")
        rag_service = RAGService(config['rag'])
        index_path = config['rag'].get('index_path', 'data/processed/faiss_index')
        
        if Path(index_path).exists():
            rag_service.load_index(index_path)
        else:
            logger.warning("⚠️  Index RAG non trouvé. Construction en cours...")
            rag_service.build_index(save_path=index_path)


def analyze_lesion_complete(
    image,
    use_opencv,
    opencv_data_manual,
    max_tokens,
    temperature,
    num_sources
):
    """
    Pipeline complet d'analyse.
    
    Args:
        image: Image PIL
        use_opencv: Utiliser OpenCV pour extraction
        opencv_data_manual: Données manuelles (si use_opencv=False)
        max_tokens: Max tokens à générer
        temperature: Température de sampling
        num_sources: Nombre de sources RAG
        
    Returns:
        Tuple (opencv_output, sources_text, diagnosis_text)
    """
    if image is None:
        return "⚠️ Veuillez télécharger une image d'abord!", "", ""
    
    try:
        # Initialiser les services (lazy)
        initialize_services()
        
        # Convertir image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        
        # Phase 1: Extraction OpenCV ou données manuelles
        if use_opencv:
            logger.info("🔬 Extraction OpenCV...")
            opencv_result = opencv_service.analyze_lesion(image)
            
            if opencv_result.get('error'):
                return opencv_result['description'], "", ""
            
            opencv_description = opencv_result['description']
            mode_text = "🔬 OpenCV Feature Extraction Used"
        else:
            if opencv_data_manual and opencv_data_manual.strip():
                opencv_description = opencv_data_manual
                mode_text = "📊 Manual Pre-computed Data Used"
            else:
                opencv_description = None
                mode_text = "👁️ Direct VLM Analysis (No Pre-computed Data)"
        
        # Phase 2: Recherche RAG
        logger.info("📚 Recherche RAG...")
        key_terms = [
            "melanoma", "atypical nevus", "dysplastic nevus",
            "asymmetry", "irregular borders", "multiple colors",
            "pigmented lesion", "ABCDE criteria", "basal cell carcinoma",
            "squamous cell carcinoma", "dermatoscopy", "skin cancer"
        ]
        query_text = " ".join(key_terms)
        rag_results = rag_service.search(query_text, top_k=int(num_sources))
        
        # Formater les sources
        sources_text = f"**{mode_text}**\n\n"
        sources_text += f"**Found {len(rag_results)} relevant medical abstracts:**\n\n"
        
        retrieved_context = ""
        for i, (doc, score) in enumerate(rag_results, 1):
            sources_text += f"**[Source {i}]** (Relevance: {score:.4f})\n"
            sources_text += f"{doc.page_content}\n"
            sources_text += f"{'-'*80}\n\n"
            retrieved_context += f"\n[Source {i}]:\n{doc.page_content}\n"
        
        # Phase 3: Construire le prompt
        if opencv_description:
            prompt = format_prompt(opencv_description, retrieved_context, "with_opencv")
        else:
            prompt = format_prompt("", retrieved_context, "direct")
        
        # Phase 4: Génération VLM
        logger.info("🤖 Génération du diagnostic...")
        diagnosis = vlm_service.generate_diagnosis(
            image=image,
            prompt=prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        
        # Préparer le bloc OpenCV séparément pour éviter d'avoir des backslashes
        if use_opencv and opencv_description:
            opencv_block = (
                "OPENCV FEATURE EXTRACTION:\n"
                + str(opencv_description)
                + "\n\n"
                + ("=" * 80)
                + "\n"
            )
        else:
            opencv_block = ""

        # Sauvegarder le rapport
        report = f"""
SKIN LESION ANALYSIS REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Mode: {mode_text}

{opencv_block}

EVIDENCE-BASED DIAGNOSIS:
{diagnosis}

RETRIEVED SOURCES:
{retrieved_context}

DISCLAIMER: For research and educational purposes only. NOT a substitute for
professional medical advice, diagnosis, or treatment. Consult a qualified dermatologist.
{'='*80}
"""

        filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ Analyse terminée. Rapport: {filename}")
        
        # Retourner les résultats
        opencv_output = opencv_description if use_opencv else ""
        return opencv_output, sources_text, diagnosis
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        import traceback
        error_msg = f"❌ Erreur: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, "", ""


# ============================================================================
# Interface Gradio
# ============================================================================

custom_css = """
    .scrollable-output textarea {
        max-height: 500px !important;
        overflow-y: auto !important;
    }
    .gradio-container {
        max-width: 1400px !important;
    }
    #opencv_output, #sources_output, #diagnosis_output {
        max-height: 500px;
        overflow-y: auto;
    }
"""

with gr.Blocks(
    title="VLM-Bot - Dermatological Analysis",
) as demo:
    
    gr.Markdown("""
    # 🔬 VLM-Bot - Système d'Analyse Dermatologique
    
    **OpenCV + Phi-3-Vision + RAG**
    
    - 🎨 **OpenCV**: Extraction quantitative automatique
    - 🤖 **VLM**: Phi-3-Vision-128k avec quantisation 4-bit
    - 📚 **RAG**: Diagnostic basé sur la littérature médicale
    
    ⚠️ **DISCLAIMER**: Usage éducatif uniquement. Consultez toujours un dermatologue.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Étape 1: Upload Image")
            image_input = gr.Image(type="pil", label="Image de la lésion")
            
            gr.Markdown("### 🔬 Étape 2: Mode d'Analyse")
            use_opencv = gr.Checkbox(
                value=True,
                label="✅ Utiliser OpenCV (Recommandé)",
                info="Extraction automatique de mesures"
            )
            
            gr.Markdown("### 📊 Étape 3: Données Manuelles (Optionnel)")
            opencv_data_manual = gr.Textbox(
                label="Mesures manuelles",
                placeholder="Seulement si OpenCV désactivé...",
                lines=8,
                info="Ignoré si OpenCV est activé"
            )
            
            gr.Markdown("### ⚙️ Étape 4: Paramètres")
            with gr.Accordion("Paramètres avancés", open=False):
                max_tokens = gr.Slider(
                    512, 2048, value=1024, step=128,
                    label="Tokens de génération"
                )
                temperature = gr.Slider(
                    0.1, 1.0, value=0.5, step=0.1,
                    label="Température"
                )
                num_sources = gr.Slider(
                    1, 10, value=5, step=1,
                    label="Nombre de sources médicales"
                )
            
            analyze_btn = gr.Button("🔬 Analyser", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Résultats")
            
            with gr.Tabs():
                with gr.Tab("🔬 Caractéristiques OpenCV"):
                    opencv_output = gr.Textbox(
                        label="Mesures quantitatives",
                        lines=15,
                        max_lines=30,
                        elem_id="opencv_output"
                    )
                
                with gr.Tab("📚 Sources Médicales"):
                    sources_output = gr.Textbox(
                        label="Littérature récupérée",
                        lines=12,
                        max_lines=25,
                        elem_id="sources_output"
                    )
                
                with gr.Tab("🏥 Diagnostic"):
                    diagnosis_output = gr.Textbox(
                        label="Diagnostic clinique avec citations",
                        lines=12,
                        max_lines=25,
                        elem_id="diagnosis_output"
                    )
    
    with gr.Row():
        gr.Markdown("""
        ---
        ### 🎯 Instructions:
        
        1. **Téléchargez** une image de lésion cutanée
        2. **Activez OpenCV** pour extraction automatique (ou fournissez données manuelles)
        3. **Ajustez** les paramètres si nécessaire
        4. **Cliquez** sur "Analyser"
        5. **Consultez** les résultats dans les 3 onglets
        6. Le rapport complet est sauvegardé automatiquement (analysis_YYYYMMDD_HHMMSS.txt)
        
        ### ⚡ Note: Premier lancement
        Le premier lancement prend ~1-2 minutes (chargement des modèles).
        """)
    
    # Connecter le bouton
    analyze_btn.click(
        fn=analyze_lesion_complete,
        inputs=[
            image_input,
            use_opencv,
            opencv_data_manual,
            max_tokens,
            temperature,
            num_sources
        ],
        outputs=[opencv_output, sources_output, diagnosis_output]
    )


# ============================================================================
# Lancement
# ============================================================================

if __name__ == "__main__":
    gradio_config = config.get('gradio', {})
    
    logger.info("="*80)
    logger.info("🚀 Lancement de VLM-Bot Gradio App")
    logger.info("="*80)
    logger.info(f"   Port: {gradio_config.get('port', 7860)}")
    logger.info(f"   Share: {gradio_config.get('share', False)}")
    logger.info("="*80)
    
    demo.launch(
        server_name=gradio_config.get('server_name', '0.0.0.0'),
        server_port=gradio_config.get('port', 7860),
        share=gradio_config.get('share', False),
        debug=True,
        show_error=True
    )
