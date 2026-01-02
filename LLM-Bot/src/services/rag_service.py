"""
RAG Service - Retrieval-Augmented Generation
Service pour indexation et recherche dans les abstracts médicaux.

Cette implémentation suit la logique utilisée par VLM-Bot :
- charge le dataset complet (train/test),
- filtre par mots-clés définis dans la config (champ `medical_abstract`),
- fallback sur l'ensemble des abstracts si aucun match,
- split en chunks, calcul d'embeddings et construction de l'index FAISS.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import re

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset

logger = logging.getLogger(__name__)


class RAGService:
    """Service de recherche RAG avec FAISS."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le service RAG.
        
        Args:
            config: Configuration RAG
        """
        self.config = config
        self.vectorstore = None
        self.embeddings = None
        logger.info("🔧 RAGService initialized")
    
    def build_index(self, save_path: Optional[str] = None) -> None:
        """
        Construit l'index FAISS depuis les abstracts médicaux.
        
        Args:
            save_path: Chemin de sauvegarde de l'index
        """
        try:
            logger.info("📥 Loading medical abstracts dataset...")
            dataset_name = self.config.get('dataset', 'TimSchopf/medical_abstracts')
            # Charger train + test si présent
            dataset = load_dataset(dataset_name)

            # Concat train/test si disponibles
            records = []
            if 'train' in dataset:
                records.extend(list(dataset['train']))
            if 'test' in dataset:
                records.extend(list(dataset['test']))
            if 'validation' in dataset:
                records.extend(list(dataset['validation']))

            logger.info(f"   Total abstracts: {len(records)}")

            # Filtrer par mots-clés
            logger.info("🔍 Filtering by keywords...")
            filtered_docs = []

            for item in records:
                # Utiliser le champ 'medical_abstract' si présent (format TimSchopf)
                abstract = ''
                if isinstance(item, dict):
                    abstract = item.get('medical_abstract') or item.get('abstract') or ''
                else:
                    # objet datasets avec attributs
                    abstract = getattr(item, 'medical_abstract', None) or getattr(item, 'abstract', '')

                category = self._classify_abstract(abstract)

                if category and abstract and abstract.strip():
                    doc = Document(
                        page_content=abstract,
                        metadata={
                            'category': category,
                            'source': dataset_name
                        }
                    )
                    filtered_docs.append(doc)

            logger.info(f"   Filtered abstracts: {len(filtered_docs)}")

            # Si aucun abstract ne correspond, utiliser tous les abstracts médicaux
            if len(filtered_docs) == 0:
                logger.warning("⚠️  No abstracts matched dermatology keywords!")
                logger.info("📚 Using all medical abstracts instead (general medical knowledge)...")
                filtered_docs = []
                for item in records:
                    abstract = ''
                    if isinstance(item, dict):
                        abstract = item.get('medical_abstract') or item.get('abstract') or ''
                    else:
                        abstract = getattr(item, 'medical_abstract', None) or getattr(item, 'abstract', '')

                    if abstract and abstract.strip():
                        filtered_docs.append(Document(page_content=abstract, metadata={'category': 'general_medical', 'source': dataset_name}))

                logger.info(f"   Total medical abstracts: {len(filtered_docs)}")

            # Chunking
            logger.info("✂️  Splitting documents...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get('chunk_size', 500),
                chunk_overlap=self.config.get('chunk_overlap', 100),
                length_function=len
            )
            chunks = text_splitter.split_documents(filtered_docs)
            logger.info(f"   Total chunks: {len(chunks)}")

            if len(chunks) == 0:
                raise RuntimeError("No document chunks were created. Check dataset fields and keywords in config.")

            # Créer embeddings
            logger.info("🧮 Creating embeddings...")
            embedding_model = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'}
            )

            # Créer index FAISS
            logger.info("📊 Building FAISS index...")
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

            # Sauvegarder
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                self.vectorstore.save_local(save_path)
                logger.info(f"💾 Index saved to: {save_path}")

            logger.info("✅ Index built successfully")

            # Test search
            try:
                test_results = self.search("melanoma diagnosis", top_k=1)
                logger.info(f"🧪 Test search returned {len(test_results)} results")
            except Exception:
                logger.warning("🧪 Test search failed (index may be empty)")

        except Exception as e:
            logger.error(f"❌ Error building index: {e}")
            raise

    def load_index(self, index_path: str) -> None:
        """
        Charge un index FAISS existant.
        
        Args:
            index_path: Chemin vers l'index
        """
        try:
            if not Path(index_path).exists():
                raise FileNotFoundError(f"Index not found: {index_path}")

            logger.info(f"📥 Loading index from: {index_path}")

            # Charger embeddings
            embedding_model = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'}
            )

            # Charger index
            self.vectorstore = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            logger.info("✅ Index loaded successfully")

        except Exception as e:
            logger.error(f"❌ Error loading index: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Recherche dans l'index FAISS.
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de (Document, score)
        """
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore not loaded. Call build_index() or load_index() first.")

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            return results

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            raise

    def _classify_abstract(self, text: str) -> Optional[str]:
        """
        Classifie un abstract selon les mots-clés configurés.
        
        Args:
            text: Texte de l'abstract
            
        Returns:
            Nom de la catégorie la plus pertinente ou None
        """
        if not isinstance(text, str):
            return None

        text_lower = text.lower()

        # Récupérer toutes les catégories de keywords depuis config
        all_keywords = self.config.get('keywords', {})

        # Compter les matches par catégorie
        category_scores = {}

        for category, keywords in all_keywords.items():
            score = 0
            for keyword in keywords:
                # Recherche de mots-clés (word boundary pour mots simples)
                if ' ' not in keyword:
                    # Mot simple - utiliser word boundary
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                        score += 1
                else:
                    # Phrase - recherche directe
                    if keyword in text_lower:
                        score += 1

            if score > 0:
                category_scores[category] = score

        # Retourner la catégorie avec le plus de matches
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            return best_category

        return None
