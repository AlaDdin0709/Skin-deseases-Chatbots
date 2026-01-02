"""
RAG Service - Retrieval Augmented Generation
Gère l'indexation FAISS et la recherche de contexte médical.
"""

import pandas as pd
import re
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGService:
    """Service pour le système RAG (indexation + recherche)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le service RAG.
        
        Args:
            config: Configuration RAG (dict depuis config.yaml)
        """
        self.config = config
        self.vectorstore: Optional[FAISS] = None
        self.embeddings = None
        
    def build_index(self, save_path: Optional[str] = None) -> None:
        """
        Construit l'index FAISS depuis le dataset médical.
        
        Args:
            save_path: Chemin pour sauvegarder l'index (optionnel)
        """
        logger.info("🗃️ Construction de l'index RAG...")
        
        # Charger le modèle d'embeddings
        embedding_model = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}  # Embeddings sur CPU
        )
        
        # Charger et filtrer le dataset
        dataset_name = self.config.get('dataset', 'TimSchopf/medical_abstracts')
        df = self._load_and_filter_dataset(dataset_name)
        
        # Créer les documents
        docs = [
            Document(
                page_content=row['medical_abstract'],
                metadata={'topic': row['topic']}
            )
            for _, row in df.iterrows()
        ]
        
        # Chunking
        chunk_size = self.config.get('chunk_size', 500)
        chunk_overlap = self.config.get('chunk_overlap', 100)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        
        logger.info(f"   📄 {len(chunks)} chunks créés depuis {len(df)} abstracts")
        
        # Construire l'index FAISS
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Sauvegarder l'index si demandé
        if save_path:
            self.save_index(save_path)
        
        logger.info("✅ Index RAG prêt!")
    
    def load_index(self, index_path: str) -> None:
        """
        Charge un index FAISS pré-construit.
        
        Args:
            index_path: Chemin vers l'index FAISS
        """
        logger.info(f"📥 Chargement de l'index depuis: {index_path}")
        
        # Charger le modèle d'embeddings
        embedding_model = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Charger l'index
        self.vectorstore = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info("✅ Index chargé!")
    
    def save_index(self, save_path: str) -> None:
        """
        Sauvegarde l'index FAISS.
        
        Args:
            save_path: Chemin de sauvegarde
        """
        if self.vectorstore is None:
            raise RuntimeError("Aucun index à sauvegarder. Construisez l'index d'abord.")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(save_path)
        logger.info(f"💾 Index sauvegardé: {save_path}")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Recherche les documents pertinents.
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats (défaut: config.top_k)
            
        Returns:
            Liste de tuples (Document, score)
        """
        if self.vectorstore is None:
            raise RuntimeError("Index non chargé. Appelez build_index() ou load_index().")
        
        k = top_k or self.config.get('top_k', 5)
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        return results
    
    def _load_and_filter_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Charge et filtre le dataset médical.
        
        Args:
            dataset_name: Nom du dataset HuggingFace
            
        Returns:
            DataFrame filtré
        """
        logger.info(f"📥 Chargement du dataset: {dataset_name}")
        
        # Charger le dataset
        dataset = load_dataset(dataset_name)
        df = pd.concat([
            pd.DataFrame(dataset['train']),
            pd.DataFrame(dataset['test'])
        ])
        
        # Appliquer la classification par mots-clés
        df['topic'] = df['medical_abstract'].apply(self._classify_abstract)
        df = df[df['topic'].notna()].copy()
        
        logger.info(f"   ✅ {len(df)} abstracts filtrés")
        return df
    
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
    
    def format_context(self, results: List[Tuple[Document, float]]) -> str:
        """
        Formate les résultats de recherche en contexte textuel.
        
        Args:
            results: Résultats de search()
            
        Returns:
            Contexte formaté pour le prompt
        """
        context_parts = []
        
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Source {i}] (Score: {score:.4f}):\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
