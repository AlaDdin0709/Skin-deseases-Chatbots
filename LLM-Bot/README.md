# 🤖 LLM-Bot - Système de Q&A Dermatologique

Système de questions/réponses dermatologiques basé sur **Google Flan-T5-XL** et **RAG** (Retrieval-Augmented Generation).

## 🎯 Fonctionnalités

- **LLM**: Google Flan-T5-XL avec quantisation 8-bit (compatible 4GB VRAM)
- **RAG**: Recherche dans la littérature médicale (TimSchopf/medical_abstracts)
- **Interface Gradio**: 2 modes (Analyse de symptômes + Q&A général)
- **Architecture modulaire**: Services LLM, RAG, utilitaires

## 📋 Prérequis

- **GPU**: NVIDIA RTX 3050 Laptop (4GB VRAM) ou supérieur
- **CUDA**: 12.1+ (Driver 591.59+)
- **Python**: 3.10+
- **Conda**: Recommandé pour gestion d'environnement
- **HuggingFace Token**: Requis pour téléchargement des modèles

## 🚀 Installation Rapide

### 1. Créer l'environnement Conda

```powershell
# Utiliser l'environnement 'rag' existant (partagé avec VLM-Bot)
conda activate rag

# OU créer un nouvel environnement
conda create -n rag python=3.10 -y
conda activate rag
```

### 2. Installer PyTorch avec CUDA 12.1

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Installer les dépendances

```powershell
cd E:\Chatbots\LLM-Bot
pip install -r requirements.txt
```

### 4. Configurer l'environnement

```powershell
# Copier le template
copy .env.example .env

# Éditer .env et ajouter votre token HuggingFace
# HF_TOKEN=hf_your_actual_token_here
```

Obtenir un token: https://huggingface.co/settings/tokens

### 5. Vérifier l'installation

```powershell
python scripts\check_installation.py
```

### 6. Construire l'index FAISS

```powershell
python scripts\build_index.py
```

*Durée: ~2-3 minutes (télécharge et indexe les abstracts médicaux)*

### 7. Lancer l'application

```powershell
python src\app.py
```

Interface disponible: http://localhost:7861

## 📁 Structure du Projet

```
LLM-Bot/
├── config.yaml              # Configuration centrale
├── requirements.txt         # Dépendances Python
├── .env.example            # Template d'environnement
├── .gitignore
│
├── src/
│   ├── app.py              # Application Gradio
│   ├── services/
│   │   ├── llm_service.py  # Service Flan-T5-XL
│   │   └── rag_service.py  # Service RAG + FAISS
│   └── utils/
│       └── helpers.py      # Fonctions utilitaires
│
├── scripts/
│   ├── build_index.py      # Construction index FAISS
│   └── check_installation.py  # Vérification installation
│
└── data/
    ├── raw/                # Données brutes
    └── processed/          # Index FAISS
        └── faiss_index/
```

## ⚙️ Configuration

### Modèle LLM (config.yaml)

```yaml
models:
  llm:
    name: "google/flan-t5-xl"
    quantization:
      load_in_8bit: true        # 8-bit pour 4GB VRAM
      device_map: "auto"
    torch_dtype: "float16"
    max_memory:
      0: "4GB"                  # GPU 0
      "cpu": "16GB"
```

### RAG (config.yaml)

```yaml
rag:
  dataset: "TimSchopf/medical_abstracts"
  chunk_size: 500
  chunk_overlap: 100
  top_k: 5
  index_path: "data/processed/faiss_index"
  keywords:                     # Filtrage par mots-clés
    skin_cancer: [...]
    benign_lesions: [...]
    # ... 6 catégories, 80+ termes
```

## 🎮 Utilisation

### Mode 1: Analyse de Symptômes

1. Onglet **"Analyse de Symptômes"**
2. Décrire les symptômes (ex: "Lésion pigmentée avec bords irréguliers")
3. Indiquer la durée (ex: "3 mois")
4. Activer RAG (recommandé)
5. Cliquer **"Analyser"**

### Mode 2: Questions Générales

1. Onglet **"Questions Générales"**
2. Poser une question (ex: "Quels sont les critères ABCDE?")
3. Activer RAG (recommandé)
4. Cliquer **"Demander"**

### Paramètres Avancés

- **Nombre de sources**: 1-10 (défaut: 5)
- **Tokens de génération**: 128-1024 (défaut: 512)
- **Température**: 0.1-1.0 (défaut: 0.7)

## 📊 Spécifications Techniques

### Modèle LLM

- **Nom**: google/flan-t5-xl
- **Taille**: ~3GB (quantisé 8-bit)
- **VRAM**: ~2.5-3GB lors de l'inférence
- **Architecture**: T5 (Text-to-Text Transfer Transformer)

### RAG

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (CPU)
- **Vectorstore**: FAISS (CPU, ~200MB)
- **Dataset**: TimSchopf/medical_abstracts (~50k abstracts)
- **Filtrage**: 80+ keywords dermatologiques (6 catégories)

### Performance

- **Chargement initial**: ~30-60 secondes
- **Génération**: ~5-10 secondes (512 tokens)
- **Recherche RAG**: <1 seconde
- **Mémoire GPU**: ~3GB (peak)

## 🔧 Dépendances Clés

```
torch>=2.0.0               # PyTorch avec CUDA
transformers>=4.36.0       # Hugging Face Transformers
bitsandbytes>=0.41.0       # Quantisation 8-bit
sentencepiece>=0.1.99      # Tokenizer T5
langchain>=0.1.0           # RAG framework
faiss-cpu>=1.7.4           # Recherche vectorielle
sentence-transformers>=2.2.2  # Embeddings
gradio>=4.0.0              # Interface web
```

## 🐛 Dépannage

### Erreur: "CUDA out of memory"

- Réduire `max_memory["0"]` dans config.yaml (ex: "3GB")
- Réduire `max_new_tokens` (ex: 256)
- Fermer autres applications GPU

### Erreur: "Index not found"

```powershell
python scripts\build_index.py
```

### Erreur: "HuggingFace token required"

1. Créer un token: https://huggingface.co/settings/tokens
2. Ajouter dans `.env`: `HF_TOKEN=hf_...`

### Erreur: "ModuleNotFoundError"

```powershell
# Réinstaller dépendances
pip install -r requirements.txt --force-reinstall
```

### Performance lente

- Vérifier GPU utilisé: `torch.cuda.is_available()`
- Installer CUDA 12.1: `conda install pytorch-cuda=12.1 -c nvidia`

## 📝 Notes Importantes

- **Disclaimer**: À usage éducatif uniquement. Consultez toujours un professionnel.
- **Rapports**: Sauvegardés automatiquement (`analysis_YYYYMMDD_HHMMSS.txt`)
- **Compatibilité**: Environnement `rag` partagé avec VLM-Bot (pas de conflit)

## 🔄 Différences avec VLM-Bot

| Fonctionnalité | LLM-Bot | VLM-Bot |
|---------------|---------|---------|
| Modèle | Flan-T5-XL (LLM) | Llava-1.5-7B (VLM) |
| Entrée | Texte uniquement | Image + Texte |
| OpenCV | ❌ Non | ✅ Oui |
| Quantisation | 8-bit | 4-bit |
| VRAM | ~3GB | ~3.5GB |
| Port Gradio | 7861 | 7860 |

## 📚 Ressources

- [Flan-T5 Paper](https://arxiv.org/abs/2210.11416)
- [FAISS Documentation](https://faiss.ai/)
- [Gradio Docs](https://www.gradio.app/docs/)

## 🤝 Contribution

Environnement compatible avec VLM-Bot - utilisez le même env `rag` pour les deux projets.

---

**Version**: 1.0.0  
**License**: Educational Use Only  
**Contact**: Consultez un dermatologue qualifié pour tout avis médical.
