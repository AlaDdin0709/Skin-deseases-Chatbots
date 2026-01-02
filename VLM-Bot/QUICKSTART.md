# 🚀 Guide de Démarrage Rapide - VLM-Bot

## Installation en 5 minutes

### 1. Créer l'environnement Conda

```bash
# Créer l'environnement 'rag' avec Python 3.10
conda create -n rag python=3.10 -y

# Activer
conda activate rag
```

### 2. Installer PyTorch + CUDA

```bash
# Vérifier votre version CUDA (vous avez CUDA 13.1 compatible)
nvidia-smi

# ✅ RECOMMANDÉ pour RTX 3050 (4GB VRAM) - CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Note: PyTorch officiel ne supporte pas encore CUDA 13.x directement
# Mais CUDA 12.1 est rétrocompatible avec votre driver 591.59
# Alternative si problème: CUDA 11.8
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# OU CPU uniquement (pas de GPU, plus lent)
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### 3. Installer les dépendances

```bash
cd VLM-Bot
pip install -r requirements.txt
```

### 4. Configuration

```bash
# Copier le template
cp .env.example .env

# Éditer .env avec votre token HuggingFace
# Obtenir un token: https://huggingface.co/settings/tokens
# Ajouter: HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

### 5. Construire l'index RAG

```bash
python scripts/build_index.py
```

Attendez 2-3 minutes pendant le téléchargement et l'indexation du dataset médical.

### 6. Lancer l'application

```bash
python src/app.py
```

Ouvrir dans votre navigateur: http://localhost:7860

---

## Vérification rapide

```bash
# Vérifier l'installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Vérifier les dépendances
python -c "import transformers, langchain, cv2, gradio; print('✅ Toutes les dépendances sont installées')"
```

---

## Commandes utiles

```bash
# Activer l'environnement
conda activate rag

# Désactiver
conda deactivate

# Lister les packages installés
conda list

# Mettre à jour une dépendance
pip install --upgrade transformers

# Reconstruire l'index RAG
python scripts/build_index.py

# Lancer l'app avec partage public (Gradio share link)
# Dans config.yaml, changer: share: true
python src/app.py
```

---

## Structure des fichiers

```
VLM-Bot/
├── src/
│   ├── app.py                    # ← Lancer ceci
│   ├── services/
│   │   ├── vlm_service.py        # Phi-3-Vision
│   │   ├── rag_service.py        # FAISS
│   │   └── opencv_service.py     # Extraction features
│   └── utils/helpers.py
│
├── scripts/
│   └── build_index.py            # ← Lancer en premier
│
├── data/processed/
│   └── faiss_index/              # Index généré
│
├── config.yaml                   # Configuration centrale
├── requirements.txt
└── .env                          # Vos secrets (à créer)
```

---

## Troubleshooting rapide

### ❌ ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### ❌ CUDA not available

```bash
# Vérifier CUDA
nvidia-smi

# Réinstaller PyTorch avec CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### ❌ HuggingFace authentication error

```bash
# Login manuel
huggingface-cli login
# Entrer votre token
```

### ❌ Out of Memory (OOM) GPU

Dans `config.yaml`:
```yaml
models:
  vlm:
    max_memory:
      0: "3.5GB"  # Réduire si nécessaire
```

### ❌ Index RAG non trouvé

```bash
python scripts/build_index.py
```

---

## Support

- 📖 README complet: [README.md](README.md)
- 🐛 Issues: [GitHub Issues](https://github.com/votre-repo/issues)
- 📧 Email: support@vlm-bot.example

---

**Version Python recommandée**: 3.10  
**Testé sur**: Windows 11, Ubuntu 22.04, macOS (CPU)
