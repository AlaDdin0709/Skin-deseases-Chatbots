# ⚡ LLM-Bot - Guide de Démarrage Rapide (5 minutes)

## 🎯 Installation Express

### Étape 1: Environnement (30 secondes)

```powershell
# Activer l'environnement existant
conda activate rag

# OU créer un nouvel environnement
conda create -n rag python=3.10 -y
conda activate rag
```

### Étape 2: PyTorch + CUDA (2-3 minutes)

```powershell
cd E:\Chatbots\LLM-Bot
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Étape 3: Dépendances (1-2 minutes)

```powershell
pip install -r requirements.txt
```

### Étape 4: Configuration (30 secondes)

```powershell
# Copier template
copy .env.example .env

# Éditer .env avec Notepad
notepad .env

# Ajouter votre token HuggingFace:
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

**Obtenir un token:** https://huggingface.co/settings/tokens

### Étape 5: Vérification (15 secondes)

```powershell
python scripts\check_installation.py
```

Vérifier: ✅ Tous les packages OK, ✅ GPU ready

### Étape 6: Index FAISS (2-3 minutes)

```powershell
python scripts\build_index.py
```

Attendez: "✅ Index built successfully!"

### Étape 7: Lancement (10 secondes)

```powershell
python src\app.py
```

Ouvrir: http://localhost:7861

## 🎮 Premier Test

### Option A: Analyse de Symptômes

1. Onglet **"Analyse de Symptômes"**
2. Symptômes: `Lésion pigmentée asymétrique avec bords irréguliers et plusieurs couleurs`
3. Durée: `2 mois`
4. ✅ RAG activé
5. Cliquer **"Analyser"**

### Option B: Question Générale

1. Onglet **"Questions Générales"**
2. Question: `Quels sont les critères ABCDE pour le diagnostic de mélanome?`
3. ✅ RAG activé
4. Cliquer **"Demander"**

## ⚙️ Configuration Minimale

### GPU RTX 3050 (4GB VRAM)

Dans `config.yaml`, vérifier:

```yaml
models:
  llm:
    quantization:
      load_in_8bit: true        # ESSENTIEL pour 4GB
    max_memory:
      0: "4GB"                  # Limite GPU
      "cpu": "16GB"             # Overflow vers CPU
```

### Paramètres Recommandés

- **Tokens**: 512 (défaut)
- **Température**: 0.7 (défaut)
- **Sources RAG**: 5 (défaut)

## 🐛 Dépannage Express

### ❌ "CUDA out of memory"

```yaml
# config.yaml - Réduire limite GPU
max_memory:
  0: "3GB"      # Au lieu de 4GB
```

### ❌ "Index not found"

```powershell
python scripts\build_index.py
```

### ❌ "HuggingFace token required"

1. https://huggingface.co/settings/tokens → Create token
2. `.env` → `HF_TOKEN=hf_votre_token`

### ❌ "Module not found"

```powershell
pip install -r requirements.txt
```

## 📊 Utilisation Mémoire

| Composant | VRAM | RAM |
|-----------|------|-----|
| Flan-T5-XL (8-bit) | ~2.5GB | ~1GB |
| Embeddings (CPU) | 0GB | ~500MB |
| FAISS (CPU) | 0GB | ~200MB |
| **Total** | **~3GB** | **~2GB** |

## ✅ Checklist Rapide

- [ ] Conda env `rag` activé
- [ ] PyTorch CUDA 12.1 installé
- [ ] Toutes dépendances installées
- [ ] `.env` avec `HF_TOKEN` configuré
- [ ] `check_installation.py` → Tout OK
- [ ] Index FAISS construit
- [ ] App lancée sur port 7861
- [ ] Test de question → Réponse reçue

## 🎯 Commandes Essentielles

```powershell
# Activer environnement
conda activate rag

# Vérifier installation
python scripts\check_installation.py

# Reconstruire index (si keywords changés)
python scripts\build_index.py

# Lancer application
python src\app.py

# Arrêter application
Ctrl+C
```

## 📝 Différences avec VLM-Bot

✅ Même environnement `rag`  
✅ Pas de conflit de dépendances  
❌ Pas de VLM (Llava)  
❌ Pas de OpenCV  
✅ Texte uniquement (pas d'images)  
✅ Port différent (7861 vs 7860)

## 🚀 Prêt!

Votre LLM-Bot est maintenant opérationnel.

**Port**: http://localhost:7861  
**Mode 1**: Analyse de Symptômes  
**Mode 2**: Questions Générales

⚠️ **IMPORTANT**: Usage éducatif uniquement. Consultez toujours un professionnel de santé.

---

Pour plus de détails: Voir **README.md**
