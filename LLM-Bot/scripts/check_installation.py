"""
Check Installation
Vérifie que toutes les dépendances sont installées correctement.
"""

import sys
from pathlib import Path

print("="*80)
print("🔍 LLM-Bot Installation Check")
print("="*80)

# ============================================================================
# Path Configuration
# =================)===========================================================
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# ============================================================================
# Python Version
# ============================================================================
print("\n📌 Python Version:")
print(f"   {sys.version}")
if sys.version_info < (3, 10):
    print("   ⚠️  Python 3.10+ recommended")
else:
    print("   ✅ OK")

# ============================================================================
# Critical Dependencies
# ============================================================================
print("\n📦 Critical Dependencies:")

critical_packages = [
    'torch',
    'transformers',
    'accelerate',
    'bitsandbytes',
    'sentencepiece',
    'protobuf',
    'langchain',
    'langchain_community',
    'faiss',
    'sentence_transformers',
    'datasets',
    'gradio',
    'yaml',
    'dotenv',
    'pydantic',
    'numpy',
    'pandas'
]

missing = []
for package in critical_packages:
    try:
        if package == 'yaml':
            __import__('yaml')
        elif package == 'dotenv':
            __import__('dotenv')
        elif package == 'faiss':
            __import__('faiss')
        elif package == 'protobuf':
            try:
                __import__('protobuf')
            except ImportError:
                __import__('google.protobuf')
        else:
            __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - NOT FOUND")
        missing.append(package)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt")
else:
    print("\n✅ All critical packages installed")

# ============================================================================
# PyTorch & CUDA
# ============================================================================
print("\n🔥 PyTorch & CUDA:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("   ✅ GPU ready")
    else:
        print("   ⚠️  CPU only mode")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================================================
# Transformers
# ============================================================================
print("\n🤗 Transformers:")
try:
    import transformers
    print(f"   Version: {transformers.__version__}")
    
    if tuple(map(int, transformers.__version__.split('.')[:2])) >= (4, 36):
        print("   ✅ Version OK (4.36+)")
    else:
        print("   ⚠️  Version 4.36+ recommended")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================================================
# Project Structure
# ============================================================================
print("\n📁 Project Structure:")

required_paths = [
    'config.yaml',
    'requirements.txt',
    '.env.example',
    'src/app.py',
    'src/services/llm_service.py',
    'src/services/rag_service.py',
    'src/utils/helpers.py',
    'scripts/build_index.py',
    'data/raw',
    'data/processed'
]

for path_str in required_paths:
    path = Path(path_str)
    if path.exists():
        print(f"   ✅ {path_str}")
    else:
        print(f"   ❌ {path_str} - NOT FOUND")

# ============================================================================
# Environment Variables
# ============================================================================
print("\n🔐 Environment:")

env_file = Path('.env')
if env_file.exists():
    print("   ✅ .env file found")
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    hf_token = os.getenv('HF_TOKEN')
    if hf_token and hf_token != 'hf_your_token_here':
        print("   ✅ HF_TOKEN configured")
    else:
        print("   ⚠️  HF_TOKEN not set (required for model downloads)")
else:
    print("   ⚠️  .env file not found")
    print("      Copy .env.example to .env and add your HF_TOKEN")

# ============================================================================
# RAG Index
# ============================================================================
print("\n📊 RAG Index:")

try:
    from src.utils.helpers import load_config
    config = load_config()
    index_path = Path(config['rag']['index_path'])
    
    if index_path.exists():
        print(f"   ✅ Index found at: {index_path}")
    else:
        print(f"   ⚠️  Index not found at: {index_path}")
        print("      Run: python scripts/build_index.py")
except Exception as e:
    print(f"   ❌ Error checking index: {e}")

# ============================================================================
# Model Test
# ============================================================================
print("\n🧪 Quick Model Test:")

try:
    import torch
    from transformers import AutoTokenizer
    
    print("   Testing tokenizer download...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    test_text = "What is dermatology?"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"   ✅ Tokenizer works (encoded {len(tokens['input_ids'][0])} tokens)")
    
except Exception as e:
    print(f"   ⚠️  Tokenizer test failed: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)

if missing:
    print("❌ INSTALLATION INCOMPLETE")
    print(f"   Missing packages: {', '.join(missing)}")
    print("\n   Next steps:")
    print("   1. Install PyTorch: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y")
    print("   2. Install deps: pip install -r requirements.txt")
else:
    print("✅ INSTALLATION LOOKS GOOD!")
    print("\n   Next steps:")
    if not env_file.exists():
        print("   1. Copy .env.example to .env and add HF_TOKEN")
    if not Path(config['rag']['index_path']).exists():
        print("   2. Build index: python scripts/build_index.py")
    print("   3. Run app: python src/app.py")

print("="*80)
