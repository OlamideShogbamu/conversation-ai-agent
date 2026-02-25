from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Model paths
LLM_MODEL_PATH = BASE_DIR / "models" / "Qwen_Qwen3-4B-Q3_K_M.gguf"
EMBED_MODEL_PATH = BASE_DIR / "models" / "bge-m3-Q4_K_M.gguf"

# LLM settings
LLM_CONTEXT_SIZE = 4096
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_THREADS = 4

# Embedding settings
EMBED_DIMENSION = 1024  # BGE-M3

# Qdrant settings (embedded mode - no Docker needed)
QDRANT_PATH = BASE_DIR / "data" / "qdrant"  # Local storage path
QDRANT_COLLECTION = "globus_ai"

# Agent settings
MAX_AGENT_ITERATIONS = 10
