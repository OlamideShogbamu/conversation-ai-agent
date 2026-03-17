from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Model paths
LLM_MODEL_PATH = BASE_DIR / "models" / "Qwen_Qwen3-4B-Q3_K_M.gguf"
EMBED_MODEL_PATH = BASE_DIR / "models" / "bge-small-en-v1.5-q4_k_m.gguf"

# LLM settings
LLM_CONTEXT_SIZE = 2048
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_THREADS = 10
LLM_BATCH_SIZE = 512

# Embedding settings
EMBED_DIMENSION = 384  # BGE-Small

# Qdrant settings (embedded mode - no Docker needed)
QDRANT_PATH = BASE_DIR / "data" / "qdrant"  # Local storage path
QDRANT_COLLECTION = "globus_ai"

# Agent settings
MAX_AGENT_ITERATIONS = 10
