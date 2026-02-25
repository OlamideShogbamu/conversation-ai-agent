from flask import Flask, request, jsonify
from pathlib import Path
from src.inference.llm import LLMEngine
from src.inference.embedder import Embedder
from src.memory.vector_store import VectorStore
from src.tools.registry import ToolRegistry
from src.tools.banking import create_banking_tools
from src.agent.orchestrator import AgentOrchestrator
from src.db.schema import init_db
from config.settings import BASE_DIR

app = Flask(__name__)

# Initialize components once at startup
llm = None
agent = None
_embedder = None
_vector_store = None

def init_agent():
    global llm, agent, _embedder, _vector_store

    print("Loading models...")
    init_db()

    llm = LLMEngine()
    _embedder = Embedder()
    _vector_store = VectorStore()
    tool_registry = ToolRegistry()

    llm.load()
    _embedder.load()

    create_banking_tools(tool_registry, _vector_store, _embedder)

    agent = AgentOrchestrator(llm, tool_registry)
    print("Agent ready!")

@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Body: {"message": "What loans do you offer?"}
    Response: {"response": "...", "token_usage": int}
    """
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_message = data["message"]

    try:
        response = agent.run(user_message)
        return jsonify({
            "response": response,
            "token_usage": agent.get_token_usage()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    """Reset conversation history"""
    agent.reset()
    return jsonify({"status": "Conversation reset"})

@app.route("/history", methods=["GET"])
def history():
    """
    GET /history
    Returns the current conversation history (excluding system messages).
    Response: {"messages": [...], "summary": "...", "token_usage": int}
    """
    messages = agent.memory.get_history()
    return jsonify({
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "tool_name": msg.tool_name,
            }
            for msg in messages
            if msg.role != "system"
        ],
        "summary": agent.memory.summary,
        "token_usage": agent.get_token_usage(),
    })

@app.route("/ingest", methods=["POST"])
def ingest():
    """
    POST /ingest
    Trigger data ingestion.

    For products (Qdrant vector store):
      Body: {"type": "products"}

    For customer/transaction/card data (SQLite):
      Body: {"type": "excel"}
      Body: {"type": "excel", "file": "/path/to/file.xlsx", "reset": true}
    """
    data = request.get_json() or {}
    ingest_type = data.get("type")

    if ingest_type == "products":
        from src.rag.ingest import ingest_products
        products_path = BASE_DIR / "products.md"
        if not products_path.exists():
            return jsonify({"error": f"products.md not found at {products_path}"}), 404
        count = ingest_products(str(products_path), _embedder, _vector_store)
        return jsonify({"status": "ok", "products_ingested": count})

    elif ingest_type == "excel":
        from src.db.ingest_excel import ingest_excel
        file_path = data.get("file", str(BASE_DIR / "customer_and_banking_data.xlsx"))
        reset = data.get("reset", False)
        if not Path(file_path).exists():
            return jsonify({"error": f"File not found: {file_path}"}), 404
        ingest_excel(file_path, reset=reset)
        return jsonify({"status": "ok", "file": file_path})

    else:
        return jsonify({"error": "Missing or invalid 'type'. Use 'products' or 'excel'"}), 400

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": llm is not None})

if __name__ == "__main__":
    init_agent()
    app.run(host="0.0.0.0", port=5000, debug=False)
