import argparse
from pathlib import Path

from config.settings import BASE_DIR
from src.inference.llm import LLMEngine
from src.inference.embedder import Embedder
from src.memory.vector_store import VectorStore
from src.tools.registry import ToolRegistry
from src.tools.banking import create_banking_tools
from src.agent.orchestrator import AgentOrchestrator
from src.rag.ingest import ingest_products
from src.db.schema import init_db
from src.db.ingest_excel import ingest_excel


def main():
    parser = argparse.ArgumentParser(description="Globus Bank Conversational AI Agent")
    parser.add_argument(
        "--ingest-products",
        action="store_true",
        help="Ingest products.md into Qdrant vector store",
    )
    parser.add_argument(
        "--ingest-excel",
        type=str,
        metavar="FILE",
        help="Ingest Excel file (Customer, Transaction, Card sheets) into SQLite",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset database before ingesting Excel data",
    )
    args = parser.parse_args()

    print("Initializing Conversational AI Agent...")

    # Initialize database
    init_db()

    # Handle Excel ingestion
    if args.ingest_excel:
        excel_path = Path(args.ingest_excel)
        if excel_path.exists():
            print(f"Ingesting Excel file: {excel_path}")
            ingest_excel(str(excel_path), reset=args.reset_db)
            print("Excel ingestion complete!")
        else:
            print(f"Error: Excel file not found: {excel_path}")
        return

    # Initialize components
    llm = LLMEngine()
    embedder = Embedder()
    vector_store = VectorStore()
    tool_registry = ToolRegistry()

    print("Loading models...")
    llm.load()
    embedder.load()

    # Register tools
    create_banking_tools(tool_registry, vector_store, embedder)

    # Handle product ingestion
    if args.ingest_products:
        products_path = BASE_DIR / "products.md"
        if products_path.exists():
            count = ingest_products(str(products_path), embedder, vector_store)
            print(f"Ingested {count} products into vector store.")
        else:
            print(f"Error: {products_path} not found.")
        return

    # Create agent
    agent = AgentOrchestrator(llm, tool_registry)

    print("\n" + "=" * 50)
    print("  Globus Bank AI Assistant")
    print("  Type 'quit' to exit, 'reset' to clear conversation")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Thank you for banking with Globus Bank. Goodbye!")
                break
            if user_input.lower() == "reset":
                agent.reset()
                print("Conversation reset.\n")
                continue

            response = agent.run(user_input)
            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\nThank you for banking with Globus Bank. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
