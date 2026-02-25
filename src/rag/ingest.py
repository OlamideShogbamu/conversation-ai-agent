import re
from pathlib import Path
from src.inference.embedder import Embedder
from src.memory.vector_store import VectorStore


def parse_products_md(filepath: str) -> list[dict]:
    """Parse products.md into structured product data."""
    content = Path(filepath).read_text()
    products = []

    # Split by product categories
    sections = re.split(r"\n(?=Debit Cards|Loan Products|Investment Products|Saving Products)", content)

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split("\n")
        if not lines:
            continue

        # Determine category
        first_line = lines[0].strip()
        if "Debit Cards" in first_line:
            products.extend(_parse_debit_cards(lines))
        elif "Loan Products" in first_line:
            products.extend(_parse_loans(lines))
        elif "Investment Products" in first_line:
            products.extend(_parse_investments(lines))
        elif "Saving Products" in first_line:
            products.extend(_parse_savings(lines))

    return products


def _parse_debit_cards(lines: list[str]) -> list[dict]:
    text = "\n".join(lines)
    desc_match = re.search(r"Description:\s*(.+?)(?=Features:|$)", text, re.DOTALL)
    feat_match = re.search(r"Features:\s*(.+?)$", text, re.DOTALL)

    return [
        {
            "id": "debit_cards",
            "name": "Globus Bank Debit Cards",
            "category": "cards",
            "description": desc_match.group(1).strip() if desc_match else "",
            "features": [f.strip() for f in feat_match.group(1).split(".") if f.strip()] if feat_match else [],
        }
    ]


def _parse_loans(lines: list[str]) -> list[dict]:
    products = []
    text = "\n".join(lines)

    loan_types = {
        "salary_advance": "Salary Advance Loan",
        "home_extension": "Home Extension Loan",
        "personal": "Globus Personal Loan",
        "business": "Business Loan",
        "mortgage": "Mortgage Loan",
        "vehicle": "Vehicle Loan",
        "overdraft": "Overdraft",
        "working_capital": "Working Capital Loan",
    }

    for loan_id, loan_name in loan_types.items():
        pattern = rf"\d+\.\s*{re.escape(loan_name)}[:\s]*(.+?)(?=\d+\.|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            desc = match.group(1).strip()
            products.append(
                {
                    "id": f"loan_{loan_id}",
                    "name": loan_name,
                    "category": "loans",
                    "description": desc,
                    "features": _extract_features(desc),
                }
            )

    return products


def _parse_investments(lines: list[str]) -> list[dict]:
    products = []
    text = "\n".join(lines)

    inv_types = {
        "bonds": "Bonds",
        "commercial_papers": "Commercial Papers",
        "money_market": "Money Market Deposits",
        "treasury_bills": "Treasury Bills",
    }

    for inv_id, inv_name in inv_types.items():
        pattern = rf"\d+\.\s*{re.escape(inv_name)}\s*Description:\s*(.+?)(?=Benefits:|Features:|$)"
        match = re.search(pattern, text, re.DOTALL)
        desc = match.group(1).strip() if match else ""

        feat_pattern = rf"{re.escape(inv_name)}.*?Features:\s*(.+?)(?=Benefits:|$|\d+\.)"
        feat_match = re.search(feat_pattern, text, re.DOTALL)
        features = feat_match.group(1).strip() if feat_match else ""

        ben_pattern = rf"{re.escape(inv_name)}.*?Benefits:\s*(.+?)(?=\d+\.|$)"
        ben_match = re.search(ben_pattern, text, re.DOTALL)
        benefits = ben_match.group(1).strip() if ben_match else ""

        products.append(
            {
                "id": f"investment_{inv_id}",
                "name": inv_name,
                "category": "investments",
                "description": desc,
                "features": [f.strip() for f in features.split(".") if f.strip()][:5],
                "benefits": [b.strip() for b in benefits.split(".") if b.strip()][:5],
            }
        )

    return products


def _parse_savings(lines: list[str]) -> list[dict]:
    products = []
    text = "\n".join(lines)

    sav_types = {
        "domiciliary": "Globus Domiciliary Account",
        "kiddies": "Globus Kiddies Account",
        "savings": "Savings Account",
        "current": "Individual Current Account",
        "non_resident": "Non Resident Account",
    }

    for sav_id, sav_name in sav_types.items():
        pattern = rf"\d+\.\s*{re.escape(sav_name)}[^\n]*\nDescription:\s*(.+?)(?=Features:|$)"
        match = re.search(pattern, text, re.DOTALL)
        desc = match.group(1).strip() if match else ""

        feat_pattern = rf"{re.escape(sav_name)}.*?Features:\s*(.+?)(?=\d+\.|$)"
        feat_match = re.search(feat_pattern, text, re.DOTALL)
        features = feat_match.group(1).strip() if feat_match else ""

        products.append(
            {
                "id": f"savings_{sav_id}",
                "name": sav_name,
                "category": "savings",
                "description": desc,
                "features": [f.strip() for f in features.split(".") if f.strip()][:5],
            }
        )

    return products


def _extract_features(text: str) -> list[str]:
    features = []
    for keyword in ["flexible", "quick", "competitive", "short-term", "long-term"]:
        if keyword in text.lower():
            features.append(keyword)
    return features


def ingest_products(
    filepath: str,
    embedder: Embedder,
    vector_store: VectorStore,
):
    """Parse products.md and ingest into vector store."""
    products = parse_products_md(filepath)
    vector_store.create_collection()

    for product in products:
        # Create embedding from name + description
        embed_text = f"{product['name']}: {product['description']}"
        vector = embedder.embed(embed_text)
        vector_store.upsert(product["id"], vector, product)

    return len(products)
