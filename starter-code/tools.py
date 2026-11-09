"""Local tools used by the VinAssistant lab.

The tools deliberately use JSON files instead of external services so that the lab is
repeatable and can be run without an API key.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")


def search_product_catalog(
    category: str, max_price: int = 999_999_999_999
) -> List[Dict[str, Any]]:
    """Return catalog items in *category* costing no more than *max_price*.

    Category comparisons are case-insensitive and the price limit is inclusive.
    Returning a structured error for a missing fixture makes a configuration issue
    visible to an agent instead of silently presenting it as an empty search.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    with open(catalog_file, "r", encoding="utf-8") as file:
        products = json.load(file)

    if not isinstance(products, list):
        return [{"error": "Product catalog has an invalid format."}]

    normalized_category = str(category).strip().casefold()
    try:
        price_limit = int(max_price)
    except (TypeError, ValueError):
        return []

    return [
        product
        for product in products
        if isinstance(product, dict)
        and str(product.get("category", "")).casefold() == normalized_category
        and isinstance(product.get("price_vnd"), (int, float))
        and product["price_vnd"] <= price_limit
    ]


def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium",
) -> Dict[str, Any]:
    """Create and persist a new open support ticket.

    Existing records are retained.  The sequence is calculated per day so ticket
    IDs do not collide merely because old tickets from other dates exist.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as file:
            loaded_tickets = json.load(file)
        if isinstance(loaded_tickets, list):
            existing_tickets = loaded_tickets

    today = datetime.now().strftime("%Y%m%d")
    prefix = f"TK-{today}-"
    daily_sequences = []
    for ticket in existing_tickets:
        ticket_id = str(ticket.get("ticket_id", "")) if isinstance(ticket, dict) else ""
        if ticket_id.startswith(prefix):
            try:
                daily_sequences.append(int(ticket_id.rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                continue
    ticket_id = f"{prefix}{max(daily_sequences, default=0) + 1:03d}"

    normalized_priority = str(priority).strip().lower()
    if normalized_priority not in {"low", "medium", "high"}:
        normalized_priority = "medium"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": str(customer_name).strip(),
        "issue_description": str(issue_description).strip(),
        "priority": normalized_priority,
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": "general",
    }
    existing_tickets.append(new_ticket)

    with open(tickets_file, "w", encoding="utf-8") as file:
        json.dump(existing_tickets, file, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": new_ticket["customer_name"],
        "priority": normalized_priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# JSON schemas allow an LLM/function-calling provider to discover the local tools.
TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' hoặc 'du_lich'.",
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ.",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": "Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Tên khách hàng.",
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả vấn đề cần hỗ trợ.",
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên của yêu cầu.",
                    "enum": ["low", "medium", "high"],
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket,
}
