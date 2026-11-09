"""Lab #4: System Prompt Engineering & Tool Calling Engine.

This module uses deterministic local intent routing so the exercise can be run
without an LLM API key, while preserving the same tool-calling workflow.
"""

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP

SYSTEM_PROMPT = """
You are VinAssistant, the official AI assistant for the Vingroup ecosystem.

## PERSONA
- Name: VinAssistant.
- Role: Professional advisor for VinFast products and Vinpearl services.
- Style: Friendly, concise, respectful, and accurate in Vietnamese.

## AVAILABLE TOOLS
- search_product_catalog(category, max_price): find VinFast/Vinpearl catalog
  products using current local catalog data.
- submit_support_ticket(customer_name, issue_description, priority): create an
  open customer-support ticket.

## CORE RULES
1. Never invent product prices, availability, product features, or ticket IDs.
2. Call the relevant tool before answering catalog or support-ticket requests.
3. Clearly say when a search returns no suitable result.
4. Keep personal data limited to what is necessary to create a support ticket.

## OPERATIONAL BOUNDARIES
Answer only requests related to Vingroup, including VinFast and Vinpearl. For
requests outside this scope, politely explain the limitation.

## OUTPUT CONTRACT
Use the internal sequence Thought -> Action -> Observation -> Final Answer.
Only present the useful Final Answer to the customer; trace tool calls separately.
""".strip()


class ChatbotBaseline:
    """A deliberately tool-free baseline, useful for comparing grounded answers."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline",
        }


def _fold_text(value: str) -> str:
    """Lowercase text and remove Vietnamese accents for robust keyword matching."""
    normalized = unicodedata.normalize("NFD", value.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")


def _parse_price(text: str) -> Optional[int]:
    """Extract a VND ceiling from phrases such as 'dưới 600 triệu' or '6 trieu'."""
    folded = _fold_text(text).replace(",", ".")
    match = re.search(r"(?:duoi|toi da|khong qua|gia[^.]{0,25}?)(\d+(?:\.\d+)?)\s*(ty|trieu|nghin|k)?", folded)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2) or ""
    multipliers = {"ty": 1_000_000_000, "trieu": 1_000_000, "nghin": 1_000, "k": 1_000}
    return int(amount * multipliers.get(unit, 1))


def _extract_name(text: str) -> str:
    patterns = [
        r"(?:tôi tên|toi ten|tên tôi là|ten toi la)\s+([^,.\n]+)",
        r"(?:tôi là|toi la)\s+([^,.\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .")
    return "Khách hàng"


def _extract_issue(text: str) -> str:
    """Remove introductory name/priority clauses while retaining the reported issue."""
    issue = re.sub(
        r"(?:tôi tên|toi ten|tên tôi là|ten toi la|tôi là|toi la)\s+[^,.\n]+[,;:.]*\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    issue = re.split(r"(?:[.\n]|,)?\s*(?:đây là|day la|mức độ|muc do|cần xử lý|can xu ly)", issue, maxsplit=1, flags=re.IGNORECASE)[0]
    issue = issue.strip(" .,:;\n")
    return issue or "Khách hàng cần hỗ trợ."


class ToolCallingAgent:
    """A bounded, observable local tool-calling agent."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        folded = _fold_text(user_input)
        is_faq = any(keyword in folded for keyword in ("bao hanh", "chinh sach", "bao lau"))
        catalog_category: Optional[str] = None
        if any(keyword in folded for keyword in ("resort", "vinpearl", "khach san", "du lich")):
            catalog_category = "du_lich"
        elif any(keyword in folded for keyword in ("xe dien", "vinfast", "vf 3", "vf 5", "vf 6", "vf 7", "vf 8", "vf 9")):
            catalog_category = "xe_dien"

        ticket_keywords = (
            "loi", "su co", "ho tro", "phan hoi", "khieu nai", "ghi nhan",
            "am moc", "hong", "gấp", "gap", "nghiem trong",
        )
        needs_ticket = any(
            re.search(rf"\b{re.escape(keyword)}\b", folded) is not None
            for keyword in ticket_keywords
        )
        catalog_request_keywords = (
            "xem", "tim", "tra cuu", "gia", "duoi", "toi da", "khong qua",
            "san pham", "resort", "dat phong", "lua chon",
        )
        # A fault report can mention a VF model; it is not automatically a catalog
        # request.  Policy questions likewise do not need catalog data.
        needs_catalog = (
            catalog_category is not None
            and not is_faq
            and any(keyword in folded for keyword in catalog_request_keywords)
        )

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq and not needs_ticket,
            "category": catalog_category,
            "max_price": _parse_price(user_input) or 999_999_999_999,
        }

    @staticmethod
    def _ticket_priority(folded_input: str) -> str:
        if any(keyword in folded_input for keyword in ("nghiem trong", "gap", "khan cap")):
            return "high"
        if "thap" in folded_input:
            return "low"
        return "medium"

    @staticmethod
    def _format_catalog_answer(products: List[Dict[str, Any]]) -> str:
        errors = [product["error"] for product in products if isinstance(product, dict) and "error" in product]
        if errors:
            return f"Rất tiếc, hiện chưa thể tra cứu danh mục: {errors[0]}"
        if not products:
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp với tiêu chí của bạn."
        items = [f"{product['name']} ({product['price_vnd']:,} VNĐ)" for product in products]
        return "Các lựa chọn phù hợp: " + "; ".join(items) + "."

    def run(self, user_input: str) -> Dict[str, Any]:
        """Detect independent intents, execute required tools, and synthesize an answer."""
        self.trace = [{"step": "init", "user_input": user_input}]
        intents = self._detect_intents(user_input)
        self.trace.append({"step": "intent_detection", "intents": intents.copy()})

        if self.max_iterations < 1:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached",
            }

        answers: List[str] = []
        executed_steps = 0

        # Execute both branches independently: one customer message can need both tools.
        if intents["needs_catalog"]:
            if executed_steps >= self.max_iterations:
                return self._max_iterations_result(executed_steps)
            arguments = {"category": intents["category"], "max_price": intents["max_price"]}
            observation = TOOL_MAP["search_product_catalog"](**arguments)
            executed_steps += 1
            self.trace.append({
                "iteration": executed_steps,
                "action": "search_product_catalog",
                "arguments": arguments,
                "observation": observation,
            })
            answers.append(self._format_catalog_answer(observation))

        if intents["needs_ticket"]:
            if executed_steps >= self.max_iterations:
                return self._max_iterations_result(executed_steps)
            arguments = {
                "customer_name": _extract_name(user_input),
                "issue_description": _extract_issue(user_input),
                "priority": self._ticket_priority(_fold_text(user_input)),
            }
            observation = TOOL_MAP["submit_support_ticket"](**arguments)
            executed_steps += 1
            self.trace.append({
                "iteration": executed_steps,
                "action": "submit_support_ticket",
                "arguments": arguments,
                "observation": observation,
            })
            answers.append(
                f"Đã tạo ticket {observation['ticket_id']} cho {observation['customer_name']} "
                f"với mức ưu tiên {observation['priority']}."
            )

        if intents["is_faq"]:
            executed_steps = max(executed_steps, 1)
            faq_answer = "Pin xe điện VinFast được bảo hành đến 10 năm theo chính sách áp dụng cho từng dòng xe."
            self.trace.append({"iteration": executed_steps, "action": "faq", "observation": faq_answer})
            answers.append(faq_answer)

        if not answers:
            executed_steps = max(executed_steps, 1)
            answer = "VinAssistant chỉ hỗ trợ thông tin về các sản phẩm và dịch vụ thuộc Vingroup."
            self.trace.append({"iteration": executed_steps, "action": "scope_response", "observation": answer})
            answers.append(answer)

        return {
            "answer": "\n".join(answers),
            "trace": self.trace,
            "iterations": executed_steps,
            "status": "completed",
        }

    def _max_iterations_result(self, executed_steps: int) -> Dict[str, Any]:
        self.trace.append({"step": "max_iterations_reached", "iterations": executed_steps})
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": executed_steps,
            "status": "max_iterations_reached",
        }


def main() -> None:
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."
    print("=== RUNNING CHATBOT BASELINE ===")
    print(ChatbotBaseline().query(user_query))
    print("\n=== RUNNING TOOL CALLING AGENT ===")
    result = ToolCallingAgent(max_iterations=5).run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(result["trace"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
