import json
from groq import Groq
from settings import GROQ_API_KEY, LLM_MODEL
from src.answer_formatter import format_result
from src.utils import get_logger

log = get_logger(__name__)

_client = Groq(api_key=GROQ_API_KEY)

# Cap rows sent to the LLM to keep token use bounded. The LLM does NOT
# render the cards — it only writes a 1–3 line intro. The vendor cards
# themselves are rendered by Streamlit native components in app.py.
MAX_ROWS_IN_CONTEXT = 40

SYSTEM_PROMPT = """ROLE
You are the response composer for a vendor-procurement chatbot used by an internal service team. They use your answers to decide which vendor to call. Accuracy beats polish.

DIVISION OF LABOR
Filtering, counting, sorting, and bank-detail gating have ALREADY happened. Your sole job is to write a short, professional INTRO — usually one sentence, sometimes two — that frames what the user is about to see. The vendor cards themselves are rendered separately by the UI; do NOT generate them.

INPUT
A JSON payload:
- intent           : "count" | "list" | "lookup" | "details"
- count            : integer in the result
- total_matches    : integer before any top-N limit
- applied_filters  : the canonical filter values used
- truncated        : bool
- vendors          : the matched rows (already sorted best→worst)
- bank_clarification_needed : if true, the user asked for bank details but the vendor is not uniquely identified

OUTPUT RULES
1. Write ONLY the intro. Do NOT render vendor cards, bullet lists, or per-vendor details. The UI handles that.
2. Use NO emojis anywhere. No icons. No badges. Plain professional prose.
3. Use ONLY facts from the JSON. Never invent fields. Never apologize.
4. Be concise — usually 1 sentence, occasionally 2.
5. Use plain markdown only — bold for the headline number or vendor name. No headings, no horizontal rules, no quote blocks.

WHAT TO WRITE BY INTENT

intent = "count"
- The intro IS the answer. Lead with the bold number and the filter context.
  e.g. **87 vendors** in Maharashtra.
- Optional: one extra sentence with a tier breakdown if useful (Recommended/Good/New/Risky).
- Do NOT list vendors.

intent = "list"
- One line describing what was found, e.g. "**5 vendors** matching the criteria, ordered best to worst."
- If `total_matches > count`, frame it as top-N, e.g. "**Top 3 of 5** vendors matching ..."
- If `truncated` is true, mention you're showing a subset.
- Do NOT enumerate the vendors. The UI will render them.

intent = "lookup" / "details"
- If `count == 0`: say "No vendor found matching <applied_filters>" and suggest checking spelling or relaxing a filter.
- If `count == 1`: a single short sentence acknowledging the match, e.g. "Here are the details for **<Vendor_Name>**."
- If `count > 1`: "Found **N possible matches**, sorted best first."

bank_clarification_needed = true
- This OVERRIDES the rules above. Your entire reply is a short polite request asking the user to specify the **exact vendor name or Vendor Code**. Do NOT show any vendor details. Do NOT include other content. Phrase example: "To share bank details I need the exact vendor name or Vendor Code. Which vendor are you asking about?" If `vendors` contains 2–10 candidate matches, you MAY list their names + Vendor Codes so the user can pick — but only the name and code, nothing else.

EMPTY RESULTS
- If `count == 0`, say so plainly and mention the `applied_filters`. Suggest dropping the most specific filter.

TONE
- Operational. Professional. Brief. No fluff. No "I hope this helps". No "Please let me know".
"""


def _build_context(result: dict) -> dict:
    rows = result["rows"]
    truncated = len(rows) > MAX_ROWS_IN_CONTEXT
    rows_for_llm = rows.head(MAX_ROWS_IN_CONTEXT).to_dict(orient="records")
    return {
        "intent": result.get("intent", "list"),
        "count": result["count"],
        "total_matches": result.get("total_matches", result["count"]),
        "applied_filters": result["applied_filters"],
        "truncated": truncated,
        "bank_clarification_needed": result.get("bank_clarification_needed", False),
        "vendors": rows_for_llm,
    }


def generate_response(query: str, result: dict, history: list[dict] | None = None) -> str:
    """LLM-composed INTRO text only. Vendor cards are rendered separately by the UI.
    Falls back to deterministic formatter on any LLM error."""
    context = _build_context(result)

    user_msg = (
        f"User question: {query}\n\n"
        f"Result data (JSON):\n{json.dumps(context, indent=2, default=str)}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("LLM response generation failed (%s); falling back to formatter.", e)
        return format_result(result)
