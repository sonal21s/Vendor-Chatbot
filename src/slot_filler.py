import json
from groq import Groq
from settings import GROQ_API_KEY, LLM_MODEL
from src.utils import get_logger

log = get_logger(__name__)

_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a query parser for a vendor database. Extract structured filters from the user's question.

Return ONLY valid JSON matching this schema:
{
  "intent": "count" | "list" | "lookup" | "details",
  "state": string or null,
  "city": string or null,
  "work_type": string or null,
  "vendor_name": string or null,
  "recommendation": "Recommended" | "Good" | "Risky" | null,
  "exclude_risky": true | false,
  "min_rating": number or null
}

Rules:
- "intent" is "count" for "how many" questions, "list" for "list/show all", "lookup" for "tell me about X" or "details of X", "details" same as lookup.
- "state" must be normalized to the full canonical Indian state name. Examples:
    * "andhra" / "AP" / "andra pradesh" → "Andhra Pradesh"
    * "MH" / "maharastra" / "maha" → "Maharashtra"
    * "tamil" / "TN" / "tamilnadu" → "Tamil Nadu"
    * "UP" / "uttar" → "Uttar Pradesh"
    * "MP" / "madhya" → "Madhya Pradesh"
    * "WB" / "bengal" → "West Bengal"
    * "delhi" / "new delhi" / "DL" → "Delhi"
  If the state is genuinely ambiguous (e.g. just "uttar" could be Uttar Pradesh or Uttarakhand), pick the most populous match.
- For "city" and "vendor_name", preserve user spelling — do NOT correct typos. Fuzzy matching happens downstream.
- "recommendation" maps user phrasing to one of three tags:
    * "Recommended" — user asks for recommended/preferred/top/best vendors
    * "Good" — user asks for good/decent/acceptable vendors
    * "Risky" — user asks for risky/problematic vendors
    * null — recommendation tier not mentioned
- "exclude_risky" is true if user explicitly wants to avoid risky vendors ("not risky", "avoid risky", "safe vendors"), false otherwise.
- Set fields to null when not mentioned.
- Output JSON only, no prose."""


def extract_slots(query: str, history: list[dict] | None = None) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-4:])  # last 2 turns for follow-up context
    messages.append({"role": "user", "content": query})

    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content
    log.info("Slot extraction: %s", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Slot filler returned invalid JSON: %s", raw)
        return {"intent": "lookup", "state": None, "city": None,
                "work_type": None, "vendor_name": None,
                "recommendation": None, "exclude_risky": False,
                "min_rating": None}
