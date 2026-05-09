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
  "min_rating": number or null,
  "limit": integer or null
}

CRITICAL — handling conversation history:
- Each user turn is treated as INDEPENDENT by default. Extract only what is explicitly stated in the CURRENT user message.
- Do NOT inherit filters from earlier turns (e.g. earlier work_type, earlier recommendation tier) unless the current turn is clearly a referential follow-up.
- A turn is a referential follow-up ONLY when it uses pronouns or refers back explicitly: "show me the recommended ones", "now filter by Mumbai", "and how many of those are risky?". In those cases, merge the prior filters with the new ones.
- A turn that names a fresh subject ("list vendors from Andhra", "vendors in Pune") is a NEW query — discard prior work_type, recommendation, etc., even if recent turns mentioned them.

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
- "recommendation" maps user phrasing to one of three explicit tier names:
    * "Recommended" — user explicitly says "recommended" or "preferred" vendors
    * "Good" — user explicitly says "good" / "decent" / "acceptable"
    * "Risky" — user explicitly says "risky" / "problematic"
    * null — tier not explicitly mentioned
  IMPORTANT: words like "top", "best", "highest", "5 best" do NOT mean "Recommended". They mean "highest-scored" — the data is already sorted by score, so set recommendation = null and use the "limit" field instead.
- "exclude_risky" is true if user explicitly wants to avoid risky vendors ("not risky", "avoid risky", "safe vendors"), false otherwise.
- "limit" captures phrases like "top 5", "best 3", "show me 10", "first 7". Extract the integer. If the user says "top vendors" without a number, leave as null. If no limit phrasing, leave as null.
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
                "min_rating": None, "limit": None}
