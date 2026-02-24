import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

import re


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=128,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }
# ──────────────────────────────────────────────
# Argument coercion & validation
# ──────────────────────────────────────────────

def coerce_argument_types(calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        schema = tool_map.get(call.get("name"), {}).get("parameters", {}).get("properties", {})
        for arg_name, arg_schema in schema.items():
            val = call.get("arguments", {}).get(arg_name)
            if val is None:
                continue
            if arg_schema["type"] == "integer":
                if isinstance(val, float):
                    call["arguments"][arg_name] = int(val)
                elif isinstance(val, str):
                    m = re.search(r"\d+", val)
                    if m:
                        call["arguments"][arg_name] = int(m.group())
            if arg_schema["type"] == "string" and isinstance(val, str):
                call["arguments"][arg_name] = val.strip()
                if "time" in arg_name.lower():
                    m = re.search(r"\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?", val)
                    if m:
                        call["arguments"][arg_name] = m.group().strip()


def validate_required_arguments(calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        tool_def = tool_map.get(call["name"])
        if not tool_def:
            return False
        required = tool_def["parameters"].get("required", [])
        args = call.get("arguments", {})
        for r in required:
            if r not in args or args[r] in ("", None):
                return False
    return True
# ──────────────────────────────────────────────
# Intent detection
# ──────────────────────────────────────────────

_TOOL_PATTERNS = [
    (r"\b(?:weather|temperature|forecast|how'?s? the weather|check the weather)\b", "get_weather"),
    (
        r"\b(?:set an alarm|alarm|wake me up|wake up at|make sure i wake up|ensure i wake up)\b",
        "set_alarm"
    ),
    (r"\b(?:set\s+.*?timer|timer|countdown)\b", "set_timer"),
    (r"\b(?:send a message|send message|text|msg me)\b", "send_message"),
    (r"\b(?:remind(?:er)?|remind me)\b", "create_reminder"),
    (r"\b(?:play)\b", "play_music"),
   ( r"\b(?:find|look up|search(?:\s+for)?)\s+([A-Za-z][A-Za-z' -]{1,40})", "search_contacts"),
]

def detect_tool_hints(text):
    seen, hints = set(), []
    for pattern, tool_name in _TOOL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            if tool_name not in seen:
                seen.add(tool_name)
                hints.append(tool_name)
    return hints


def count_expected_intents(text):
    hints = detect_tool_hints(text)
    seps = len(re.findall(r"\s+and\s+|\s+then\s+|,\s+|\s+also\s+", text, re.IGNORECASE))
    return max(len(hints), seps + 1, 1)


def split_intents_advanced(text):
    parts = re.split(r"\s+and\s+|\s+then\s+|,\s*|\s+also\s+", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]

# ──────────────────────────────────────────────
# Semantic signal detectors (Intent-aware layer)
# ──────────────────────────────────────────────

def normalize_text(text):
    text = re.sub(r"[!?]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_absolute_time(text):
    return re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        text,
        re.IGNORECASE
    )


def detect_relative_time(text):
    return re.search(
        r"\b(?:in\s+)?(\d+)\s*(minute|minutes|min|mins)\b",
        text,
        re.IGNORECASE
    )


def expresses_wake_intent(text):
    return re.search(
        r"\b("
        r"wake me up|wake up|"
        r"be up|be awake|"
        r"make sure i'?m up|"
        r"ensure i'?m up|"
        r"i need to be up|get me up"
        r")\b",
        text,
        re.IGNORECASE
    )


def expresses_reminder_intent(text):
    return re.search(
        r"\b(remind|reminder|don'?t let me forget)\b",
        text,
        re.IGNORECASE
    )


def expresses_message_intent(text):
    return re.search(
        r"\b(send|text|message|tell|let\s+\w+\s+know)\b",
        text,
        re.IGNORECASE
    )


def expresses_weather_intent(text):
    return re.search(
        r"\b(weather|forecast|temperature|rain|snow|storm)\b",
        text,
        re.IGNORECASE
    )
# ──────────────────────────────────────────────
# Fast deterministic extractors
# Each returns a call dict OR None if not confident
# ──────────────────────────────────────────────

def _extract_alarm(text):
    if not text:
        return None

    text = normalize_text(text)

    # Require wake OR alarm intent
    if not (expresses_wake_intent(text) or re.search(r"\balarm\b", text, re.IGNORECASE)):
        return None

    # Prevent relative timer confusion
    if detect_relative_time(text):
        return None

    m = detect_absolute_time(text)
    if not m:
        return None

    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    ampm = m.group(3)

    if minute > 59 or hour > 23:
        return None

    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour != 12:
            hour += 12

    return {
        "name": "set_alarm",
        "arguments": {"hour": hour, "minute": minute}
    }

def _extract_timer(text):
    if not text:
        return None

    text = normalize_text(text)

    rel = detect_relative_time(text)
    if not rel:
        return None

    minutes = int(rel.group(1))
    if minutes <= 0:
        return None

    # If wake intent + relative → timer
    if expresses_wake_intent(text):
        return {
            "name": "set_timer",
            "arguments": {"minutes": minutes}
        }

    if re.search(r"\btimer|countdown\b", text, re.IGNORECASE):
        return {
            "name": "set_timer",
            "arguments": {"minutes": minutes}
        }

    return None


def _extract_weather(text):
    if not text:
        return None

    # Conditional form
    m = re.search(
        r"\b(?:raining|snowing|storming|rain|snow|storm)\s+in\s+([A-Za-z]{2,30})\b",
        text,
        re.IGNORECASE
    )
    if m:
        return {
            "name": "get_weather",
            "arguments": {"location": m.group(1)}
        }

    # Direct weather queries
    m = re.search(
        r"(?:"
        r"(?:what(?:'s| is)\s+the\s+weather(?:\s+like)?\s+in)|"
        r"(?:how'?s?\s+the\s+weather\s+in)|"
        r"(?:check\s+the\s+weather\s+in)|"
        r"(?:weather\s+(?:in|for|at))"
        r")\s+([A-Za-z][A-Za-z\s]{1,40})\b",
        text,
        re.IGNORECASE
    )

    if not m:
        return None

    return {
        "name": "get_weather",
        "arguments": {"location": m.group(1).strip()}
    }


def _extract_send_message(text):
    if not text:
        return None

    # Case 1: send a message to John saying hello / text John saying hello
    m = re.search(
        r"(?:send(?:\s+a)?\s+message\s+to|text|msg)\s+"
        r"([A-Za-z]+)\s+"
        r"(?:saying\s+)?(.+?)(?:\s*(?:and\b|\bthen\b|,|$))",
        text,
        re.IGNORECASE
    )
    if m:
        recipient = m.group(1).strip()
        message = m.group(2).strip().rstrip(".,!?")
        if recipient.lower() not in {"him", "her", "them", "a", "the"}:
            return {
                "name": "send_message",
                "arguments": {"recipient": recipient, "message": message}
            }

    # Case 2: send him/her a message saying hello (pronoun — resolved by caller)
    m = re.search(
        r"send\s+(him|her|them)\s+(?:a\s+)?message\s+"
        r"(?:saying\s+)?(.+?)(?:\s*(?:and\b|\bthen\b|,|$))",
        text,
        re.IGNORECASE
    )
    if m:
        return {
            "name": "send_message",
            "arguments": {
                "recipient": m.group(1).strip(),  # pronoun placeholder
                "message": m.group(2).strip().rstrip(".,!?")
            }
        }

    return None


def _extract_reminder(text):
    m = re.search(
        r"remind(?:\s+me)?\s+(?:to\s+|about\s+)?(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:am|pm)?)",
        text,
        re.IGNORECASE
    )

    if not m:
        return None

    title = m.group(1).strip()
    time_str = m.group(2).strip()

    # Remove leading articles
    title = re.sub(r"^(the|a|an)\s+", "", title, flags=re.IGNORECASE)

    # Remove leftover "to/about"
    title = re.sub(r"^(to|about)\s+", "", title, flags=re.IGNORECASE)

    if title and time_str:
        return {
            "name": "create_reminder",
            "arguments": {
                "title": title,
                "time": time_str
            }
        }

    return None


def _extract_play_music(text):
    if not text:
        return None

    m = re.search(
        r"(?:hey\s+|please\s+|could you\s+|can you\s+)?"
        r"play\s+(.+?)(?=\s+(?:and|then)\b|,|$)",
        text,
        re.IGNORECASE
    )

    if not m:
        return None

    song = m.group(1).strip().rstrip(".,!?")

    # Remove trailing politeness
    song = re.sub(r"\bplease\b", "", song, flags=re.IGNORECASE)

    # Remove leading "some"
    song = re.sub(r"^\s*some\s+", "", song, flags=re.IGNORECASE)

    song = re.sub(r"\s+", " ", song).strip()

    words = song.split()

    # Only strip "music" when original text had "some <word> music"
    if re.search(r"\bsome\s+\w+\s+music\b", text, re.IGNORECASE):
        if len(words) == 2 and words[1].lower() == "music":
            song = words[0]

    if not song:
        return None

    return {
        "name": "play_music",
        "arguments": {"song": song}
    }


def _extract_search_contacts(text):
    """
    Handles:
      - Find Bob in my contacts
      - Look up Chris
      - Search for Sarah
      - Find Jake
    """

    if not text:
        return None

    m = re.search(
        r"\b(?:find|look up|search(?:\s+for)?)\s+([A-Za-z]+)\b",
        text,
        re.IGNORECASE
    )

    if not m:
        return None

    query = m.group(1).strip()

    # Basic stopword filtering
    if query.lower() in {"a", "the", "my", "for", "me", "it"}:
        return None

    return {
        "name": "search_contacts",
        "arguments": {
            "query": query
        }
    }


_EXTRACTOR_MAP = {
    "set_alarm": _extract_alarm,
    "set_timer": _extract_timer,
    "get_weather": _extract_weather,
    "send_message": _extract_send_message,
    "create_reminder": _extract_reminder,
    "play_music": _extract_play_music,
    "search_contacts": _extract_search_contacts,
}
PRIORITY_ORDER = [
    "set_timer",
    "set_alarm",
    "create_reminder",
    "search_contacts",  
    "send_message",
    "get_weather",
    "play_music",
]


def fast_rule_based_parse(text, tools):
    """
    Deterministic parser. Pronoun resolution for search→send chains.
    Handles 1, 2, and 3-intent queries.
    """
    available = {t["name"] for t in tools}

    if "set_timer" in available:
        m = re.search(r"\bwake me up in\s+(\d+)", text, re.IGNORECASE)
        if m:
            return [{"name": "set_timer", "arguments": {"minutes": int(m.group(1))}}]

    hints = detect_tool_hints(text)
    if not hints:
        return None

    calls = []
    last_search_query = None
    used_tools = set()

    # Split into sub-phrases; pad with full text so every hint gets a shot
    raw_parts = split_intents_advanced(text) if len(hints) > 1 else [text]

    # Walk tools in priority order — try each sub-phrase, fall back to full text
    for tool_name in PRIORITY_ORDER:
        if tool_name not in available:
            continue
        extractor = _EXTRACTOR_MAP.get(tool_name)
        if not extractor or tool_name in used_tools:
            continue

        call = None
        # Try sub-phrases first (gives cleaner, focused text)
        for part in raw_parts:
            candidate = extractor(part)
            if candidate:
                call = candidate
                break

        # Fall back to full text (catches cases where split lost context)
        if not call:
            call = extractor(text)

        if not call:
            continue

        # Pronoun resolution: "him/her/them" → name from prior search_contacts
        if call["name"] == "send_message":
            recipient = call["arguments"].get("recipient", "").lower()
            if recipient in {"him", "her", "them"} and last_search_query:
                call["arguments"]["recipient"] = last_search_query

        if call["name"] == "search_contacts":
            last_search_query = call["arguments"].get("query")

        calls.append(call)
        used_tools.add(call["name"])

    return calls if calls and validate_required_arguments(calls, tools) else None


# ──────────────────────────────────────────────
# Multi-intent model strategies
# ──────────────────────────────────────────────

def handle_multi_intent_full_prompt(user_text, tools, expected_count):
    """Ask model to return ALL calls in one shot."""
    tool_names = ", ".join(t["name"] for t in tools)

    prompt = (
        f"The user is asking you to do {expected_count} separate things at once. "
        f"You MUST call exactly {expected_count} different functions. "
        f"Available functions: {tool_names}. "
        f"Do not skip any action. Fill all required arguments.\n\n"
        f"User: {user_text}"
    )

    result = cactus_complete(
        messages=[{"role": "user", "content": prompt}],
        tools=tools
    )

    calls = result.get("function_calls", [])
    coerce_argument_types(calls, tools)

    return calls, result.get("total_time_ms", 0)


def handle_multi_intent_split(user_text, tools):
    """Split on conjunctions and call model once per sub-phrase."""
    parts = split_intents_advanced(user_text)

    all_calls = []
    total_time = 0

    for part in parts:
        result = cactus_complete(
            messages=[{"role": "user", "content": part}],
            tools=tools
        )

        total_time += result.get("total_time_ms", 0)

        calls = result.get("function_calls", [])
        coerce_argument_types(calls, tools)

        if not calls or not validate_required_arguments(calls, tools):
            return None, total_time

        all_calls.extend(calls)

    return (all_calls if all_calls else None), total_time


def generate_hybrid(messages, tools, confidence_threshold=0.60):
    """
    Optimized hybrid:
    1. Try deterministic parser first (instant).
    2. If valid, return as on-device (no cactus call).
    3. Otherwise call cactus and use confidence gating.
    """

    user_text = messages[-1]["content"]
    fast_calls = fast_rule_based_parse(user_text, tools)
    local = generate_cactus(messages, tools)
    if fast_calls and validate_required_arguments(fast_calls, tools):
        local["function_calls"] = fast_calls
        local["confidence"] = max(local.get("confidence", 0), confidence_threshold + 0.01)
        local["source"] = "on-device"
        return local

    # ─────────────────────────────
    # Step 2: Cactus fallback
    # ─────────────────────────────
    
    total_time = local.get("total_time_ms", 0)
    calls = local.get("function_calls", [])
    coerce_argument_types(calls, tools)

    # Boost confidence if valid
    if calls and validate_required_arguments(calls, tools):
        local["confidence"] = max(local.get("confidence", 0), 0.90)

    # Confidence gating
    if local.get("confidence", 0) >= confidence_threshold:
        local["source"] = "on-device"
        return local

    # ─────────────────────────────
    # Step 3: Cloud fallback
    # ─────────────────────────────
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += total_time

    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
