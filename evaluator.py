import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from crewai import LLM




def _extract_budget_from_input(task_input: dict) -> Optional[float]:
    """Try to find a numeric budget cap in the task input fields."""
    
    if "budget" in task_input and task_input["budget"]:
        raw = str(task_input["budget"])
        nums = re.findall(r"[\d,]+(?:\.\d+)?", raw.replace(",", ""))
        if nums:
            return float(nums[0])

    
    combined = " ".join(str(v) for v in task_input.values())
    patterns = [
        r"budget\s*(?:of|is|:)?\s*\$?([\d,]+(?:\.\d+)?)",
        r"\$?([\d,]+(?:\.\d+)?)\s*budget",
        r"under\s*\$?([\d,]+(?:\.\d+)?)",
        r"max(?:imum)?\s*\$?([\d,]+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def _extract_totals_from_output(output_text: str) -> List[float]:
    """Extract dollar/currency amounts near 'total', 'budget', 'cost'."""
    totals = []
    
    for line in output_text.split("\n"):
        lower = line.lower()
        if any(kw in lower for kw in ["total", "budget", "overall cost", "grand total", "estimated cost"]):
            amounts = re.findall(
                r"[\$€£]?\s*([\d,]+(?:\.\d{1,2})?)\s*(?:usd|eur|gbp|dollars?|euros?)?",
                line, re.IGNORECASE,
            )
            for a in amounts:
                val = float(a.replace(",", ""))
                if val > 10:  
                    totals.append(val)
    return totals


def check_budget(task_input: dict, output_text: str) -> dict:
   
    budget_cap = _extract_budget_from_input(task_input)
    if budget_cap is None:
        return {"result": "not_applicable", "reason": "No budget specified in input."}

    totals = _extract_totals_from_output(output_text)
    if not totals:
        return {"result": "not_applicable", "reason": "No cost total found in output to compare."}

    max_total = max(totals)
    if max_total > budget_cap:
        return {
            "result": False,
            "reason": f"Output total ${max_total:.0f} exceeds budget cap ${budget_cap:.0f}.",
        }
    return {"result": True, "reason": ""}


def _parse_trip_length(task_input: dict) -> Optional[int]:
    
    date_range = str(task_input.get("date_range", ""))

    
    m = re.search(r"(\d{1,2})\s*[-–]\s*(\d{1,2})", date_range)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if end > start:
            return end - start + 1

    
    m = re.search(r"(\d+)\s*[-]?\s*days?", date_range, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def _count_days_in_output(output_text: str) -> int:
    
    day_nums = set()

    
    for m in re.finditer(r"\bday\s*(\d+)\b", output_text, re.IGNORECASE):
        day_nums.add(int(m.group(1)))

    if day_nums:
        return len(day_nums)

    
    months = (
        r"(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    )
    
    for m in re.finditer(months + r"\s+(\d{1,2})\b", output_text, re.IGNORECASE):
        day_nums.add(int(m.group(1)))
    
    for m in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + months, output_text, re.IGNORECASE
    ):
        day_nums.add(int(m.group(1)))

    if day_nums:
        return len(day_nums)

    
    weekdays = set()
    for m in re.finditer(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        output_text, re.IGNORECASE,
    ):
        weekdays.add(m.group(1).lower())

    if weekdays:
        return len(weekdays)

    
    ordinals = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    for word, num in ordinals.items():
        if re.search(rf"\b{word}\s+day\b", output_text, re.IGNORECASE):
            day_nums.add(num)
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)\s+day\b", output_text, re.IGNORECASE):
        day_nums.add(int(m.group(1)))

    if day_nums:
        return len(day_nums)

    
    heading_count = 0
    for m in re.finditer(
        r"^(?:#{1,4}\s+.+|(?:\*\*).+(?:\*\*))$", output_text, re.MULTILINE
    ):
        heading_text = m.group().lower()
        if any(
            kw in heading_text
            for kw in ["morning", "afternoon", "evening", "arrival", "departure", "itinerary"]
        ):
            heading_count += 1

    
    if heading_count >= 2:
        return max(1, heading_count // 3) or heading_count

    return 0


def check_days(task_input: dict, output_text: str) -> dict:
    
    expected = _parse_trip_length(task_input)
    if expected is None:
        return {"result": "not_applicable", "reason": "Could not parse trip length from input."}

    actual = _count_days_in_output(output_text)
    if actual == 0:
        return {"result": False, "reason": "No day markers found in output (tried Day N, dates, weekdays, ordinals)."}

    
    if abs(actual - expected) <= 1:
        return {"result": True, "reason": ""}

    return {
        "result": False,
        "reason": f"Expected {expected} days (±1 tolerance), found {actual} days in output.",
    }



def _split_into_day_sections(output_text: str) -> Dict[str, str]:
    
    sections = {}
    
    parts = re.split(r"(?i)\b(day\s*\d+)\b", output_text)
    current_day = None
    for part in parts:
        if re.match(r"(?i)day\s*\d+", part.strip()):
            current_day = part.strip().lower()
            sections[current_day] = ""
        elif current_day:
            sections[current_day] += part
    return sections


def _extract_venue_names(text: str) -> List[str]:
    
    
    TIME_PATTERN = re.compile(r'^\d{1,2}[:.]\d{2}')  
    GENERIC_PATTERN = re.compile(
        r'^(weather forecast|packing suggestion|budget breakdown|'
        r'morning|afternoon|evening|night|total|high temp|low temp|'
        r'chance of rain|items to pack|clothing|weather condition|'
        r'additional tip|practical tip|cultural insight|hidden gem|'
        r'key attraction|travel cost|budget estimate|packing list|'
        r'local transport|public transportation|accommodation|'
        r'meals out|attractions and activities|day trip|departure|'
        r'arrival|check.?in|check.?out|summary|conclusion|overview|'
        r'introduction|final answer|temperature range|flight cost|'
        r'round.?trip flight|overall description|average daily|'
        r'hotel|hostel|airbnb|resort|guest.?house|'  
        r'transport|metro|bus|taxi|uber|lyft)',
        re.IGNORECASE
    )

    venues = []

    
    for m in re.finditer(
        r'(?:visit|go to|explore|dine at|eat at|lunch at|dinner at|'
        r'breakfast at|see|tour|check out|head to|stop at|try)\s+'
        r'(?:the\s+)?(?:\*+)?([A-ZÀ-Ú][a-zA-ZÀ-ú\u00e9\u00e8\u00ea\u00eb'
        r'\u00e0\u00e2\u00e4\u00f9\u00fb\u00fc\u00f4\u00f6\u00ee\u00ef'
        r'\u00e7\'-]+(?:\s+[A-Za-zÀ-ú\u00e9\u00e8\u00ea\u00eb\u00e0\u00e2'
        r'\u00e4\u00f9\u00fb\u00fc\u00f4\u00f6\u00ee\u00ef\u00e7\'-]+)*)',
        text, re.IGNORECASE
    ):
        name = m.group(1).strip().rstrip(':.,;')
        if (len(name) > 4
            and not TIME_PATTERN.match(name)
            and not GENERIC_PATTERN.match(name)):
            venues.append(name.lower())

    
    for m in re.finditer(r'\*([^*]{5,60})\*', text):
        name = m.group(1).strip().rstrip(':.,;')
        if (not TIME_PATTERN.match(name)
            and not GENERIC_PATTERN.match(name)
            and not name[0].isdigit()):
            venues.append(name.lower())

    
    for m in re.finditer(r'\*\*([^*]+)\*\*', text):
        name = m.group(1).strip().rstrip(':.,;')
        name_lower = name.lower()
    
        has_caps = sum(1 for w in name.split() if w[0:1].isupper()) >= 2
        if (len(name) > 4
            and has_caps
            and not TIME_PATTERN.match(name)
            and not GENERIC_PATTERN.match(name)
            and not name_lower.startswith((
                'day', 'morning', 'afternoon', 'evening', 'night',
                'budget', 'total', 'packing', 'weather', 'high',
                'low', 'chance', 'clothing', 'items', 'transport',
                'flights', 'accommodation', 'lunch', 'dinner',
                'breakfast', 'tip', 'hidden', 'key', 'travel',
                'hotel', 'hostel', 'airbnb', 'resort',
            ))):
            venues.append(name_lower)

    return venues


def check_no_duplicates(task_input: dict, output_text: str) -> dict:
   
    sections = _split_into_day_sections(output_text)
    if len(sections) < 2:
        return {"result": "not_applicable", "reason": "Could not split output into day sections."}

    
    venue_days = defaultdict(set)
    for day_label, text in sections.items():
        for venue in _extract_venue_names(text):
            venue_days[venue].add(day_label)

   
    duplicates = {v: sorted(days) for v, days in venue_days.items()
                  if len(days) >= 3}

    if duplicates:
        dup_list = "; ".join(f"'{v}' in {d}" for v, d in list(duplicates.items())[:3])
        return {
            "result": False,
            "reason": f"Duplicate venues across days: {dup_list}",
        }
    return {"result": True, "reason": ""}


def run_heuristic_checks(task_input: dict, output_text: str) -> dict:
   
    budget = check_budget(task_input, output_text)
    days = check_days(task_input, output_text)
    dupes = check_no_duplicates(task_input, output_text)
    return {
        "budget_ok": budget["result"],
        "budget_reason": budget["reason"],
        "days_ok": days["result"],
        "days_reason": days["reason"],
        "no_duplicates": dupes["result"],
        "no_duplicates_reason": dupes["reason"],
    }




# Judge LLM — intentionally different from agent model to avoid self-grading
_judge_llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)


def _parse_judge_json(raw: str) -> Optional[dict]:
    
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        
        m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None


def run_llm_judge(task_input: dict, output_text: str) -> dict:
    """Run the LLM judge on the output. Returns dict of check results."""
    input_summary = (
        f"Origin: {task_input.get('origin', 'N/A')}\n"
        f"Cities: {task_input.get('cities', 'N/A')}\n"
        f"Date Range: {task_input.get('date_range', 'N/A')}\n"
        f"Interests: {task_input.get('interests', 'N/A')}\n"
    )
    if task_input.get("constraints"):
        input_summary += f"Constraints: {task_input['constraints']}\n"

    # Truncate output to fit context window
    truncated_output = output_text[:6000]

    prompt = (
        "You are a strict travel-plan evaluator. Given the TASK INPUT and "
        "the PLAN OUTPUT below, evaluate three criteria. "
        "Respond ONLY with valid JSON, no other text, no markdown fences.\n\n"
        "Criteria:\n"
        "1. constraints_ok (bool): Were all stated constraints honored? "
        "(dietary needs, mobility needs, must-see places, correct date range, "
        "correct origin city). If no special constraints were stated, check "
        "the plan matches the requested cities, dates, and interests.\n"
        "2. feasible (bool): Is the day-to-day plan realistic? No impossible "
        "same-day cross-country travel, no 20-hour sightseeing days, "
        "reasonable timing between activities.\n"
        "3. no_hallucination (bool): Do the named venues, restaurants, and "
        "hotels sound real and plausible for the stated city? Not obviously "
        "fabricated or nonsensical names.\n\n"
        "Return JSON:\n"
        '{"constraints_ok": true/false, "feasible": true/false, '
        '"no_hallucination": true/false, "notes": "brief explanation for '
        'any false verdict"}\n\n'
        f"=== TASK INPUT ===\n{input_summary}\n"
        f"=== PLAN OUTPUT ===\n{truncated_output}\n\n"
        "Your JSON verdict:"
    )

    try:
        raw_response = _judge_llm.call(prompt)
    except Exception as e:
        print(f"[evaluator] LLM judge call failed: {e}")
        return {
            "constraints_ok": None,
            "feasible": None,
            "no_hallucination": None,
            "judge_notes": f"LLM call failed: {e}",
            "judge_raw": "",
        }

    parsed = _parse_judge_json(raw_response)

    if parsed is None:
        print(f"[evaluator] WARNING: Could not parse judge JSON. Raw response:\n{raw_response[:500]}")
        return {
            "constraints_ok": None,
            "feasible": None,
            "no_hallucination": None,
            "judge_notes": "JSON parse failure",
            "judge_raw": raw_response[:500],
        }

    return {
        "constraints_ok": parsed.get("constraints_ok"),
        "feasible": parsed.get("feasible"),
        "no_hallucination": parsed.get("no_hallucination"),
        "judge_notes": parsed.get("notes", ""),
        "judge_raw": "",
    }



#  Combined Evaluator

def evaluate(task_input: dict, output_text: str) -> dict:
    
    heuristic_results = run_heuristic_checks(task_input, output_text)
    judge_results = run_llm_judge(task_input, output_text)

    
    check_keys = ["budget_ok", "days_ok", "no_duplicates",
                   "constraints_ok", "feasible", "no_hallucination"]

    all_checks = {**heuristic_results, **judge_results}

    failure_reasons = []
    checks_passed = 0
    checks_applicable = 0

    for k in check_keys:
        val = all_checks.get(k)
        
        if val == "not_applicable" or val is None:
            continue
        checks_applicable += 1
        if val is True:
            checks_passed += 1
        elif val is False:
            reason = all_checks.get(f"{k}_reason", "")
            failure_reasons.append(f"{k}: {reason}" if reason else k)

    overall_pass = (checks_applicable > 0 and checks_passed == checks_applicable)
    accuracy = (checks_passed / checks_applicable) if checks_applicable > 0 else 0.0

    return {
        "overall_pass": overall_pass,
        "accuracy": round(accuracy, 4),
        "checks_passed": checks_passed,
        "checks_applicable": checks_applicable,
        "failure_reasons": failure_reasons,
        **all_checks,
    }
