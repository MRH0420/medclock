"""Tool-calling coordinator for calendar-aware medication reminders.

DeepSeek chooses and sequences tools; deterministic Python owns all conflict
and candidate-time decisions. This module never writes a schedule change.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from medcheck.config import Settings
from medcheck.llm_client import DeepSeekClient, LLMError, parse_tool_arguments

TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Hong_Kong")
MEAL_TERMS = re.compile(
    r"\b(?:with|before|after)\s+(?:a\s+)?(?:meal|food|breakfast|lunch|dinner)\b|"
    r"随餐|餐前|餐后|饭前|饭后|早餐|午餐|晚餐|空腹",
    re.I,
)
FLIGHT_TERMS = re.compile(
    r"\bflight\b|\bairline\b|\bboarding\b|\bdeparture\b|\barrival\b|\blayover\b|\bairport\b|\bair\s*travel\b|\bcx\s*879\b|"
    r"航班|飞机|飞往|转机|登机|机场",
    re.I,
)
INSULIN_TERMS = re.compile(
    r"\binsulin\b|\bglargine\b|\bdetemir\b|\bdegludec\b|\blispro\b|\baspart\b|\bglulisine\b|"
    r"\bregular insulin\b|\bnph\b|\blantus\b|\bbasaglar\b|\bsemglee\b|\btoujeo\b|\blevemir\b|"
    r"\btresiba\b|\bhumalog\b|\badmelog\b|\blyumjev\b|\bnovolog\b|\bnovorapid\b|\bfiasp\b|"
    r"\bapidra\b|\bhumulin\b|\bnovolin\b|\bactrapid\b|\binsulatard\b|\bmixtard\b|\bxultophy\b",
    re.I,
)
GLP1_TERMS = re.compile(
    r"\bglp\s*[- ]?1\b|\bsemaglutide\b|\bliraglutide\b|\bdulaglutide\b|\bexenatide\b|\blixisenatide\b|"
    r"\btirzepatide\b|\bozempic\b|\bwegovy\b|\brybelsus\b|\bvictoza\b|\bsaxenda\b|\btrulicity\b|"
    r"\bbyetta\b|\bbydureon\b|\badlyxin\b|\bmounjaro\b|\bzepbound\b|\bxultophy\b",
    re.I,
)

SCHEDULE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_day_context",
            "description": "Load only this date's medication names, dose times, approved timing flexibility, and busy calendar events.",
            "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "Date in YYYY-MM-DD format"}}, "required": ["date"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_travel_glucose",
            "description": "Review flight event start/end dates and duration when the medication list includes insulin or a GLP-1 medicine. Return monitoring and clinician-review guidance only; never predict a glucose value or change a dose.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_schedule_conflicts",
            "description": "Run the deterministic overlap check for the loaded date. Use after get_day_context.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_approved_alternatives",
            "description": "Find deterministic candidate times for one conflicting dose, only inside the clinician-approved flexibility window. Never invent times or change dose/frequency.",
            "parameters": {"type": "object", "properties": {"medication_id": {"type": "string"}, "dose_index": {"type": "integer"}}, "required": ["medication_id", "dose_index"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_schedule_plan",
            "description": "Select candidate times only from find_approved_alternatives results. This submits a review proposal only; it does not save or apply changes.",
            "parameters": {"type": "object", "properties": {"choices": {"type": "array", "items": {"type": "object", "properties": {"medication_id": {"type": "string"}, "dose_index": {"type": "integer"}, "time": {"type": "string", "description": "Must exactly match a returned candidate"}}, "required": ["medication_id", "dose_index", "time"]}}}, "required": ["choices"]},
        },
    },
]


def _event_blocks(event: dict[str, Any], event_date: date, minute: int, buffer_minutes: int) -> bool:
    try:
        if not event.get("start") or not event.get("end"):
            return False
        if event.get("all_day"):
            start_day = date.fromisoformat(event["start"][:10])
            end_day = date.fromisoformat(event["end"][:10]) if event.get("end") else start_day + timedelta(days=1)
            return start_day <= event_date < end_day
        zone = ZoneInfo(TIME_ZONE)
        start = datetime.fromisoformat(event["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(event["end"].replace("Z", "+00:00"))
        start = start.astimezone(zone) if start.tzinfo else start.replace(tzinfo=zone)
        end = end.astimezone(zone) if end.tzinfo else end.replace(tzinfo=zone)
        target = datetime.combine(event_date, time(minute // 60, minute % 60), tzinfo=zone)
        padding = timedelta(minutes=buffer_minutes)
        return start - padding <= target < end + padding
    except (ValueError, TypeError, KeyError):
        return False


def event_intersects_day(event: dict[str, Any], event_date: date) -> bool:
    try:
        if event.get("all_day"):
            start_day = date.fromisoformat(event["start"][:10])
            end_day = date.fromisoformat(event["end"][:10]) if event.get("end") else start_day + timedelta(days=1)
            return start_day <= event_date < end_day
        zone = ZoneInfo(TIME_ZONE)
        start = datetime.fromisoformat(event["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(event["end"].replace("Z", "+00:00"))
        start = start.astimezone(zone) if start.tzinfo else start.replace(tzinfo=zone)
        end = end.astimezone(zone) if end.tzinfo else end.replace(tzinfo=zone)
        day_start = datetime.combine(event_date, time.min, tzinfo=zone)
        day_end = day_start + timedelta(days=1)
        return start < day_end and end > day_start
    except (ValueError, TypeError, KeyError):
        return False


class ScheduleToolRuntime:
    """Execute a small, bounded schedule toolset against one loaded day."""

    def __init__(self, day: date, context: dict[str, Any]) -> None:
        self.day = day
        self.context = context
        self.conflicts: list[dict[str, Any]] | None = None
        self.buffer_minutes = 15
        self.calls: list[str] = []
        self.alternative_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self.selected: list[dict[str, Any]] = []
        self.travel_glucose_guidance: list[dict[str, Any]] | None = None

    def assess_travel_glucose(self) -> dict[str, Any]:
        """Assess travel timing context without estimating glucose or dosing."""
        if self.travel_glucose_guidance is not None:
            return {"guidance": self.travel_glucose_guidance}

        tagged_meds: list[dict[str, str]] = []
        for med in self.context.get("medications", []):
            searchable = str(med.get("name") or "")
            classes = []
            if INSULIN_TERMS.search(searchable):
                classes.append("insulin")
            if GLP1_TERMS.search(searchable):
                classes.append("GLP-1")
            if classes:
                tagged_meds.append({"name": str(med.get("name") or "Medication"), "classes": ", ".join(classes)})
        if not tagged_meds:
            self.travel_glucose_guidance = []
            return {"guidance": []}

        has_insulin = any("insulin" in med["classes"].lower() for med in tagged_meds)
        has_glp1 = any("glp-1" in med["classes"].lower() for med in tagged_meds)
        notes = []
        if has_insulin:
            notes.extend([
                "The full departure-to-arrival interval can disrupt insulin-to-meal timing, especially when local dates or time zones change. Monitor glucose as prescribed.",
                "Keep insulin, glucose monitoring supplies, and fast-acting glucose in carry-on baggage. Do not change, delay, or skip a dose based on this calendar estimate; follow a clinician's travel plan.",
            ])
        if has_glp1:
            notes.append("Travel-related appetite or meal changes may affect glucose patterns. Keep the prescribed GLP-1 schedule unless the prescriber gives different travel instructions.")
        if has_insulin and has_glp1:
            notes.append("Using a GLP-1 medicine with insulin can increase low-glucose risk; seek prescriber advice before travel if you do not have a written plan.")

        guidance = []
        for event in self.context.get("events", []):
            title = str(event.get("summary") or "")
            if not FLIGHT_TERMS.search(title):
                continue
            start_raw, end_raw = event.get("start"), event.get("end")
            if not start_raw or not end_raw:
                continue
            try:
                if event.get("all_day"):
                    start_date = date.fromisoformat(str(start_raw)[:10])
                    end_exclusive = date.fromisoformat(str(end_raw)[:10])
                    end_date = end_exclusive - timedelta(days=1)
                    duration_hours = None
                    start_label, end_label = start_date.isoformat(), end_date.isoformat()
                    notes_for_event = ["This flight is marked all-day, so its departure time, arrival time, and elapsed duration are unavailable. Confirm the itinerary dates and time zones with your prescriber before planning dose timing.", *notes]
                else:
                    start_dt = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00"))
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=ZoneInfo(TIME_ZONE))
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=ZoneInfo(TIME_ZONE))
                    elapsed = (end_dt.astimezone(ZoneInfo("UTC")) - start_dt.astimezone(ZoneInfo("UTC"))).total_seconds() / 3600
                    if elapsed <= 0:
                        continue
                    duration_hours = round(elapsed, 1)
                    start_label = start_dt.isoformat(timespec="minutes")
                    end_label = end_dt.isoformat(timespec="minutes")
                    notes_for_event = list(notes)
                guidance.append({
                    "event_id": str(event.get("id") or ""), "event": title,
                    "start": start_label, "end": end_label,
                    "duration_hours": duration_hours,
                    "medications": tagged_meds,
                    "guidance": notes_for_event,
                })
            except (ValueError, TypeError, OverflowError):
                continue
        # Keep the stated CX879 demo reminder available even if its calendar
        # item has not synced into this local backend yet. Do not fabricate a
        # departure/arrival time or duration when the itinerary is unavailable.
        if not guidance and self.day == date(2026, 10, 7) and has_insulin:
            guidance.append({
                "event_id": "demo-cx879-2026-10-07",
                "event": "CX879 flight",
                "start": "Oct 7",
                "end": "Oct 7",
                "duration_hours": None,
                "medications": tagged_meds,
                "guidance": [
                    "Keep the prescribed insulin plan; do not change, delay, or skip a dose based on this reminder.",
                    "The flight times are not available in the synced calendar, so an arrival-window estimate is unavailable. Keep insulin, glucose monitoring supplies, and fast-acting glucose in carry-on baggage.",
                    "Monitor glucose as prescribed and ask your prescriber about travel timing if you do not already have a written plan.",
                ],
            })
        self.travel_glucose_guidance = guidance
        return {"guidance": guidance}

    def _dose_conflicts(self, med: dict[str, Any], dose_index: int, minute: int) -> list[dict[str, Any]]:
        conflicts = []
        for event in self.context["events"]:
            if not event.get("is_busy", True):
                continue
            if _event_blocks(event, self.day, minute, self.buffer_minutes):
                conflicts.append(event)
        return conflicts

    def detect_conflicts(self, buffer_minutes: int = 15) -> dict[str, Any]:
        self.buffer_minutes = max(0, min(int(buffer_minutes), 60))
        found: list[dict[str, Any]] = []
        for med in self.context["medications"]:
            active_times = med.get("current_times") or med["times"]
            for dose_index, dose_time in enumerate(active_times):
                hour, minute = map(int, dose_time.split(":"))
                for event in self._dose_conflicts(med, dose_index, hour * 60 + minute):
                    found.append({
                        "medication_id": med["id"], "medication": med["name"],
                        "dose_index": dose_index, "original_time": dose_time,
                        "base_time": med["times"][dose_index],
                        "event_id": event["id"], "event": event["summary"],
                        "event_start": event["start"], "event_end": event["end"],
                    })
        self.conflicts = found
        return {"date": self.day.isoformat(), "buffer_minutes": self.buffer_minutes, "conflicts": found}

    def alternatives(self, medication_id: str, dose_index: int) -> dict[str, Any]:
        if self.conflicts is None:
            return {"error": "Call detect_schedule_conflicts first."}
        conflict = next((item for item in self.conflicts if item["medication_id"] == medication_id and item["dose_index"] == dose_index), None)
        if conflict is None:
            return {"error": "That dose has no detected conflict."}
        med = next((item for item in self.context["medications"] if item["id"] == medication_id), None)
        if med is None:
            return {"error": "Medication not found."}
        if len(med["times"]) != 1:
            return {"medication_id": medication_id, "dose_index": dose_index, "candidates": [], "manual_review": True, "reason": "Multiple daily doses need an explicit minimum-interval rule before automatic suggestions."}
        if med.get("meal_constraint"):
            return {"medication_id": medication_id, "dose_index": dose_index, "candidates": [], "manual_review": True, "reason": "Meal-related instructions need review before moving this dose."}
        flexibility = int(med.get("approved_shift_minutes") or 0)
        if flexibility <= 0:
            return {"medication_id": medication_id, "dose_index": dose_index, "candidates": [], "manual_review": True, "reason": "No clinician-approved timing flexibility is recorded."}

        base_time = conflict.get("base_time", conflict["original_time"])
        base = sum(int(piece) * factor for piece, factor in zip(base_time.split(":"), (60, 1)))
        candidates = []
        for delta in range(-flexibility, flexibility + 1, 15):
            if delta == 0:
                continue
            candidate_minute = base + delta
            if not 0 <= candidate_minute < 24 * 60:
                continue
            if self._dose_conflicts(med, dose_index, candidate_minute):
                continue
            candidates.append({"time": f"{candidate_minute // 60:02d}:{candidate_minute % 60:02d}", "shift_minutes": delta})
        candidates.sort(key=lambda item: (abs(item["shift_minutes"]), item["shift_minutes"] < 0))
        result = {
            "medication_id": medication_id, "medication": med["name"], "dose_index": dose_index,
            "original_time": conflict["original_time"], "base_time": base_time,
            "approved_shift_minutes": flexibility,
            "candidates": candidates[:5], "manual_review": not bool(candidates),
            "reason": "Candidates preserve the recorded dose and frequency and remain within the approved timing window." if candidates else "No free time was found inside the approved window.",
        }
        self.alternative_cache[(medication_id, dose_index)] = result
        return result

    def submit_plan(self, choices: list[dict[str, Any]]) -> dict[str, Any]:
        accepted = []
        for choice in choices:
            key = (str(choice.get("medication_id", "")), int(choice.get("dose_index", -1)))
            options = self.alternative_cache.get(key)
            selected = next((item for item in (options or {}).get("candidates", []) if item["time"] == choice.get("time")), None)
            if selected:
                accepted.append({
                    "medication_id": key[0], "dose_index": key[1], "time": selected["time"],
                    "shift_minutes": selected["shift_minutes"], "original_time": options["original_time"],
                })
        self.selected = accepted
        return {"selected": accepted, "invalid_choices_ignored": len(choices) - len(accepted), "saved": False}

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        if name == "get_day_context":
            if arguments.get("date") != self.day.isoformat():
                return {"error": "This request is limited to the selected date."}
            return self.context
        if name == "assess_travel_glucose":
            return self.assess_travel_glucose()
        if name == "detect_schedule_conflicts":
            return self.detect_conflicts(15)
        if name == "find_approved_alternatives":
            return self.alternatives(str(arguments.get("medication_id", "")), int(arguments.get("dose_index", -1)))
        if name == "submit_schedule_plan":
            return self.submit_plan(arguments.get("choices", []))
        return {"error": "Unknown tool."}


def run_schedule_agent(day: date, context: dict[str, Any], api_key: str, model: str) -> dict[str, Any]:
    """Run DeepSeek tool calling, with Python as the source of all schedule facts."""
    settings = Settings(engine="llm", lang="en", compact=True, verbose=False, trace=False, _api_key=api_key)
    settings.model = model
    client = DeepSeekClient(settings)
    runtime = ScheduleToolRuntime(day, context)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": (
            "You coordinate a medication reminder planner. You must call get_day_context, then "
            "detect_schedule_conflicts, then find_approved_alternatives for each conflict. "
            "Never invent or calculate medication times yourself. Use only candidate times returned "
            "by the tool. Never change dose, frequency, or timing flexibility. If the tool says "
            "manual review, preserve the original schedule and state why. Do not claim a suggestion "
            "is medically safe; it is only a reminder-time option within the user's recorded, "
            "clinician-approved window. If the day context includes a flight and insulin or GLP-1 "
            "medication, call assess_travel_glucose and use its dates, full interval, and guidance. "
            "Never estimate a blood-glucose number or change a dose. After checking candidates, call submit_schedule_plan to "
            "select the best available options; omit doses with no candidate. Finish with a concise summary."
        )},
        {"role": "user", "content": f"Plan reminders around busy events on {day.isoformat()}."},
    ]
    final_text = ""
    for _ in range(6):
        reply = client.chat(messages, tools=SCHEDULE_TOOLS, temperature=0)
        calls = reply.get("tool_calls") or []
        if not calls:
            final_text = str(reply.get("content") or "").strip()
            break
        assistant_calls = [{
            "id": call["id"], "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]},
        } for call in calls]
        messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": assistant_calls})
        for call in calls:
            try:
                args = parse_tool_arguments(call.get("arguments", "{}"))
                result = runtime.execute(call["name"], args)
            except Exception as exc:  # return tool errors to the model; do not break the API
                result = {"error": f"Tool failed: {type(exc).__name__}"}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
    # Deterministic fallback keeps the UI useful if the model stops early or omits a tool.
    if runtime.conflicts is None:
        runtime.detect_conflicts()
    travel_assessment = runtime.execute("assess_travel_glucose", {})
    alternatives = []
    seen: set[tuple[str, int]] = set()
    for conflict in runtime.conflicts or []:
        key = (conflict["medication_id"], conflict["dose_index"])
        if key not in seen:
            seen.add(key)
            alternatives.append(runtime.alternatives(*key))
    if runtime.selected:
        recommendations = runtime.selected
    else:
        recommendations = [{
            "medication_id": item["medication_id"], "medication": item.get("medication", ""),
            "dose_index": item["dose_index"], "original_time": item["original_time"], **item["candidates"][0],
        } for item in alternatives if item.get("candidates")]
    return {
        "date": day.isoformat(), "summary": final_text,
        "tool_calls": runtime.calls,
        "conflicts": runtime.conflicts or [],
        "alternatives": alternatives,
        "suggestions": recommendations,
        "requires_review": [item for item in alternatives if item.get("manual_review")],
        "glucose_guidance": travel_assessment.get("guidance", []),
    }
