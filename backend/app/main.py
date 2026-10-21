from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sqlite3
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlencode, urlparse

from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google.oauth2.id_token import verify_oauth2_token
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel, Field, model_validator

from medcheck.agent import MedCheckAgent
from medcheck.config import Settings as MedCheckSettings
from medcheck.models import Verdict
from medcheck.screener import brief_sheet
from .schedule_agent import MEAL_TERMS, ScheduleToolRuntime, event_intersects_day, run_schedule_agent


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/calendar.events.owned.readonly"]
CALENDAR_READ_SCOPE = "https://www.googleapis.com/auth/calendar.events.owned.readonly"
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback").strip()
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173").strip().rstrip("/")
CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Hong_Kong")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
DB_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT / ".data" / "medclock.sqlite3")))
PERSONA_PATH = DB_PATH.parent / "persona.txt"
LEGACY_PERSONA_PATH = DB_PATH.parent / "persona.json"
try:
    FERNET = Fernet(os.environ["TOKEN_ENCRYPTION_KEY"].encode())
except (KeyError, ValueError):
    FERNET = None

_oauth_exchange_lock = Lock()

_SAFETY_FIELD_LABELS_EN = {
    "boxed_warning": "Boxed warning",
    "warnings": "Warnings",
    "warnings_and_cautions": "Warnings and precautions",
    "contraindications": "Contraindications",
    "do_not_use": "Do not use",
    "drug_interactions": "Drug interactions",
    "precautions": "Precautions",
    "adverse_reactions": "Adverse reactions",
    "overdosage": "Overdose",
    "pregnancy": "Pregnancy",
    "pregnancy_or_breast_feeding": "Pregnancy and breastfeeding",
    "nursing_mothers": "Nursing mothers",
    "pediatric_use": "Pediatric use",
    "geriatric_use": "Geriatric use",
    "use_in_specific_populations": "Use in specific populations",
    "active_ingredient": "Active ingredient",
    "dosage_and_administration": "Dosage and administration",
    "information_for_patients": "Patient information",
    "ask_doctor": "Ask a doctor",
    "ask_doctor_or_pharmacist": "Ask a doctor or pharmacist",
    "stop_use": "Stop use",
    "when_using": "When using",
    "purpose": "Purpose",
    "storage_and_handling": "Storage and handling",
}


def safety_field_label_en(field_name: str) -> str:
    return _SAFETY_FIELD_LABELS_EN.get(field_name, field_name.replace("_", " ").capitalize())


def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    with connect_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS medications (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, dosage TEXT NOT NULL,
                time TEXT NOT NULL, instructions TEXT NOT NULL, tone TEXT NOT NULL DEFAULT 'violet',
                times_json TEXT NOT NULL DEFAULT '[]',
                category TEXT NOT NULL DEFAULT 'Prescription',
                indication TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS oauth_state (
                state TEXT PRIMARY KEY, expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS google_token (
                id INTEGER PRIMARY KEY CHECK (id = 1), token BLOB NOT NULL, email TEXT
            );
            CREATE TABLE IF NOT EXISTS calendar_sync (
                event_id TEXT PRIMARY KEY, summary TEXT NOT NULL, start TEXT NOT NULL,
                end TEXT, html_link TEXT, updated_at TEXT,
                all_day INTEGER NOT NULL DEFAULT 0, is_busy INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS schedule_override (
                date TEXT NOT NULL, medication_id TEXT NOT NULL, dose_index INTEGER NOT NULL,
                original_time TEXT NOT NULL, planned_time TEXT NOT NULL,
                event_ids_json TEXT NOT NULL DEFAULT '[]', accepted_at TEXT NOT NULL,
                PRIMARY KEY(date, medication_id, dose_index)
            );
            CREATE TABLE IF NOT EXISTS personal_information (
                id INTEGER PRIMARY KEY CHECK (id = 1), profile BLOB NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        columns = {row["name"] for row in db.execute("PRAGMA table_info(medications)")}
        for name, declaration in (
            ("times_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("category", "TEXT NOT NULL DEFAULT 'Prescription'"),
            ("indication", "TEXT NOT NULL DEFAULT ''"),
            ("approved_shift_minutes", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in columns:
                db.execute(f"ALTER TABLE medications ADD COLUMN {name} {declaration}")
        calendar_columns = {row["name"] for row in db.execute("PRAGMA table_info(calendar_sync)")}
        for name, declaration in (("all_day", "INTEGER NOT NULL DEFAULT 0"), ("is_busy", "INTEGER NOT NULL DEFAULT 1")):
            if name not in calendar_columns:
                db.execute(f"ALTER TABLE calendar_sync ADD COLUMN {name} {declaration}")


app = FastAPI(title="MedClock API", version="0.1.0")
logger = logging.getLogger("medclock.oauth")
_last_oauth_failure: dict[str, Any] | None = None
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup() -> None:
    init_db()
    # Remove the old locally assembled JSON persona; only opt-in DeepSeek text
    # memories are supported by the current flow.
    LEGACY_PERSONA_PATH.unlink(missing_ok=True)


class MedicationIn(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=160)
    dosage: str = Field(default="1 dose", max_length=160)
    time: str = Field(pattern=r"^\d{2}:\d{2}$")
    times: list[str] = Field(default_factory=list, max_length=8)
    category: Literal["Prescription", "OTC", "Supplement", "Herbal/Traditional", "Other"] = "Prescription"
    indication: str = Field(default="", max_length=500)
    approved_shift_minutes: int = Field(default=0, ge=0, le=120)
    instructions: str = Field(default="Follow label instructions", max_length=500)
    tone: str = "violet"

    @model_validator(mode="after")
    def normalize_daily_times(self) -> "MedicationIn":
        times = self.times or [self.time]
        if any(not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", item) for item in times):
            raise ValueError("Each dose time must use HH:MM in 24-hour format")
        self.times = sorted(set(times))
        if not self.times:
            raise ValueError("At least one daily dose time is required")
        self.time = self.times[0]
        return self


class MedicationRiskAssessmentIn(BaseModel):
    medications: list[MedicationIn] = Field(min_length=1, max_length=20)


class SchedulePlanIn(BaseModel):
    date: date
    consent: bool = False


class ScheduleOverrideItem(BaseModel):
    medication_id: str
    dose_index: int = Field(ge=0, le=7)
    original_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    planned_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class ScheduleAcceptIn(BaseModel):
    date: date
    suggestions: list[ScheduleOverrideItem] = Field(min_length=1, max_length=40)


class PersonalInformationIn(BaseModel):
    birth_year: int = Field(ge=1900, le=2100)
    sex: Literal["男", "女"]
    pregnancy: Literal["妊娠期", "哺乳期", "否"] | None = None
    breakfast: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    lunch: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    dinner: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    wake: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    sleep: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    medical_history: str = Field(default="", max_length=1200)
    medication_limits: str = Field(default="", max_length=1200)

    @model_validator(mode="after")
    def constrain_personal_answers(self) -> "PersonalInformationIn":
        if self.birth_year > datetime.now().year:
            raise ValueError("birth_year cannot be in the future")
        if self.sex == "男":
            self.pregnancy = None
        return self


class DeepSeekConsent(BaseModel):
    consent: bool = False


def credentials_config() -> dict[str, Any]:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise HTTPException(503, "Google OAuth is not configured. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to backend/.env.")
    return {"web": {
        "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [REDIRECT_URI],
    }}


def oauth_failure_reason(exc: Exception) -> str:
    """Return a safe, allowlisted OAuth failure category (never exception text)."""
    allowed = {
        "invalid_client", "invalid_grant", "redirect_uri_mismatch",
        "unauthorized_client", "invalid_scope", "invalid_request",
    }
    error = getattr(exc, "error", None)
    if isinstance(error, str) and error in allowed:
        return error

    # OAuthlib error subclasses expose the Google OAuth error in their class
    # name even when their instance does not expose an `error` attribute.
    class_name = type(exc).__name__.lower()
    for code in allowed:
        if code.replace("_", "") in class_name.replace("_", ""):
            return code
    if "mismatchingstate" in class_name or "state" in class_name:
        return "invalid_state"
    if isinstance(exc, (TimeoutError, urllib.error.URLError)):
        return "network_error"
    if type(exc).__module__.startswith(("requests.", "urllib3.")) and any(
        marker in class_name for marker in ("connection", "timeout", "network")
    ):
        return "network_error"
    if "insecuretransport" in class_name:
        return "local_http_config"
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 401:
        return "invalid_client"
    if getattr(response, "status_code", None) == 400:
        return "invalid_grant"
    return "token_exchange"


@app.get("/api/v1/integrations/google/config")
def google_oauth_config_status() -> dict[str, Any]:
    """Expose only non-secret runtime settings useful for local OAuth setup."""
    parsed = urlparse(REDIRECT_URI)
    return {
        "client_id_configured": bool(CLIENT_ID),
        "client_id_format_valid": CLIENT_ID.endswith(".apps.googleusercontent.com"),
        "client_secret_configured": bool(CLIENT_SECRET),
        "redirect_uri": REDIRECT_URI,
        "redirect_uri_is_loopback_http": parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"},
        "frontend_url": FRONTEND_URL,
        "last_token_exchange_failure": _last_oauth_failure,
    }


def exchange_google_token(flow: Flow, authorization_response: str) -> None:
    """Permit OAuthlib's HTTP exception only for this loopback callback exchange."""
    callback = urlparse(authorization_response)
    configured = urlparse(REDIRECT_URI)
    loopback_hosts = {"localhost", "127.0.0.1", "::1"}
    is_local_http_callback = (
        configured.scheme == "http"
        and configured.hostname in loopback_hosts
        and callback.scheme == "http"
        and callback.hostname == configured.hostname
        and callback.port == configured.port
    )
    if not is_local_http_callback:
        flow.fetch_token(authorization_response=authorization_response)
        return

    # OAuthlib reads this process-wide flag. Scope it to the short local token
    # exchange, restore the prior value immediately, and serialize this section.
    with _oauth_exchange_lock:
        previous_insecure = os.environ.get("OAUTHLIB_INSECURE_TRANSPORT")
        previous_relax_scope = os.environ.get("OAUTHLIB_RELAX_TOKEN_SCOPE")
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        # Google may return a normalized or previously granted scope set.
        # OAuthlib raises a bare Warning on any difference by default; the
        # callback separately verifies that the required Calendar scope exists.
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
        try:
            flow.fetch_token(authorization_response=authorization_response)
        finally:
            if previous_insecure is None:
                os.environ.pop("OAUTHLIB_INSECURE_TRANSPORT", None)
            else:
                os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = previous_insecure
            if previous_relax_scope is None:
                os.environ.pop("OAUTHLIB_RELAX_TOKEN_SCOPE", None)
            else:
                os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = previous_relax_scope


def load_credentials() -> Credentials:
    if FERNET is None:
        raise HTTPException(503, "TOKEN_ENCRYPTION_KEY is missing or invalid. Generate a Fernet key and add it to backend/.env.")
    with connect_db() as db:
        row = db.execute("SELECT token FROM google_token WHERE id=1").fetchone()
    if not row:
        raise HTTPException(401, "Google Calendar is not connected.")
    try:
        token = json.loads(FERNET.decrypt(row["token"]).decode())
    except (InvalidToken, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "Saved Google credentials cannot be decrypted. Check TOKEN_ENCRYPTION_KEY.") from exc
    creds = Credentials.from_authorized_user_info(token, SCOPES)
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleRequest())
            save_credentials(creds)
        except RefreshError as exc:
            with connect_db() as db:
                db.execute("DELETE FROM google_token WHERE id=1")
            raise HTTPException(401, "Google authorization expired. Please connect Google Calendar again.") from exc
    return creds


def save_credentials(creds: Credentials, email: str | None = None) -> None:
    if FERNET is None:
        raise HTTPException(503, "TOKEN_ENCRYPTION_KEY is required before connecting Google Calendar.")
    encrypted = FERNET.encrypt(creds.to_json().encode())
    with connect_db() as db:
        old = db.execute("SELECT email FROM google_token WHERE id=1").fetchone()
        resolved_email = email if email is not None else (old["email"] if old else None)
        db.execute("INSERT OR REPLACE INTO google_token(id, token, email) VALUES(1, ?, ?)", (encrypted, resolved_email))


def read_personal_information() -> PersonalInformationIn | None:
    if FERNET is None:
        raise HTTPException(503, "Local encryption is not configured; set TOKEN_ENCRYPTION_KEY in backend/.env.")
    with connect_db() as db:
        row = db.execute("SELECT profile FROM personal_information WHERE id=1").fetchone()
    if not row:
        return None
    try:
        raw = json.loads(FERNET.decrypt(row["profile"]).decode("utf-8"))
        return PersonalInformationIn.model_validate(raw)
    except (InvalidToken, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "Saved personal information cannot be decrypted. Check TOKEN_ENCRYPTION_KEY.") from exc


def store_persona_file(memory: str) -> int:
    PERSONA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = memory.strip().encode("utf-8")
    temporary = PERSONA_PATH.with_suffix(".txt.tmp")
    temporary.write_bytes(data)
    temporary.replace(PERSONA_PATH)
    return len(data)


def summarize_with_deepseek(profile: PersonalInformationIn) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(503, "DeepSeek is not configured. Add DEEPSEEK_API_KEY to backend/.env; your saved profile remains local.")
    facts = {key: value for key, value in profile.model_dump().items() if value not in (None, "")}
    facts["sex"] = {"男": "Male", "女": "Female"}.get(facts.get("sex"), facts.get("sex"))
    facts["pregnancy"] = {
        "妊娠期": "Pregnant", "哺乳期": "Breastfeeding", "否": "Neither",
    }.get(facts.get("pregnancy"), facts.get("pregnancy"))
    request_body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Create a very short English memory for future medication-risk hints, under 280 characters. "
                    "Record only explicitly stated birth year (never calculate age), sex, pregnancy/breastfeeding, "
                    "medical history, allergies, long-term medicines, medication limits, and useful routine times. "
                    "Do not diagnose, infer blank fields, judge a drug's safety, or give treatment advice. "
                    "Unknown means unknown; blank does not mean no history. Prioritize allergies, contraindications, "
                    'and clinician instructions. Return JSON only: {"memory":"short text"}.'
                ),
            },
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False, separators=(",", ":"))},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 180,
        "stream": False,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except urllib.error.HTTPError as exc:
        raise HTTPException(502, f"DeepSeek API returned HTTP {exc.code}. Check its key and account quota.") from exc
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(502, "DeepSeek summary failed; your locally saved answers are unchanged.") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("memory"), str):
        raise HTTPException(502, "DeepSeek returned an invalid persona format.")
    memory = " ".join(parsed["memory"].split())
    if not memory:
        raise HTTPException(502, "DeepSeek returned an empty memory.")
    return memory[:280]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "medclock-api"}


@app.get("/auth/google/start")
def google_start() -> RedirectResponse:
    if FERNET is None:
        raise HTTPException(503, "Set TOKEN_ENCRYPTION_KEY in backend/.env before authorizing Google.")
    flow = Flow.from_client_config(credentials_config(), scopes=SCOPES, redirect_uri=REDIRECT_URI)
    state = secrets.token_urlsafe(32)
    url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true", state=state, prompt="consent")
    with connect_db() as db:
        db.execute("DELETE FROM oauth_state WHERE expires_at < ?", (datetime.now(timezone.utc).isoformat(),))
        db.execute("INSERT INTO oauth_state(state, expires_at) VALUES(?, ?)", (state, (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()))
    return RedirectResponse(url)


@app.get("/auth/google/callback")
def google_callback(request: Request) -> RedirectResponse:
    global _last_oauth_failure
    state = request.query_params.get("state", "")
    with connect_db() as db:
        row = db.execute("SELECT expires_at FROM oauth_state WHERE state=?", (state,)).fetchone()
        db.execute("DELETE FROM oauth_state WHERE state=?", (state,))
    if not row or datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        return RedirectResponse(f"{FRONTEND_URL}/?google=error&reason=invalid_state")
    if request.query_params.get("error"):
        reason = request.query_params.get("error", "access_denied")
        return RedirectResponse(f"{FRONTEND_URL}/?{urlencode({'google': 'error', 'reason': reason})}")
    flow = Flow.from_client_config(credentials_config(), scopes=SCOPES, redirect_uri=REDIRECT_URI, state=state)
    try:
        exchange_google_token(flow, str(request.url))
    except Exception as exc:
        reason = oauth_failure_reason(exc)
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        _last_oauth_failure = {
            "category": reason,
            "exception_type": type(exc).__name__,
            "http_status": status_code if isinstance(status_code, int) else None,
        }
        # Log only an allowlisted error code and exception class. Exception
        # strings/tracebacks can contain the one-time authorization code.
        logger.error("Google OAuth token exchange failed (type=%s, reason=%s, status=%s)", type(exc).__name__, reason, status_code)
        return RedirectResponse(f"{FRONTEND_URL}/?google=error&reason={reason}")
    else:
        _last_oauth_failure = None
    creds = flow.credentials
    granted_scopes = set(creds.scopes or SCOPES)
    if CALENDAR_READ_SCOPE not in granted_scopes:
        _last_oauth_failure = {
            "category": "insufficient_scope",
            "exception_type": "CalendarScopeNotGranted",
            "http_status": None,
        }
        logger.error("Google OAuth completed without the required Calendar read scope")
        return RedirectResponse(f"{FRONTEND_URL}/?google=error&reason=insufficient_scope")
    try:
        identity = verify_oauth2_token(creds.id_token, GoogleRequest(), CLIENT_ID) if creds.id_token else {}
    except Exception as exc:
        logger.error("Google OAuth identity verification failed (%s)", type(exc).__name__)
        return RedirectResponse(f"{FRONTEND_URL}/?google=error&reason=identity_verification")
    try:
        save_credentials(creds, identity.get("email"))
    except Exception as exc:
        logger.error("Google OAuth credential storage failed (%s)", type(exc).__name__)
        return RedirectResponse(f"{FRONTEND_URL}/?google=error&reason=credential_storage")
    return RedirectResponse(f"{FRONTEND_URL}/?google=connected")


@app.get("/api/v1/integrations/google/status")
def google_status() -> dict[str, Any]:
    with connect_db() as db:
        row = db.execute("SELECT email FROM google_token WHERE id=1").fetchone()
    return {"connected": bool(row), "email": row["email"] if row else None}


@app.delete("/api/v1/integrations/google")
def google_disconnect() -> Response:
    with connect_db() as db:
        db.execute("DELETE FROM google_token WHERE id=1")
        db.execute("DELETE FROM calendar_sync")
    return Response(status_code=204)


@app.get("/api/v1/personal-information")
def get_personal_information() -> dict[str, Any]:
    profile = read_personal_information()
    memory = PERSONA_PATH.read_text(encoding="utf-8") if PERSONA_PATH.exists() else None
    return {
        "profile": profile.model_dump() if profile else None,
        "memory": memory,
        "memory_bytes": len(memory.encode("utf-8")) if memory else 0,
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "deepseek_model": DEEPSEEK_MODEL,
    }


@app.put("/api/v1/personal-information")
def save_personal_information(profile: PersonalInformationIn) -> dict[str, Any]:
    if FERNET is None:
        raise HTTPException(503, "Local encryption is not configured; set TOKEN_ENCRYPTION_KEY in backend/.env.")
    previous = read_personal_information()
    changed = previous is None or previous.model_dump() != profile.model_dump()
    encrypted = FERNET.encrypt(json.dumps(profile.model_dump(), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    with connect_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO personal_information(id, profile, updated_at) VALUES(1, ?, ?)",
            (encrypted, datetime.now(timezone.utc).isoformat()),
        )
    # A generated memory is derived from the profile. Invalidate it only when
    # answers change so hints cannot use stale medical restrictions.
    if changed:
        PERSONA_PATH.unlink(missing_ok=True)
        LEGACY_PERSONA_PATH.unlink(missing_ok=True)
    memory = PERSONA_PATH.read_text(encoding="utf-8") if PERSONA_PATH.exists() else None
    return {
        "profile": profile.model_dump(),
        "memory": memory,
        "memory_bytes": len(memory.encode("utf-8")) if memory else 0,
    }


@app.post("/api/v1/personal-information/persona")
def generate_persona(consent: DeepSeekConsent) -> dict[str, Any]:
    if not consent.consent:
        raise HTTPException(400, "Explicit consent is required before sending health information to DeepSeek.")
    profile = read_personal_information()
    if profile is None:
        raise HTTPException(404, "Save Personal information before generating a persona.")
    memory = summarize_with_deepseek(profile)
    size = store_persona_file(memory)
    return {"memory": memory, "memory_bytes": size, "summary_source": "deepseek"}


@app.get("/api/v1/calendar/events")
def calendar_events(
    start: datetime = Query(..., description="ISO-8601 inclusive lower bound"),
    end: datetime = Query(..., description="ISO-8601 exclusive upper bound"),
) -> list[dict[str, Any]]:
    if start >= end:
        raise HTTPException(400, "start must be earlier than end")
    try:
        service = build("calendar", "v3", credentials=load_credentials(), cache_discovery=False)
        items = []
        page_token = None
        while True:
            result = service.events().list(
                calendarId=CALENDAR_ID, timeMin=start.isoformat(), timeMax=end.isoformat(),
                singleEvents=True, orderBy="startTime", maxResults=250, pageToken=page_token,
            ).execute()
            items.extend(result.get("items", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break
    except HttpError as exc:
        raise HTTPException(exc.resp.status, "Google Calendar request failed. Check Calendar API and OAuth access.") from exc
    events = []
    for event in items:
        begins = event.get("start", {})
        finishes = event.get("end", {})
        events.append({
            "id": event["id"], "summary": event.get("summary", "(No title)"),
            "start": begins.get("dateTime", begins.get("date")),
            "end": finishes.get("dateTime", finishes.get("date")),
            "all_day": "date" in begins, "html_link": event.get("htmlLink"),
            "description": event.get("description"),
            "is_busy": event.get("transparency") != "transparent" and event.get("status") != "cancelled",
        })
    return events


@app.post("/api/v1/calendar/sync")
def sync_calendar(
    start: datetime = Query(...), end: datetime = Query(...),
) -> dict[str, Any]:
    events = calendar_events(start, end)
    synced_at = datetime.now(timezone.utc).isoformat()
    with connect_db() as db:
        db.execute("DELETE FROM calendar_sync WHERE substr(start, 1, 10) >= ? AND substr(start, 1, 10) < ?", (start.date().isoformat(), end.date().isoformat()))
        for event in events:
            db.execute("""INSERT OR REPLACE INTO calendar_sync(event_id, summary, start, end, html_link, updated_at, all_day, is_busy)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)""", (event["id"], event["summary"], event["start"], event["end"], event["html_link"], synced_at, event["all_day"], event["is_busy"]))
    return {"synced_at": synced_at, "count": len(events), "events": events}


@app.get("/api/v1/calendar/synced")
def synced_events(start: datetime = Query(...), end: datetime = Query(...)) -> list[dict[str, Any]]:
    with connect_db() as db:
        rows = db.execute("SELECT event_id AS id, summary, start, end, html_link FROM calendar_sync WHERE substr(start, 1, 10) >= ? AND substr(start, 1, 10) < ? ORDER BY start", (start.date().isoformat(), end.date().isoformat())).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/v1/medications")
def list_medications() -> list[dict[str, Any]]:
    with connect_db() as db:
        rows = db.execute("SELECT * FROM medications ORDER BY time, name").fetchall()
    return [medication_row(row) for row in rows]


def medication_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    medication = dict(row)
    raw_times = medication.pop("times_json", "[]")
    try:
        times = json.loads(raw_times) if isinstance(raw_times, str) else []
    except json.JSONDecodeError:
        times = []
    medication["times"] = times if isinstance(times, list) and times else [medication["time"]]
    return medication


def save_medication_record(db: sqlite3.Connection, med: MedicationIn, replace: bool = False) -> None:
    statement = "INSERT OR REPLACE" if replace else "INSERT"
    db.execute(
        f"{statement} INTO medications(id,name,dosage,time,instructions,tone,times_json,category,indication,approved_shift_minutes) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            med.id, med.name, med.dosage, med.time, med.instructions, med.tone,
            json.dumps(med.times, ensure_ascii=False), med.category, med.indication,
            med.approved_shift_minutes,
        ),
    )


@app.post("/api/v1/medications", status_code=201)
def create_medication(med: MedicationIn) -> dict[str, Any]:
    with connect_db() as db:
        save_medication_record(db, med, replace=True)
    return med.model_dump()


@app.patch("/api/v1/medications/{med_id}")
def update_medication(med_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    with connect_db() as db:
        row = db.execute("SELECT * FROM medications WHERE id=?", (med_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Medication not found")
        values = {**medication_row(row), **patch}
        if "time" in patch and "times" not in patch:
            values["times"] = [patch["time"]]
        med = MedicationIn.model_validate(values)
        db.execute(
            "UPDATE medications SET name=?,dosage=?,time=?,instructions=?,tone=?,times_json=?,category=?,indication=?,approved_shift_minutes=? WHERE id=?",
            (med.name, med.dosage, med.time, med.instructions, med.tone, json.dumps(med.times, ensure_ascii=False), med.category, med.indication, med.approved_shift_minutes, med_id),
        )
        db.execute("DELETE FROM schedule_override WHERE medication_id=?", (med_id,))
    return med.model_dump()


@app.delete("/api/v1/medications/{med_id}", status_code=204)
def delete_medication(med_id: str) -> Response:
    with connect_db() as db:
        db.execute("DELETE FROM schedule_override WHERE medication_id=?", (med_id,))
        db.execute("DELETE FROM medications WHERE id=?", (med_id,))
    return Response(status_code=204)


@app.get("/api/v1/schedule")
def schedule(day: date = Query(...)) -> list[dict[str, Any]]:
    with connect_db() as db:
        meds = [medication_row(row) for row in db.execute("SELECT * FROM medications ORDER BY time")]
        overrides = {
            (row["medication_id"], row["dose_index"]): row["planned_time"]
            for row in db.execute("SELECT medication_id, dose_index, planned_time FROM schedule_override WHERE date=?", (day.isoformat(),))
        }
    rows = [
        {**med, "medication_id": med["id"], "dose_index": index,
         "time": overrides.get((med["id"], index), dose_time), "original_time": dose_time,
         "planned_override": (med["id"], index) in overrides,
         "date": day.isoformat(), "kind": "medication"}
        for med in meds
        for index, dose_time in enumerate(med["times"])
    ]
    return sorted(rows, key=lambda row: (row["time"], row["medication_id"], row["dose_index"]))


def load_schedule_context(day: date) -> dict[str, Any]:
    with connect_db() as db:
        med_rows = db.execute("SELECT id, name, time, times_json, instructions, approved_shift_minutes FROM medications ORDER BY time").fetchall()
        override_rows = db.execute("SELECT medication_id, dose_index, planned_time FROM schedule_override WHERE date=?", (day.isoformat(),)).fetchall()
        event_rows = db.execute(
            "SELECT event_id AS id, summary, start, end, all_day, is_busy FROM calendar_sync WHERE substr(end,1,10) >= ? AND substr(start,1,10) <= ? ORDER BY start",
            ((day - timedelta(days=1)).isoformat(), (day + timedelta(days=1)).isoformat()),
        ).fetchall()
    overrides = {(row["medication_id"], row["dose_index"]): row["planned_time"] for row in override_rows}
    meds = []
    for row in med_rows:
        med = dict(row)
        try:
            times = json.loads(med.pop("times_json", "[]"))
        except json.JSONDecodeError:
            times = []
        times = times if isinstance(times, list) and times else [med["time"]]
        instructions = med.pop("instructions", "")
        meal_constraint = bool(MEAL_TERMS.search(instructions))
        current_times = [overrides.get((med["id"], index), value) for index, value in enumerate(times)]
        meds.append({**med, "times": times, "current_times": current_times, "meal_constraint": meal_constraint})
    events = []
    for row in event_rows:
        event = dict(row)
        event["all_day"] = bool(event["all_day"])
        event["is_busy"] = bool(event["is_busy"])
        if event_intersects_day(event, day):
            events.append(event)
    return {"date": day.isoformat(), "medications": meds, "events": events}


@app.post("/api/v1/schedule/plan")
def plan_schedule(payload: SchedulePlanIn) -> dict[str, Any]:
    if not payload.consent:
        raise HTTPException(400, "Consent is required to send medication names, dose times, approved timing windows, and event titles/times to DeepSeek.")
    if not DEEPSEEK_API_KEY:
        raise HTTPException(503, "DeepSeek is not configured. Add DEEPSEEK_API_KEY to backend/.env.")
    context = load_schedule_context(payload.date)
    try:
        return run_schedule_agent(payload.date, context, DEEPSEEK_API_KEY, DEEPSEEK_MODEL)
    except LLMError as exc:
        raise HTTPException(502, "DeepSeek schedule planning failed. Check the API key, model access, and quota.") from exc


@app.post("/api/v1/schedule/accept")
def accept_schedule(payload: ScheduleAcceptIn) -> dict[str, Any]:
    """Persist only candidate times that the deterministic planner regenerates."""
    context = load_schedule_context(payload.date)
    runtime = ScheduleToolRuntime(payload.date, context)
    runtime.detect_conflicts()
    accepted = []
    for suggestion in payload.suggestions:
        options = runtime.alternatives(suggestion.medication_id, suggestion.dose_index)
        valid_times = {item["time"] for item in options.get("candidates", [])}
        if suggestion.planned_time not in valid_times or suggestion.original_time != next(
            (item["original_time"] for item in runtime.conflicts or []
             if item["medication_id"] == suggestion.medication_id and item["dose_index"] == suggestion.dose_index), ""
        ):
            raise HTTPException(409, "This suggestion is no longer valid. Re-plan the day and review the new options.")
        event_ids = [item["event_id"] for item in runtime.conflicts or [] if item["medication_id"] == suggestion.medication_id and item["dose_index"] == suggestion.dose_index]
        accepted.append((payload.date.isoformat(), suggestion.medication_id, suggestion.dose_index, suggestion.original_time,
                         suggestion.planned_time, json.dumps(event_ids), datetime.now(timezone.utc).isoformat()))
    with connect_db() as db:
        db.executemany("""INSERT OR REPLACE INTO schedule_override
            (date, medication_id, dose_index, original_time, planned_time, event_ids_json, accepted_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)""", accepted)
    return {"date": payload.date.isoformat(), "accepted": len(accepted), "schedule": schedule(payload.date)}


@app.post("/api/v1/medications/risk-assessment")
def medication_risk_assessment(payload: MedicationRiskAssessmentIn) -> dict[str, Any]:
    """Screen every current pair using the vendored deterministic MedCheck engine."""
    meds = payload.medications
    settings = MedCheckSettings(
        engine="template", lang="en", compact=True, verbose=False, trace=False,
        fixture_fallback=True, request_timeout=6.0,
    )
    summaries: dict[str, dict[str, Any]] = {}
    pair_results: list[dict[str, Any]] = []
    risk_by_id: dict[str, list[str]] = {med.id: [] for med in meds}

    def medication_summary(med: MedicationIn, record: Any | None) -> dict[str, Any]:
        if record is None:
            return {
                "id": med.id, "query": med.name, "matched": False,
                "display_name": med.name, "generic_names": [], "active_ingredients": [],
                "drug_classes": [], "class_source": "", "label_url": "",
                "label_effective_time": "", "fetched_from": "", "message": "No official label found.",
                "leaflet": [], "missing_sections": [],
            }
        sheet = brief_sheet(record, max_sentences=1)
        leaflet = []
        for section in sheet.get("sections", []):
            for item in section.get("items", []):
                leaflet.append({
                    "section": safety_field_label_en(item.get("field", "")),
                    "text": item.get("text", ""),
                    "source_url": record.url,
                })
        return {
            "id": med.id, "query": med.name, "matched": record.matched,
            "display_name": record.display,
            "generic_names": record.generic_names,
            "active_ingredients": record.active_ingredients,
            "drug_classes": record.pharm_classes,
            "class_source": record.pharm_class_source,
            "label_url": record.url,
            "label_effective_time": record.effective_time,
            "fetched_from": record.fetched_from,
            "message": ("Official label found." if record.matched else "No official label found."),
            "leaflet": leaflet[:12],
            "missing_sections": sheet.get("missing", []),
        }

    if len(meds) == 1:
        med = meds[0]
        agent = MedCheckAgent(settings)
        try:
            record = agent.ensure_label(med.name, slot="A")
        except Exception:
            record = None
        summaries[med.id] = medication_summary(med, record)
    else:
        for i, med_a in enumerate(meds):
            for med_b in meds[i + 1:]:
                try:
                    agent = MedCheckAgent(settings)
                    result = agent.analyze(med_a.name, med_b.name)
                    verdict = result.verdict if result.ok and result.verdict else Verdict.UNKNOWN
                    findings = []
                    for finding in (result.screen.findings if result.screen else []):
                        localized = finding.localized("en")
                        evidence_items = [agent.runtime.registry.get(evidence_id) for evidence_id in finding.evidence_ids]
                        evidence_payload = []
                        seen_quotes: set[tuple[str, str]] = set()
                        for evidence in evidence_items:
                            if evidence is None or (evidence.quote, evidence.url) in seen_quotes:
                                continue
                            seen_quotes.add((evidence.quote, evidence.url))
                            evidence_payload.append({
                                "id": evidence.evidence_id,
                                "drug": evidence.drug,
                                "section": safety_field_label_en(evidence.section),
                                "quote": evidence.short(450),
                                "url": evidence.url,
                            })
                        findings.append({
                            "rule_id": finding.rule_id,
                            "severity": finding.severity,
                            "title": localized["title"],
                            "detail": localized["detail"],
                            "action": localized["action"],
                            "evidence_ids": finding.evidence_ids,
                            "evidence": evidence_payload,
                        })
                    limitations = []
                    for note in (result.screen.limitations if result.screen else []):
                        text = note.en or "Some label limitations apply; review the source label."
                        if re.search(r"[\u4e00-\u9fff]", text):
                            text = "Class-level evidence is indirect and does not confirm a drug-specific interaction."
                        limitations.append(text)
                    records = result.records
                    record_a = records[0] if records else None
                    record_b = records[1] if len(records) > 1 else None
                except Exception:
                    verdict = Verdict.UNKNOWN
                    findings = []
                    limitations = ["The check failed or official data is unavailable. Try again later and ask a pharmacist."]
                    record_a = record_b = None

                pair_results.append({
                    "drug_a_id": med_a.id,
                    "drug_b_id": med_b.id,
                    "drug_a": med_a.name,
                    "drug_b": med_b.name,
                    "verdict": verdict,
                    "findings": findings,
                    "limitations": limitations,
                })
                risk_by_id[med_a.id].append(verdict)
                risk_by_id[med_b.id].append(verdict)
                if med_a.id not in summaries or record_a and record_a.matched:
                    summaries[med_a.id] = medication_summary(med_a, record_a)
                if med_b.id not in summaries or record_b and record_b.matched:
                    summaries[med_b.id] = medication_summary(med_b, record_b)

    rank = {Verdict.LOW: 1, Verdict.UNKNOWN: 2, Verdict.HIGH: 3, Verdict.CRITICAL: 4}
    for med in meds:
        summary = summaries.get(med.id) or medication_summary(med, None)
        pair_levels = risk_by_id[med.id]
        if not pair_levels:
            summary.update({"risk": Verdict.UNKNOWN, "risk_label": "Not assessed", "assessment": "not_assessed", "consult": False})
        else:
            worst = max(pair_levels, key=lambda level: rank.get(level, 2))
            label = {
                Verdict.CRITICAL: "High risk",
                Verdict.HIGH: "Caution",
                Verdict.LOW: "No direct conflict found",
                Verdict.UNKNOWN: "Insufficient evidence",
            }.get(worst, "Insufficient evidence")
            summary.update({
                "risk": worst,
                "risk_label": label,
                "assessment": "assessed" if all(level != Verdict.UNKNOWN for level in pair_levels) else "incomplete",
                "consult": worst in (Verdict.CRITICAL, Verdict.HIGH, Verdict.UNKNOWN),
            })
        summary["pair_count"] = len(pair_levels)
        summaries[med.id] = summary

    source = "openFDA official drug labeling, local cache, or bundled fixtures"
    return {
        "source": source,
        "medications": [summaries[med.id] for med in meds],
        "pairs": pair_results,
        "checked_pairs": len(pair_results),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
