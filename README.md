# MedClock

Mobile-first MedClock prototype: Vite + React + TypeScript frontend and a FastAPI backend with Google Calendar read/sync.

## Run

```sh
npm ci
npm run dev
```

## Run the API

In a second terminal:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Generate an encryption key and put the printed value in `backend/.env` as `TOKEN_ENCRYPTION_KEY`:

```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

In Google Cloud, enable **Google Calendar API**, create an OAuth **Web application** client, add `http://localhost:8000/auth/google/callback` as an authorized redirect URI, and add your Google account as a test user while the consent screen is in testing. Copy the client ID and secret into `backend/.env`. Keep the client secret private. The backend requests OpenID/email identity plus `calendar.events.owned.readonly` to read and sync events on the user's primary calendar; it does not access Gmail or modify Calendar events.

Start FastAPI:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

The token is encrypted in `backend/.data/medclock.sqlite3`. Keep `TOKEN_ENCRYPTION_KEY` stable or the saved authorization cannot be decrypted. Visit `http://localhost:8000/docs` for the API.

## Personal information and DeepSeek

Profile answers (birth year, sex, optional pregnancy/lactation state, daily rhythm, medical history, and medication limits) are encrypted in the local SQLite database. Saving a profile alone does not create an AI memory. To enable generation, set `DEEPSEEK_API_KEY` in `backend/.env` and restart FastAPI. The UI requires an explicit consent checkbox and **Save & generate short memory** action before sending health answers to DeepSeek. DeepSeek returns a short text memory stored locally at `backend/.data/persona.txt`; editing the profile invalidates the previous memory. The memory can contain sensitive health information and is plain text on this device. It is a context aid, not medical advice.

In another terminal, create `.env.local` at the project root:

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

Then run `npm run dev`, open the Calendar tab and choose **Connect**. Complete Google consent. MedClock will fetch this month's events after returning to the app; **Sync now** refreshes them. Google OAuth is a browser redirect, and credentials stay on the backend.

## API and demo behavior

Set `VITE_API_BASE_URL` to the backend API root (for example `http://localhost:8000/api/v1`). `src/api.ts` is the single place for HTTP behavior. The backend provides:

- `GET /medications` → medication array
- `POST /medications` → create a medication
- `PATCH /medications/{id}` → update a medication
- `DELETE /medications/{id}` → remove a medication
- `POST /medications/risk-assessment` → retrieve official label/class data and screen every listed medication pair; returns risk badges, evidence findings, and quick leaflet excerpts
- `GET /schedule?date=YYYY-MM-DD` → schedule entries
- `GET /integrations/google/status` → connection state and account email
- `GET /auth/google/start` → begins Google OAuth
- `GET /auth/google/callback` → OAuth callback
- `GET /calendar/events?start=...&end=...` → read Google events in an ISO-8601 range
- `POST /calendar/sync?start=...&end=...` → refresh and cache Google events for that range
- `GET /calendar/synced?start=...&end=...` → return cached events
- `DELETE /integrations/google` → disconnect and clear cached Google data
- `GET /personal-information` → get the saved profile, generated memory, and DeepSeek configuration status
- `PUT /personal-information` → encrypt and save the profile locally, then invalidate any stale generated memory
- `POST /personal-information/persona` → summarize the saved profile with DeepSeek (requires explicit consent and `DEEPSEEK_API_KEY`)

Medication entries support a category (prescription, OTC, supplement, herbal/traditional, or other), an optional indication, and up to eight daily dose times. Existing single-time entries are migrated on startup and continue to work.

When the API URL is absent, the prototype uses local demo data and persists medication entries in `localStorage`. Google OAuth needs a browser-accessible backend URL and valid Google OAuth credentials; without those, the UI can run in local demo mode but Google sync cannot be completed.

## Git workflow

This folder is an independent Git repository. After its GitHub remote is configured, use standard commands here:

```sh
git status
git add .
git commit -m "Describe your change"
git push
```

Environment files and local health/calendar databases are excluded. Start from `.env.example` and `backend/.env.example`; all API key values in the examples are blank. Do not commit real `.env` files.
