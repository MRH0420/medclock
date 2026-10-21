/** FastAPI boundary. Set VITE_API_BASE_URL when the backend is ready. */
const API_BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '')
const API_ORIGIN = API_BASE ? new URL(API_BASE).origin : ''

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_BASE) throw new Error('FastAPI is not configured')
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) throw new Error(`API ${response.status}: ${await response.text()}`)
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const medicationApi = {
  list: <T,>() => api<T[]>('/medications'),
  create: <T,>(data: unknown) => api<T>('/medications', { method: 'POST', body: JSON.stringify(data) }),
  update: <T,>(id: string, data: unknown) => api<T>(`/medications/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  remove: (id: string) => api<void>(`/medications/${id}`, { method: 'DELETE' }),
}

export type MedicationRiskLevel = 'critical' | 'high' | 'low' | 'unknown'
export type MedicationSafety = {
  id: string
  query: string
  matched: boolean
  display_name: string
  generic_names: string[]
  active_ingredients: string[]
  drug_classes: string[]
  class_source: string
  label_url: string
  label_effective_time: string
  fetched_from: string
  message: string
  leaflet: { section: string; text: string; source_url: string }[]
  missing_sections: string[]
  risk: MedicationRiskLevel
  risk_label: string
  assessment: 'assessed' | 'incomplete' | 'not_assessed'
  consult: boolean
  pair_count: number
}
export type MedicationRiskPair = {
  drug_a_id: string
  drug_b_id: string
  drug_a: string
  drug_b: string
  verdict: MedicationRiskLevel
  findings: {
    rule_id: string
    severity: string
    title: string
    detail: string
    action: string
    evidence_ids: string[]
    evidence: { id: string; drug: string; section: string; quote: string; url: string }[]
  }[]
  limitations: string[]
}

export const medicationRiskApi = {
  assess: (medications: unknown) => api<{
    source: string
    medications: MedicationSafety[]
    pairs: MedicationRiskPair[]
    checked_pairs: number
    checked_at: string
  }>('/medications/risk-assessment', { method: 'POST', body: JSON.stringify({ medications }) }),
}

export const scheduleApi = {
  list: <T,>(date: string) => api<T[]>(`/schedule?date=${encodeURIComponent(date)}`),
  plan: (date: string, consent: boolean) => api<SchedulePlan>(`/schedule/plan`, {
    method: 'POST', body: JSON.stringify({ date, consent }),
  }),
  accept: (date: string, suggestions: ScheduleSuggestion[]) => api<{
    date: string; accepted: number; schedule: ScheduleDose[]
  }>('/schedule/accept', { method: 'POST', body: JSON.stringify({ date, suggestions }) }),
}

export type ScheduleSuggestion = {
  medication_id: string
  medication: string
  dose_index: number
  original_time: string
  time: string
  shift_minutes: number
}
export type ScheduleDose = { medication_id: string; dose_index: number; time: string; original_time: string; planned_override: boolean }
export type SchedulePlan = {
  date: string
  summary: string
  tool_calls: string[]
  conflicts: { medication_id: string; medication: string; dose_index: number; original_time: string; event_id: string; event: string; event_start: string; event_end: string }[]
  alternatives: { medication_id: string; dose_index: number; original_time?: string; candidates?: { time: string; shift_minutes: number }[]; manual_review: boolean; reason: string }[]
  suggestions: ScheduleSuggestion[]
  requires_review: { medication_id: string; dose_index: number; reason: string }[]
  glucose_guidance: {
    event_id: string
    event: string
    start: string
    end: string
    duration_hours: number | null
    medications: { name: string; classes: string }[]
    guidance: string[]
  }[]
}

export type GoogleCalendarEvent = {
  id: string
  summary: string
  start: string
  end: string
  all_day: boolean
  is_busy?: boolean
  html_link?: string
  description?: string
}

export const googleCalendarApi = {
  status: () => api<{ connected: boolean; email: string | null }>('/integrations/google/status'),
  connectUrl: () => `${API_ORIGIN}/auth/google/start`,
  sync: (start: string, end: string) => api<{ synced_at: string; count: number; events: GoogleCalendarEvent[] }>(
    `/calendar/sync?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`,
    { method: 'POST' },
  ),
}

export type PersonalInformation = {
  birth_year: number
  sex: '男' | '女'
  pregnancy: '妊娠期' | '哺乳期' | '否' | null
  breakfast: string | null
  lunch: string | null
  dinner: string | null
  wake: string | null
  sleep: string | null
  medical_history: string
  medication_limits: string
}

export const personalInformationApi = {
  get: () => api<{
    profile: PersonalInformation | null
    memory: string | null
    memory_bytes: number
    deepseek_configured: boolean
    deepseek_model: string
  }>('/personal-information'),
  save: (profile: PersonalInformation) => api<{
    profile: PersonalInformation
    memory: string | null
    memory_bytes: number
  }>('/personal-information', { method: 'PUT', body: JSON.stringify(profile) }),
  summarize: () => api<{
    memory: string
    memory_bytes: number
    summary_source: 'deepseek'
  }>('/personal-information/persona', { method: 'POST', body: JSON.stringify({ consent: true }) }),
}
