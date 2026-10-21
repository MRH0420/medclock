import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { googleCalendarApi, medicationApi, medicationRiskApi, scheduleApi } from './api'
import type { GoogleCalendarEvent, MedicationRiskPair, MedicationSafety, ScheduleDose, SchedulePlan } from './api'
import { PersonalInformationScreen } from './PersonalInformation'
import { Icon, StatusGlyphs } from './icons'
import type { IconName } from './icons'

type Tab = 'Today' | 'Calendar' | 'Medications' | 'Profile'
type Med = {
  id: string
  name: string
  dosage: string
  time: string
  times: string[]
  category: 'Prescription' | 'OTC' | 'Supplement' | 'Herbal/Traditional' | 'Other'
  indication: string
  approved_shift_minutes: number
  instructions: string
  tone: string
}
type TimelineEvent = { time: string; end?: string; title: string; icon: IconName; kind: 'point' | 'range'; med?: string; safety?: MedicationSafety; plannedOverride?: boolean }
function localDayKey(value: Date): string {
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(value)
  const part = (type: Intl.DateTimeFormatPartTypes) => parts.find((item) => item.type === type)?.value ?? '00'
  return `${part('year')}-${part('month')}-${part('day')}`
}

const TABS: Tab[] = ['Today', 'Calendar', 'Medications', 'Profile']
const TAB_ICON: Record<Tab, IconName> = {
  Today: 'today',
  Calendar: 'calendar',
  Medications: 'pill',
  Profile: 'profile',
}

const TZ = 'Asia/Hong_Kong'
const LA_TZ = 'America/Los_Angeles'
const clockFmt = new Intl.DateTimeFormat('en-GB', {
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
  timeZone: TZ,
})
const laClockPartsFmt = new Intl.DateTimeFormat('en-US', {
  timeZone: LA_TZ, hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZoneName: 'short',
})
const laDatePartsFmt = new Intl.DateTimeFormat('en-US', {
  timeZone: LA_TZ, year: 'numeric', month: '2-digit', day: '2-digit',
})
function hktToLosAngeles(dateKey: string, timeValue?: string): string {
  const match = timeValue?.match(/^(\d{2}):(\d{2})(?:\s*\+(\d+))?$/)
  if (!match) return '—'
  const [, hour, minute, dayOffset = '0'] = match
  const [year, month, day] = dateKey.split('-').map(Number)
  const hktDate = new Date(Date.UTC(year, month - 1, day + Number(dayOffset)))
  const instant = new Date(Date.UTC(
    hktDate.getUTCFullYear(), hktDate.getUTCMonth(), hktDate.getUTCDate(), Number(hour) - 8, Number(minute),
  ))
  const clockParts = laClockPartsFmt.formatToParts(instant)
  const part = (parts: Intl.DateTimeFormatPart[], type: string) => parts.find((item) => item.type === type)?.value ?? ''
  const time = `${part(clockParts, 'hour')}:${part(clockParts, 'minute')}`
  const zone = part(clockParts, 'timeZoneName')
  const laDateParts = laDatePartsFmt.formatToParts(instant)
  const laDate = `${part(laDateParts, 'year')}-${part(laDateParts, 'month')}-${part(laDateParts, 'day')}`
  const dayDifference = Math.round((Date.parse(`${laDate}T00:00:00Z`) - Date.UTC(
    hktDate.getUTCFullYear(), hktDate.getUTCMonth(), hktDate.getUTCDate(),
  )) / 86_400_000)
  const offset = dayDifference === 0 ? '' : ` ${dayDifference < 0 ? '−' : '+'}${Math.abs(dayDifference)}`
  return `${time} ${zone}${offset}`
}
function timezoneOffsetLabel(zone: string, value: Date): string {
  const parts = new Intl.DateTimeFormat('en-US', { timeZone: zone, hour: '2-digit', timeZoneName: 'longOffset' }).formatToParts(value)
  const offset = parts.find((part) => part.type === 'timeZoneName')?.value ?? ''
  return offset.replace(/^GMT([+-])0?(\d{1,2}):00$/, 'GMT$1$2')
}
const calendarDateFmt = new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', timeZone: TZ })
const calendarAllDayDateFmt = new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', timeZone: 'UTC' })
const calendarDateTimeFmt = new Intl.DateTimeFormat('en', {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZone: TZ,
})
function calendarEventDateLabel(event: GoogleCalendarEvent): string {
  if (event.all_day) {
    const startDay = event.start.slice(0, 10)
    const exclusiveEnd = new Date(`${event.end.slice(0, 10)}T00:00:00Z`)
    exclusiveEnd.setUTCDate(exclusiveEnd.getUTCDate() - 1)
    const endDay = exclusiveEnd.toISOString().slice(0, 10)
    const startLabel = calendarAllDayDateFmt.format(new Date(`${startDay}T00:00:00Z`))
    const endLabel = calendarAllDayDateFmt.format(exclusiveEnd)
    return startDay === endDay ? startLabel : `${startLabel} – ${endLabel}`
  }
  const start = new Date(event.start)
  const end = new Date(event.end)
  const startLabel = calendarDateFmt.format(start)
  return localDayKey(start) === localDayKey(end) ? startLabel : `${startLabel} – ${calendarDateFmt.format(end)}`
}
function calendarEventIntervalLabel(event: GoogleCalendarEvent): string {
  if (event.all_day) return 'All day'
  const start = new Date(event.start)
  const end = new Date(event.end)
  return localDayKey(start) === localDayKey(end)
    ? `${clockFmt.format(start)} – ${clockFmt.format(end)}`
    : `${calendarDateTimeFmt.format(start)} – ${calendarDateTimeFmt.format(end)}`
}
function travelMomentLabel(value: string): string {
  return value.replace('T', ' ').replace(/([+-]\d{2}:\d{2}|Z)$/, ' (UTC$1)')
}
function withDayOffset(value: string) {
  const match = value.match(/^(.*?)(\s+[+−-]\d+)$/)
  return match ? <>{match[1]}<span className="day-offset">{match[2].trim()}</span></> : value
}
const COPY: Record<Tab, { overline: string; sub: string }> = {
  Today: { overline: 'Your day, in rhythm', sub: 'Every dose, plan and break on one spine.' },
  Calendar: { overline: 'Plan ahead', sub: 'Keep your routine and your plans in sync.' },
  Medications: { overline: 'Your routine', sub: 'A little rhythm makes a big difference.' },
  Profile: { overline: 'Device & connections', sub: 'Make MedClock work for your day.' },
}
const CONNECTIONS: { icon: IconName; title: string; desc: string }[] = [
  { icon: 'calendar', title: 'Google Calendar', desc: 'See events beside your medications.' },
  { icon: 'heart-pulse', title: 'Health Connect', desc: 'Bring your health context together.' },
  { icon: 'map-pin', title: 'Location & time zone', desc: 'Keep times accurate while travelling.' },
]

const API_ENABLED = Boolean(import.meta.env.VITE_API_BASE_URL)

const seedMeds: Med[] = [
  { id: 'a', name: 'Medication A', dosage: '1 tablet · 500 mg', time: '08:00', times: ['08:00'], category: 'Prescription', indication: '', approved_shift_minutes: 0, instructions: 'Take with breakfast', tone: 'violet' },
  { id: 'b', name: 'Medication B', dosage: '1 capsule · 20 mg', time: '20:00', times: ['20:00'], category: 'Prescription', indication: '', approved_shift_minutes: 0, instructions: 'Take after dinner', tone: 'blue' },
]
const baseEvents: TimelineEvent[] = [
  { time: '07:00', title: 'Wake', icon: 'sun', kind: 'point' },
  { time: '10:00', end: '11:30', title: 'Class', icon: 'cap', kind: 'range' },
  { time: '13:00', end: '13:45', title: 'Lunch', icon: 'utensils', kind: 'range' },
  { time: '16:00', end: '17:00', title: 'Walk', icon: 'activity', kind: 'range' },
  { time: '22:30', end: '07:00 +1', title: 'Sleep', icon: 'moon', kind: 'range' },
]

function readStoredMeds(): Med[] {
  try {
    const saved = localStorage.getItem('medclock.meds')
    const parsed = saved ? (JSON.parse(saved) as Med[]) : null
    return Array.isArray(parsed) && parsed.length ? parsed.map((med) => ({
      ...med,
      times: Array.isArray(med.times) && med.times.length ? med.times : [med.time],
      category: med.category ?? 'Prescription',
      indication: med.indication ?? '',
      approved_shift_minutes: med.approved_shift_minutes ?? 0,
    })) : seedMeds
  } catch {
    return seedMeds
  }
}

export default function App() {
  const [tab, setTab] = useState<Tab>('Today')
  const [meds, setMeds] = useState<Med[]>(readStoredMeds)
  const [medicationsLoaded, setMedicationsLoaded] = useState(!API_ENABLED)
  const [draftTimes, setDraftTimes] = useState<string[]>(['08:00'])
  const [medicationSafety, setMedicationSafety] = useState<MedicationSafety[]>([])
  const [medicationRiskPairs, setMedicationRiskPairs] = useState<MedicationRiskPair[]>([])
  const [safetyLoading, setSafetyLoading] = useState(false)
  const [safetyError, setSafetyError] = useState(false)
  const [sheetOpen, setSheetOpen] = useState(false)
  const [editingMed, setEditingMed] = useState<Med | null>(null)
  const [selected, setSelected] = useState<Med | null>(null)
  const [toast, setToast] = useState('')
  const [month, setMonth] = useState(() => new Date(new Date().getFullYear(), new Date().getMonth(), 1))
  const [weekView, setWeekView] = useState(false)
  const [pickedDay, setPickedDay] = useState<number | null>(null)
  const [now, setNow] = useState(() => new Date())
  const [googleConnected, setGoogleConnected] = useState(false)
  const [googleEmail, setGoogleEmail] = useState<string | null>(null)
  const [googleEvents, setGoogleEvents] = useState<GoogleCalendarEvent[]>([])
  const [calendarSyncing, setCalendarSyncing] = useState(false)
  const [scheduleConsent, setScheduleConsent] = useState(false)
  const [schedulePlanning, setSchedulePlanning] = useState(false)
  const [schedulePlan, setSchedulePlan] = useState<SchedulePlan | null>(null)
  const [scheduleAccepting, setScheduleAccepting] = useState(false)
  const [todaySchedule, setTodaySchedule] = useState<ScheduleDose[]>([])
  const [calendarSchedule, setCalendarSchedule] = useState<ScheduleDose[]>([])
  const [calendarScheduleDate, setCalendarScheduleDate] = useState('')
  const [personalInformationOpen, setPersonalInformationOpen] = useState(false)

  const scrollRef = useRef<HTMLDivElement | null>(null)
  const navRef = useRef<HTMLElement | null>(null)
  const sheetRef = useRef<HTMLFormElement | null>(null)
  const openerRef = useRef<HTMLElement | null>(null)
  const preserveSelectedOnPop = useRef(false)
  const scheduleConsentRef = useRef(false)
  const activeDateRef = useRef('')
  const toastTimer = useRef<number | undefined>(undefined)
  const scrollMemory = useRef<Record<string, number>>({})
  /* One history entry per overlay, so device/browser back closes it instead of
     leaving the app — and the underlying screen keeps its scroll position. */
  const pushed = useRef<number>(window.history.state && window.history.state.medclock ? 1 : 0)

  const notify = useCallback((message: string) => {
    setToast(message)
    window.clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast(''), 3200)
  }, [])

  const saveMeds = useCallback((next: Med[]) => {
    setMeds(next)
    try {
      localStorage.setItem('medclock.meds', JSON.stringify(next))
    } catch {
      /* private mode / storage disabled — the session still works in memory */
    }
  }, [])

  const pushLayer = useCallback(() => {
    window.history.pushState({ medclock: true }, '')
    pushed.current += 1
  }, [])

  const closeLayer = useCallback(() => {
    if (pushed.current > 0) {
      window.history.back()
      return
    }
    setSelected(null)
    setSheetOpen(false)
    setPersonalInformationOpen(false)
  }, [])

  const openMedication = useCallback((med: Med) => {
    pushLayer()
    setSelected(med)
  }, [pushLayer])

  const openSheet = useCallback((med?: Med) => {
    openerRef.current = document.activeElement as HTMLElement | null
    preserveSelectedOnPop.current = Boolean(med)
    setEditingMed(med ?? null)
    setDraftTimes(med?.times?.length ? med.times : [med?.time ?? '08:00'])
    pushLayer()
    setSheetOpen(true)
  }, [pushLayer])

  const openPersonalInformation = useCallback(() => {
    pushLayer()
    setPersonalInformationOpen(true)
  }, [pushLayer])

  /* A modest ticker: clock, greeting and the "now" marker stay truthful. */
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 30000)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => () => window.clearTimeout(toastTimer.current), [])

  useEffect(() => {
    const onPop = () => {
      pushed.current = Math.max(0, pushed.current - 1)
      if (!preserveSelectedOnPop.current) setSelected(null)
      preserveSelectedOnPop.current = false
      setSheetOpen(false)
      setEditingMed(null)
      setPersonalInformationOpen(false)
    }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  /* Load the server list once; local demo data covers an offline FastAPI. */
  useEffect(() => {
    if (!API_ENABLED) return
    let cancelled = false
    medicationApi
      .list<Med>()
      .then((remote) => {
        if (!cancelled && Array.isArray(remote)) saveMeds(remote)
      })
      .catch(() => {
        if (!cancelled) notify('FastAPI is offline · showing local demo data')
      })
      .finally(() => {
        if (!cancelled) setMedicationsLoaded(true)
      })
    return () => {
      cancelled = true
    }
  }, [notify, saveMeds])

  useEffect(() => {
    if (!API_ENABLED || !medicationsLoaded || !meds.length) {
      setMedicationSafety([])
      setMedicationRiskPairs([])
      setSafetyLoading(false)
      return
    }
    let cancelled = false
    setSafetyLoading(true)
    setSafetyError(false)
    medicationRiskApi.assess(meds).then((result) => {
      if (cancelled) return
      setMedicationSafety(result.medications)
      setMedicationRiskPairs(result.pairs)
    }).catch(() => {
      if (cancelled) return
      setMedicationSafety([])
      setMedicationRiskPairs([])
      setSafetyError(true)
    }).finally(() => {
      if (!cancelled) setSafetyLoading(false)
    })
    return () => { cancelled = true }
  }, [meds, medicationsLoaded])

  useEffect(() => {
    if (!API_ENABLED) return
    googleCalendarApi.status().then(({ connected, email }) => {
      setGoogleConnected(connected)
      setGoogleEmail(email)
    }).catch(() => undefined)
    const params = new URLSearchParams(window.location.search)
    const googleResult = params.get('google')
    if (googleResult === 'connected') notify('Google Calendar connected')
    if (googleResult === 'error') {
      const reason = params.get('reason')
      const message = reason === 'invalid_client'
        ? 'Google rejected the OAuth client · use the matching Web client ID and client secret from the same Google Cloud project'
        : reason === 'redirect_uri_mismatch'
          ? 'Google redirect URI mismatch · authorize http://localhost:8000/auth/google/callback in the Web client settings'
          : reason === 'invalid_grant'
            ? 'Google authorization code expired or was already used · start Connect again and finish the flow once'
            : reason === 'unauthorized_client'
              ? 'This Google OAuth client is not allowed to use the authorization code flow · create a Web application client'
                : reason === 'invalid_scope'
                  ? 'Google rejected a requested permission · check Calendar API access and the consent screen'
                  : reason === 'insufficient_scope'
                    ? 'Google did not grant Calendar read access · reconnect and approve the Calendar permission'
                  : reason === 'network_error'
                  ? 'MedClock could not reach Google’s token service · check the backend network and retry'
                  : reason === 'token_exchange'
                    ? 'Google returned to MedClock, but token exchange failed · open /api/v1/integrations/google/config on the backend to check its active settings'
        : reason === 'identity_verification'
          ? 'Google sign-in succeeded, but account verification failed · reconnect and check OpenID/email access'
          : reason === 'credential_storage'
            ? 'Google authorized MedClock, but local token storage failed · check backend encryption-key configuration'
            : reason === 'invalid_state'
              ? 'Google authorization expired or was opened out of order · start Connect again from MedClock'
              : 'Google connection failed · check backend settings and try again'
      notify(message)
    }
    if (googleResult) window.history.replaceState({}, '', window.location.pathname)
  }, [notify])

  const syncGoogleCalendar = useCallback(async (planAfterSync = false) => {
    if (!API_ENABLED) return
    setCalendarSyncing(true)
    const start = new Date(month.getFullYear(), month.getMonth(), 1).toISOString()
    const end = new Date(month.getFullYear(), month.getMonth() + 1, 1).toISOString()
    try {
      const result = await googleCalendarApi.sync(start, end)
      setGoogleEvents(result.events)
      notify(`Google Calendar synced · ${result.count} events`)
      if (planAfterSync && scheduleConsentRef.current) {
        setSchedulePlanning(true)
        try {
          setSchedulePlan(await scheduleApi.plan(activeDateRef.current, true))
        } catch (error) {
          notify(error instanceof Error ? error.message : 'AI schedule planning failed')
        } finally { setSchedulePlanning(false) }
      }
    } catch (error) {
      notify(error instanceof Error ? error.message : 'Google Calendar sync failed')
    } finally {
      setCalendarSyncing(false)
    }
  }, [month, notify])

  useEffect(() => {
    if (googleConnected) void syncGoogleCalendar()
  }, [googleConnected, syncGoogleCalendar])

  /* Sheet: escape to dismiss, tab cycles inside, background is inert. */
  useEffect(() => {
    if (!sheetOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeLayer()
        return
      }
      if (event.key !== 'Tab' || !sheetRef.current) return
      const focusable = Array.from(
        sheetRef.current.querySelectorAll<HTMLElement>('button, input, select, textarea, [href]'),
      ).filter((node) => !node.hasAttribute('disabled'))
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [sheetOpen, closeLayer])

  useEffect(() => {
    for (const node of [scrollRef.current, navRef.current]) {
      if (!node) continue
      if (sheetOpen) node.setAttribute('inert', '')
      else node.removeAttribute('inert')
    }
  }, [sheetOpen])

  /* Return focus to whatever opened the sheet once it closes. */
  useEffect(() => {
    if (!sheetOpen) openerRef.current?.focus()
  }, [sheetOpen])

  const saveMedication = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    const name = String(data.get('name') ?? '').trim()
    if (!name) {
      notify('Give the medication a name first')
      return
    }
    const times = [...new Set(data.getAll('times').map((value) => String(value)).filter(Boolean))].sort()
    const med: Med = {
      id: editingMed?.id ?? crypto.randomUUID(),
      name,
      dosage: String(data.get('dosage') ?? '').trim() || '1 dose per time',
      time: times[0] ?? '08:00',
      times: times.length ? times : ['08:00'],
      category: String(data.get('category') ?? 'Prescription') as Med['category'],
      indication: String(data.get('indication') ?? '').trim(),
      approved_shift_minutes: Number(data.get('approved_shift_minutes') ?? 0),
      instructions: String(data.get('instructions') ?? '').trim() || 'Follow label instructions',
      tone: 'violet',
    }
    let apiFailed = false
    try {
      if (API_ENABLED) {
        if (editingMed) await medicationApi.update<Med>(med.id, med)
        else await medicationApi.create(med)
      }
    } catch {
      apiFailed = true
    }
    const updated = editingMed
      ? meds.map((item) => item.id === med.id ? med : item)
      : [...meds, med]
    saveMeds(updated.sort((a, b) => a.time.localeCompare(b.time)))
    if (editingMed) setSelected(med)
    closeLayer()
    notify(apiFailed ? 'FastAPI unavailable · saved locally for this demo' : editingMed ? 'Medication updated' : 'Medication added to your routine')
  }

  const deleteMedication = async (med: Med) => {
    if (!window.confirm(`Delete ${med.name} from your medication list?`)) return
    try {
      if (API_ENABLED) await medicationApi.remove(med.id)
      saveMeds(meds.filter((item) => item.id !== med.id))
      setSelected(null)
      closeLayer()
      notify(`${med.name} deleted`)
    } catch {
      notify('Could not delete this medication · try again when FastAPI is online')
    }
  }

  /* ---------------------------------------------------------------- derived */

  const nowHM = clockFmt.format(now)
  const todayDateKey = localDayKey(now)
  const laHour = Number(new Intl.DateTimeFormat('en-US', { timeZone: LA_TZ, hour: '2-digit', hourCycle: 'h23' }).format(now))
  const greeting = laHour < 12 ? 'Good morning' : laHour < 18 ? 'Good afternoon' : 'Good evening'
  const timezoneLabel = `${timezoneOffsetLabel(LA_TZ, now)} / ${timezoneOffsetLabel(TZ, now)}`

  const sortedMeds = useMemo(() => [...meds].sort((a, b) => a.time.localeCompare(b.time)), [meds])
  const safetyById = useMemo(() => new Map(medicationSafety.map((item) => [item.id, item])), [medicationSafety])
  useEffect(() => {
    if (!API_ENABLED) return
    let cancelled = false
    scheduleApi.list<ScheduleDose>(todayDateKey).then((rows) => { if (!cancelled) setTodaySchedule(rows) }).catch(() => undefined)
    return () => { cancelled = true }
  }, [todayDateKey, medicationsLoaded, meds])
  const todayRowsByMed = useMemo(() => {
    const map = new Map<string, ScheduleDose[]>()
    for (const row of todaySchedule) map.set(row.medication_id, [...(map.get(row.medication_id) ?? []), row])
    return map
  }, [todaySchedule])
  const doseOccurrences = useMemo(() => sortedMeds.flatMap((med) => {
    const times = med.times?.length ? med.times : [med.time]
    return times.map((time, index) => {
      const row = todayRowsByMed.get(med.id)?.find((item) => item.dose_index === index)
      return { med, time: row?.time ?? time, plannedOverride: row?.planned_override ?? false }
    })
  }), [sortedMeds, todayRowsByMed])
  const nextMedication = useMemo(() => doseOccurrences.find((dose) => dose.time >= nowHM) ?? doseOccurrences[0], [doseOccurrences, nowHM])
  const nextMedicationSafety = nextMedication ? safetyById.get(nextMedication.med.id) : undefined
  const nextMedicationHasRisk = nextMedicationSafety?.risk === 'high' || nextMedicationSafety?.risk === 'critical'

  const timelineEvents: TimelineEvent[] = useMemo(
    () =>
      [
        ...baseEvents,
        ...doseOccurrences.map(({ med, time, plannedOverride }) => ({ time, title: med.name, icon: 'pill' as IconName, kind: 'point' as const, med: med.id, safety: safetyById.get(med.id), plannedOverride })),
      ].sort((a, b) => a.time.localeCompare(b.time)),
    [doseOccurrences, safetyById],
  )
  const nowIndex = timelineEvents.filter((event) => event.time <= nowHM).length

  const viewKey = selected ? `detail:${selected.id}` : personalInformationOpen ? 'personal-information' : tab
  const viewScrollKey = viewKey

  useLayoutEffect(() => {
    const node = scrollRef.current
    if (!node) return
    node.scrollTop = scrollMemory.current[viewScrollKey] ?? 0
  }, [viewScrollKey])

  const dateLine = new Intl.DateTimeFormat('en-US', { weekday: 'long', month: 'short', day: 'numeric', timeZone: LA_TZ }).format(now)

  const calendar = useMemo(() => {
    const first = new Date(month.getFullYear(), month.getMonth(), 1)
    const days = new Date(month.getFullYear(), month.getMonth() + 1, 0).getDate()
    const offset = (first.getDay() + 6) % 7
    const cellCount = Math.ceil((offset + days) / 7) * 7
    const cells = Array.from({ length: cellCount }, (_, i) => (i < offset || i >= offset + days ? null : i - offset + 1))
    const today = new Date()
    const isCurrentMonth = today.getFullYear() === month.getFullYear() && today.getMonth() === month.getMonth()
    const todayDay = isCurrentMonth ? today.getDate() : null
    const activeDay = pickedDay ?? todayDay ?? 1
    const weekStart = offset + Math.floor((activeDay - 1) / 7) * 7
    return {
      cells: weekView ? cells.slice(weekStart, weekStart + 7) : cells,
      todayDay,
      activeDay,
      label: new Intl.DateTimeFormat('en', { month: 'long', year: 'numeric' }).format(month),
      activeLabel: new Intl.DateTimeFormat('en', { weekday: 'short', month: 'short', day: 'numeric' }).format(
        new Date(month.getFullYear(), month.getMonth(), activeDay),
      ),
    }
  }, [month, weekView, pickedDay])

  const activeDate = localDayKey(new Date(month.getFullYear(), month.getMonth(), calendar.activeDay))
  scheduleConsentRef.current = scheduleConsent
  activeDateRef.current = activeDate
  useEffect(() => {
    if (!API_ENABLED) return
    let cancelled = false
    setCalendarScheduleDate('')
    scheduleApi.list<ScheduleDose>(activeDate).then((rows) => {
      if (!cancelled) { setCalendarSchedule(rows); setCalendarScheduleDate(activeDate) }
    }).catch(() => { if (!cancelled) { setCalendarSchedule([]); setCalendarScheduleDate(activeDate) } })
    return () => { cancelled = true }
  }, [activeDate, medicationsLoaded, meds])
  const calendarRowsByMed = useMemo(() => {
    const map = new Map<string, ScheduleDose[]>()
    if (calendarScheduleDate === activeDate) for (const row of calendarSchedule) map.set(row.medication_id, [...(map.get(row.medication_id) ?? []), row])
    return map
  }, [calendarSchedule, calendarScheduleDate, activeDate])
  const calendarDoseOccurrences = useMemo(() => sortedMeds.flatMap((med) => {
    const times = med.times?.length ? med.times : [med.time]
    return times.map((time, index) => {
      const row = calendarRowsByMed.get(med.id)?.find((item) => item.dose_index === index)
      return { med, time: row?.time ?? time, plannedOverride: row?.planned_override ?? false }
    })
  }), [sortedMeds, calendarRowsByMed])

  const makeSchedulePlan = async () => {
    if (!scheduleConsent) { notify('Consent is required to share event titles and dose times with DeepSeek.'); return }
    setSchedulePlanning(true)
    setSchedulePlan(null)
    try {
      const result = await scheduleApi.plan(activeDate, true)
      setSchedulePlan(result)
      notify(result.conflicts.length ? `Found ${result.conflicts.length} schedule conflict${result.conflicts.length === 1 ? '' : 's'}` : 'No medication-time conflicts found')
    } catch (error) {
      notify(error instanceof Error ? error.message : 'AI schedule planning failed')
    } finally { setSchedulePlanning(false) }
  }

  const acceptSchedulePlan = async () => {
    if (!schedulePlan?.suggestions.length) return
    setScheduleAccepting(true)
    try {
      const result = await scheduleApi.accept(activeDate, schedulePlan.suggestions)
      setCalendarSchedule(result.schedule)
      setCalendarScheduleDate(activeDate)
      if (activeDate === todayDateKey) setTodaySchedule(result.schedule)
      notify(`${result.accepted} one-day reminder change${result.accepted === 1 ? '' : 's'} saved`)
    } catch (error) {
      notify(error instanceof Error ? error.message : 'Could not save schedule changes')
    } finally { setScheduleAccepting(false) }
  }

  const dayName = (day: number) =>
    new Intl.DateTimeFormat('en', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }).format(
      new Date(month.getFullYear(), month.getMonth(), day),
    )

  const noteBlock = (title: string, body: string, icon: IconName = 'shield', warn = false) => (
    <aside className={`note${warn ? ' is-warn' : ''}`}>
      <span className="note-icon" aria-hidden="true">
        <Icon name={icon} size="sm" />
      </span>
      <div>
        <b>{title}</b>
        <p>{body}</p>
      </div>
    </aside>
  )

  /* ---------------------------------------------------------------- screens */

  const nowMarker = (
    <li className="now-marker" aria-hidden="true">
      <i />
      <span className="rule" />
      <span className="label">Now · {nowHM}</span>
    </li>
  )

  const todayScreen = (
    <div className="view">
      <div className="screen-head">
        <div>
          <p className="overline">{COPY.Today.overline}</p>
          <h1>{greeting}</h1>
        </div>
        <button type="button" className="icon-button accent" onClick={() => openSheet()} aria-label="Add medication">
          <Icon name="plus" size="lg" />
        </button>
      </div>
      <p className="date-line">
        {dateLine} · Los Angeles time / HKT
      </p>

      <section className="telemetry" aria-label="Today at a glance">
        <div className="telemetry-item">
          <span className="telemetry-label">Schedule</span>
          <strong className="telemetry-value">
            {doseOccurrences.length} {doseOccurrences.length === 1 ? 'dose' : 'doses'}
          </strong>
        </div>
        <div className="divider-v" aria-hidden="true" />
        <div className="telemetry-item">
          <span className="telemetry-label">Next up</span>
          <strong className={`telemetry-value${nextMedicationHasRisk ? ' is-risk' : ''}`}>{nextMedication ? nextMedication.med.name : '—'}</strong>
        </div>
        <div className="divider-v" aria-hidden="true" />
        <div className="telemetry-item">
          <span className="telemetry-label">
            <i className="dot is-accent" aria-hidden="true" />
            Zone
          </span>
          <strong className="telemetry-value">{timezoneLabel}</strong>
        </div>
      </section>

      <div className="section-heading">
        <h2 id="timeline-heading">Today’s timeline</h2>
        <span>{timelineEvents.length} entries</span>
      </div>
      <ol className="timeline" aria-labelledby="timeline-heading">
        {timelineEvents.map((event, index) => {
          const linkedMed = event.med ? meds.find((med) => med.id === event.med) : undefined
          const meta = linkedMed
            ? `${linkedMed.dosage} · ${linkedMed.instructions}${event.plannedOverride ? ' · Adjusted today' : ''} · ${safetyLoading ? 'Checking interactions…' : event.safety?.risk_label ?? (safetyError ? 'Risk check unavailable' : 'Not assessed')}`
            : event.kind === 'range'
              ? <>{withDayOffset(event.time)} – {withDayOffset(event.end ?? '')}</>
              : 'Daily routine'
          const body = (
            <>
              <span className="time-col">
                <b>{withDayOffset(`${event.time} HKT`)}</b>
                <small>{withDayOffset(hktToLosAngeles(todayDateKey, event.time))}</small>
                {event.kind === 'range' && <small>↳ {withDayOffset(hktToLosAngeles(todayDateKey, event.end))}</small>}
              </span>
              <span className="node" aria-hidden="true" />
              <span className="event-body">
                <span className="event-title">{event.title}</span>
                <span className="event-meta">{meta}</span>
              </span>
              <span className={`chip${linkedMed ? ` risk-chip risk-${event.safety?.risk ?? 'unknown'}` : ' is-plain'}`}>{linkedMed ? event.safety?.risk_label ?? (safetyLoading ? 'Checking' : 'Review') : 'Plan'}</span>
              <span className="event-icon" aria-hidden="true">
                <Icon name={event.icon} size="md" />
              </span>
            </>
          )
          return (
            <Fragment key={`${event.time}-${event.title}-${index}`}>
              {index === nowIndex && nowMarker}
              <li className="timeline-row">
                {linkedMed ? (
                  <button
                    type="button"
                    className="event-card is-linked"
                    onClick={() => openMedication(linkedMed)}
                    aria-label={`${event.time} ${linkedMed.name}, ${linkedMed.dosage}, ${linkedMed.instructions}. Open details.`}
                  >
                    {body}
                  </button>
                ) : (
                  <div className="event-card is-static">
                    {body}
                  </div>
                )}
              </li>
            </Fragment>
          )
        })}
        {nowIndex >= timelineEvents.length && nowMarker}
      </ol>
      {noteBlock(
        'Safety first',
        'MedClock helps you keep track of your routine. Always follow your prescription and your clinician’s advice.',
      )}
    </div>
  )

  const calendarScreen = (
    <div className="view">
      <div className="screen-head">
        <div>
          <p className="overline">{COPY.Calendar.overline}</p>
          <h1>Calendar</h1>
        </div>
      </div>
      <p className="date-line">{COPY.Calendar.sub}</p>

      <div className="segmented" role="group" aria-label="Calendar range">
        <button type="button" aria-pressed={!weekView} onClick={() => setWeekView(false)}>
          Month
        </button>
        <button type="button" aria-pressed={weekView} onClick={() => setWeekView(true)}>
          Week
        </button>
      </div>

      <div className="month-bar">
        <button
          type="button"
          className="icon-button"
          onClick={() => setMonth(new Date(month.getFullYear(), month.getMonth() - 1, 1))}
          aria-label="Previous month"
        >
          <Icon name="back" size="lg" />
        </button>
        <strong aria-live="polite">{calendar.label}</strong>
        <button
          type="button"
          className="icon-button"
          onClick={() => setMonth(new Date(month.getFullYear(), month.getMonth() + 1, 1))}
          aria-label="Next month"
        >
          <Icon name="next" size="lg" />
        </button>
      </div>

      <section className="calendar-panel" aria-label={calendar.label}>
        <div className="weekdays" aria-hidden="true">
          {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((letter, index) => (
            <span key={index}>{letter}</span>
          ))}
        </div>
        <div className="days">
          {calendar.cells.map((day, index) =>
            day === null ? (
              <span key={`empty-${index}`} className="day is-empty" aria-hidden="true" />
            ) : (
              <button
                key={`day-${day}`}
                type="button"
                className={`day${day === calendar.activeDay ? ' is-selected' : ''}${day === calendar.todayDay ? ' is-today' : ''}`}
                aria-pressed={day === calendar.activeDay}
                aria-current={day === calendar.todayDay ? 'date' : undefined}
                aria-label={dayName(day)}
                onClick={() => setPickedDay(day)}
              >
                {day}
              </button>
            ),
          )}
        </div>
      </section>

      <div className="section-heading">
        <h2>Doses · {calendar.activeLabel}</h2>
        <span>{calendarDoseOccurrences.length} scheduled</span>
      </div>
      <section className="panel">
        {sortedMeds.length === 0 ? (
          <p className="detail-text">No medications yet. Add one to build your daily rhythm.</p>
        ) : (
          <div className="dose-list">
            {calendarDoseOccurrences.map(({ med, time }, index) => (
              <div className="dose-row" key={`${med.id}-${time}`}>
                <span className="dose-index">{String(index + 1).padStart(2, '0')}</span>
                <span className="dose-body">
                  <b>{med.name}</b>
                    <small>{med.dosage}{calendarRowsByMed.get(med.id)?.find((row) => row.time === time)?.planned_override ? ' · Adjusted for this day' : ''}</small>
                </span>
                <span className="dose-time">{time}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      <div className="section-heading">
        <h2>Connections</h2>
        <span>Optional</span>
      </div>
      <section className="tile-row">
        <span className="tile-icon" aria-hidden="true">
          <Icon name="calendar" size="md" />
        </span>
        <span className="tile-body">
          <b>Google Calendar</b>
          <small>Pull classes, shifts and plans next to your doses.</small>
        </span>
        <button
          type="button"
          className="ghost-button"
          disabled={!API_ENABLED || calendarSyncing}
          onClick={() => {
            if (googleConnected) void syncGoogleCalendar(true)
            else if (API_ENABLED) window.location.assign(googleCalendarApi.connectUrl())
            else notify('Start the FastAPI backend and configure its URL to connect Google Calendar')
          }}
        >
          {calendarSyncing ? 'Syncing…' : googleConnected ? 'Sync now' : 'Connect'}
        </button>
      </section>
      {googleConnected && (
        <>
          <p className="date-line">Connected{googleEmail ? ` as ${googleEmail}` : ''} · This month</p>
          <section className="panel dose-list" aria-label="Google Calendar events">
            {googleEvents.length ? googleEvents.map((event) => (
                <div className="dose-row" key={event.id}>
                  <span className="dose-index"><Icon name="calendar" size="sm" /></span>
                  <span className="dose-body">
                    <b>{event.summary}</b>
                    <small>{calendarEventDateLabel(event)}</small>
                  </span>
                  <span className="dose-time calendar-event-interval">{calendarEventIntervalLabel(event)}</span>
                </div>
              )) : <p className="detail-text">No Google Calendar events this month.</p>}
          </section>
        </>
      )}
      {googleConnected && (
        <section className="panel schedule-planner">
          <div className="section-heading">
            <h2>Plan around events</h2>
            <span>{calendar.activeLabel}</span>
          </div>
          <p className="detail-text">DeepSeek coordinates planning tools. Python checks conflicts and eligible times.</p>
          <label className="privacy-consent schedule-consent">
            <input type="checkbox" checked={scheduleConsent} onChange={(event) => setScheduleConsent(event.target.checked)} />
            <span>Send medication names, dose times, approved timing windows, and event titles/times for this date to DeepSeek. No medical history is sent.</span>
          </label>
          <button type="button" className="primary-button is-block" disabled={!API_ENABLED || !scheduleConsent || schedulePlanning || calendarSyncing} onClick={() => void makeSchedulePlan()}>
            {schedulePlanning ? 'Planning…' : 'Plan with DeepSeek'}
          </button>
          {schedulePlan && schedulePlan.date === activeDate && (
            <div className="schedule-plan-results" aria-live="polite">
              <small>Tools: {schedulePlan.tool_calls.length ? schedulePlan.tool_calls.join(' → ') : 'Python safety fallback'}</small>
              {schedulePlan.glucose_guidance.map((item) => (
                <article className="glucose-travel-card" key={item.event_id || item.event}>
                  <b>{item.medications.some((med) => med.classes.toLowerCase().includes('insulin')) ? 'Insulin Notice: Flights & preserve' : 'Glucose & travel timing'}</b>
                  <strong>{item.event}</strong>
                  <small>
                    {item.start === item.end ? travelMomentLabel(item.start) : `${travelMomentLabel(item.start)} – ${travelMomentLabel(item.end)}`}
                    {item.duration_hours !== null ? ` · ${item.duration_hours} hr` : ' · duration unavailable'}
                  </small>
                  <small>Medicines: {item.medications.map((med) => `${med.name} (${med.classes})`).join(', ')}</small>
                  <ul>
                    {item.guidance.map((note) => <li key={note}>{note}</li>)}
                  </ul>
                </article>
              ))}
              {!schedulePlan.conflicts.length ? <p className="detail-text">No dose times overlap busy events.</p> : (
                <>
                  {schedulePlan.alternatives.map((item) => {
                    const suggestion = schedulePlan.suggestions.find((entry) => entry.medication_id === item.medication_id && entry.dose_index === item.dose_index)
                    const conflict = schedulePlan.conflicts.find((entry) => entry.medication_id === item.medication_id && entry.dose_index === item.dose_index)
                    return (
                      <article className="schedule-plan-row" key={`${item.medication_id}-${item.dose_index}`}>
                        <b>{conflict?.medication} · {item.original_time ?? conflict?.original_time}</b>
                        <small>{conflict?.event ? `Overlaps ${conflict.event}` : 'Schedule conflict'}</small>
                        {suggestion ? <p>Suggested: <strong>{suggestion.time}</strong> ({Math.abs(suggestion.shift_minutes)} min {suggestion.shift_minutes < 0 ? 'earlier' : 'later'})</p> : <p>Keep the current time. {item.reason}</p>}
                      </article>
                    )
                  })}
                  {schedulePlan.suggestions.length > 0 && (
                    <button type="button" className="ghost-button is-block" disabled={scheduleAccepting} onClick={() => void acceptSchedulePlan()}>
                      {scheduleAccepting ? 'Saving…' : 'Apply suggested times for this day'}
                    </button>
                  )}
                </>
              )}
            </div>
          )}
        </section>
      )}
      {noteBlock('Nothing is shared yet', 'Connections are opt-in, and this demo keeps every entry on this device.', 'info')}
    </div>
  )

  const medicationsScreen = (
    <div className="view">
      <div className="screen-head">
        <div>
          <p className="overline">{COPY.Medications.overline}</p>
          <h1>Your routine</h1>
        </div>
        <button type="button" className="icon-button accent" onClick={() => openSheet()} aria-label="Add medication">
          <Icon name="plus" size="lg" />
        </button>
      </div>
      <p className="date-line">{COPY.Medications.sub}</p>

      <div className="section-heading">
        <h2>Active medications</h2>
        <span>{meds.length} total</span>
      </div>

      {meds.length === 0 ? (
        <div className="empty-state">
          <Icon name="pill" size="lg" />
          <b>No medications yet</b>
          <p>Add your first entry to see doses appear on today’s timeline.</p>
          <button type="button" className="ghost-button" onClick={() => openSheet()}>
            Add medication
          </button>
        </div>
      ) : (
        sortedMeds.map((med, index) => (
          <button
            type="button"
            className="med-row"
            key={med.id}
            onClick={() => openMedication(med)}
            aria-label={`${med.name}, ${med.dosage}, at ${(med.times?.length ? med.times : [med.time]).join(', ')}. Open details.`}
          >
            <span className="med-index">{String(index + 1).padStart(2, '0')}</span>
            <span className={`pill-mark ${med.tone}`} aria-hidden="true">
              <Icon name="pill" size="md" />
            </span>
            <span className="med-body">
              <span className="med-name">{med.name}</span>
              <span className="med-meta">
                {med.category} · {med.dosage} · {(med.times?.length ? med.times : [med.time]).join(', ')}
              </span>
              {med.indication && <span className="med-meta">For {med.indication}</span>}
              <span className={`med-safety-line risk-text risk-${safetyById.get(med.id)?.risk ?? 'unknown'}`}>
                {safetyLoading ? 'Checking interactions…' : safetyById.get(med.id)?.risk_label ?? (safetyError ? 'Risk check unavailable' : 'Not assessed')}
              </span>
            </span>
            <span className="med-time">{(med.times?.length ?? 1) > 1 ? `${med.times.length}×/day` : med.time}</span>
            <Icon name="next" size="sm" className="med-chevron" />
          </button>
        ))
      )}

      {noteBlock(
        'How medication checks work',
        'MedClock checks combinations against official FDA labels and shows a short leaflet summary. Label data can be incomplete or outdated and must not guide treatment decisions by itself. Yellow/red or incomplete results need a doctor or pharmacist’s advice. Green means no direct conflict evidence was found; it does not prove a combination is safe.',
      )}
    </div>
  )

  const profileScreen = (
    <div className="view">
      <div className="screen-head">
        <div>
          <p className="overline">{COPY.Profile.overline}</p>
          <h1>Your profile</h1>
        </div>
      </div>
      <p className="date-line">{COPY.Profile.sub}</p>

      <button type="button" className="tile-row personal-entry" onClick={openPersonalInformation}>
        <span className="tile-icon" aria-hidden="true"><Icon name="heart-pulse" size="md" /></span>
        <span className="tile-body">
          <b>Personal information</b>
          <small>Set your routine, health history, and medication limits for future AI hints.</small>
        </span>
        <Icon name="next" size="sm" />
      </button>

      <section className="profile-head">
        <span className="avatar" aria-hidden="true">
          M
        </span>
        <div>
          <b>My routine</b>
          <small>Personal schedule · Hong Kong · GMT+8</small>
        </div>
      </section>

      <div className="section-heading">
        <h2>This device</h2>
      </div>
      <dl className="panel data-list">
        <div className="data-row">
          <dt>Medications</dt>
          <dd>{meds.length}</dd>
        </div>
        <div className="data-row">
          <dt>Data source</dt>
          <dd>{API_ENABLED ? 'FastAPI' : 'Local demo'}</dd>
        </div>
        <div className="data-row">
          <dt>Local time</dt>
          <dd>{nowHM} HKT</dd>
        </div>
        <div className="data-row">
          <dt>Theme</dt>
          <dd>Follows system</dd>
        </div>
      </dl>

      <div className="section-heading">
        <h2>Connections</h2>
        <span>Optional</span>
      </div>
      <div className="stack">
        {CONNECTIONS.map((item) => (
          <button
            type="button"
            className="tile-row"
            key={item.title}
            onClick={() => {
              if (item.title === 'Google Calendar' && API_ENABLED) {
                if (googleConnected) void syncGoogleCalendar()
                else window.location.assign(googleCalendarApi.connectUrl())
                return
              }
              notify(`${item.title} · connection setup can be added here`)
            }}
            aria-label={`${item.title}. ${item.desc} ${item.title === 'Google Calendar' && googleConnected ? 'Connected.' : 'Not linked.'}`}
          >
            <span className="tile-icon" aria-hidden="true">
              <Icon name={item.icon} size="md" />
            </span>
            <span className="tile-body">
              <b>{item.title}</b>
              <small>{item.desc}</small>
            </span>
            <span className="status-pill">{item.title === 'Google Calendar' && googleConnected ? 'Connected' : 'Not linked'}</span>
          </button>
        ))}
      </div>

      {noteBlock('Your choice', 'Connections are optional. Your routine stays yours.', 'shield')}
    </div>
  )

  const selectedSafety = selected ? safetyById.get(selected.id) : undefined
  const selectedPairs = selected ? medicationRiskPairs.filter((pair) => pair.drug_a_id === selected.id || pair.drug_b_id === selected.id) : []
  const detailScreen = selected && (
    <div className="view">
      <div className="screen-head">
        <div>
          <p className="overline">Medication details</p>
          <h1>{selected.name}</h1>
        </div>
        <span className={`pill-mark ${selected.tone}`} aria-hidden="true">
          <Icon name="pill" size="lg" />
        </span>
      </div>
      <p className="date-line">Your schedule at a glance.</p>

      <div className="medication-actions">
        <button type="button" className="ghost-button" onClick={() => openSheet(selected)}>Edit medication</button>
        <button type="button" className="danger-button" onClick={() => void deleteMedication(selected)}>Delete</button>
      </div>

      <section className="detail-panel">
        <span className="detail-label">Schedule</span>
        <div className="dose-hero">
          <strong>{selected.times?.join(' · ') ?? selected.time}</strong>
          <span>Daily · {selected.times?.length ?? 1} dose{(selected.times?.length ?? 1) === 1 ? '' : 's'} · Hong Kong time</span>
        </div>
        <p className="detail-text">{selected.dosage}</p>
      </section>

      <section className="detail-panel">
        <span className="detail-label">Category & indication</span>
        <p className="detail-value">{selected.category}</p>
        <p className="detail-text">{selected.indication || 'No indication recorded.'}</p>
      </section>

      <section className="detail-panel">
        <span className="detail-label">Instructions</span>
        <p className="detail-value">{selected.instructions}</p>
      </section>

      <section className="detail-panel safety-panel">
        <span className="detail-label">Interaction check</span>
        {safetyLoading ? <p className="detail-text">Checking the listed medicines against official labels…</p> : safetyError ? (
          <p className="detail-text">The risk check could not reach the backend. No safety conclusion is available.</p>
        ) : selectedSafety ? (
          <>
            <span className={`safety-badge risk-${selectedSafety.risk}`}>{selectedSafety.risk_label}</span>
            <p className="detail-text">
              {selectedSafety.assessment === 'not_assessed'
                ? 'Add your other medicines and supplements to check combinations.'
                : selectedSafety.risk === 'low'
                  ? 'No conflict was found in retrieved labels. This does not confirm safety.'
                  : selectedSafety.consult
                    ? 'Ask a doctor or pharmacist before using this combination. Do not change prescribed doses.'
                    : 'This summary is based only on the listed medicines and the retrieved label data.'}
            </p>
          </>
        ) : <p className="detail-text">No interaction check is available yet.</p>}
        {selectedPairs.map((pair) => (
          <article className={`safety-pair risk-border-${pair.verdict}`} key={`${pair.drug_a_id}:${pair.drug_b_id}`}>
            <b>{pair.drug_a} × {pair.drug_b}</b>
            {pair.findings.length ? pair.findings.slice(0, 3).map((finding) => (
              <div className="finding-detail" key={finding.rule_id}>
                <p><strong>{finding.title}</strong> · {finding.detail}{finding.action ? ` ${finding.action}` : ''}</p>
                {finding.evidence.slice(0, 2).map((evidence) => (
                  <blockquote key={evidence.id}>
                    <span>{evidence.drug} · {evidence.section}: “{evidence.quote}”</span>
                    {evidence.url && <a href={evidence.url} target="_blank" rel="noreferrer">Source ↗</a>}
                  </blockquote>
                ))}
              </div>
            )) : <p>{pair.verdict === 'low' ? 'No direct mention of this pair was found in the retrieved labels; absence of evidence is not proof of safety.' : 'The available official label data was insufficient to assess this pair.'}</p>}
            {pair.limitations.slice(0, 2).map((item, index) => <small key={index}>{item}</small>)}
          </article>
        ))}
      </section>

      <section className="detail-panel leaflet-panel">
        <span className="detail-label">Drug type & quick leaflet</span>
        {selectedSafety?.matched ? (
          <>
            <p className="leaflet-meta"><b>Class</b> · {selectedSafety.drug_classes.join(' · ') || 'Not listed in retrieved data'}</p>
            <p className="leaflet-meta"><b>Class source</b> · {selectedSafety.class_source || 'Not recorded'}</p>
            <p className="leaflet-meta"><b>Active ingredient</b> · {selectedSafety.active_ingredients.join(', ') || 'Not listed in retrieved data'}</p>
            {selectedSafety.leaflet.length ? selectedSafety.leaflet.slice(0, 8).map((item, index) => (
              <article className="leaflet-item" key={`${item.section}-${index}`}>
                <b>{item.section}</b>
                <p>{item.text}</p>
              </article>
            )) : <p className="detail-text">The retrieved label has no concise safety sections to show.</p>}
            {selectedSafety.label_url && <a className="leaflet-source" href={selectedSafety.label_url} target="_blank" rel="noreferrer">Open official label ↗</a>}
            {selectedSafety.label_effective_time && <small className="leaflet-version">Label version · {selectedSafety.label_effective_time}</small>}
          </>
        ) : <p className="detail-text">No official label match yet. Check the spelling or enter the medicine’s generic name; the app will show insufficient evidence until a label is found.</p>}
      </section>

      <section className="detail-panel">
        <span className="detail-label">Reminder</span>
        <p className="detail-text">
          Scheduled doses today: {selected.times?.join(', ') ?? selected.time}. Marking a dose as taken is stored on this device while demo mode is active.
        </p>
      </section>

      <button
        type="button"
        className="primary-button is-block"
        onClick={() => notify(`${selected.name} marked as taken at ${nowHM}`)}
      >
        <span>Mark as taken</span>
        <Icon name="check" size="md" />
      </button>

      {noteBlock(
        'Safety first',
        'Use this tracker as a reminder. Follow your prescription and your clinician’s advice.',
      )}
    </div>
  )

  return (
    <main className="app-frame">
      <div className="status-bar">
        <span>{nowHM} HKT</span>
        <StatusGlyphs />
      </div>

      <header className="topbar">
        {selected ? (
          <>
            <button type="button" className="icon-button" onClick={closeLayer} aria-label="Back to previous screen">
              <Icon name="back" size="lg" />
            </button>
            <span className="topbar-title">{selected.name}</span>
            <span className="topbar-meta">Details</span>
          </>
        ) : (
          <>
            <span className="brand-mark" aria-hidden="true">
              M
            </span>
            <span className="wordmark">MedClock</span>
            <span className="topbar-meta">{API_ENABLED ? 'FastAPI' : 'Local demo'}</span>
            <button type="button" className="avatar-button" onClick={() => setTab('Profile')} aria-label="Open profile">
              <span aria-hidden="true">M</span>
            </button>
          </>
        )}
      </header>

      <div
        className="screen-scroll"
        ref={scrollRef}
        onScroll={() => {
          const node = scrollRef.current
          if (node) scrollMemory.current[viewScrollKey] = node.scrollTop
        }}
      >
        {/* A key on the wrapper replays the entrance animation per view without
            remounting the scroll container, so scroll memory survives. */}
        <Fragment key={viewKey}>
          {selected ? detailScreen : personalInformationOpen ? (
            <PersonalInformationScreen onBack={closeLayer} notify={notify} />
          ) : tab === 'Today' ? todayScreen : tab === 'Calendar' ? calendarScreen : tab === 'Medications' ? medicationsScreen : profileScreen}
        </Fragment>
      </div>

      {!selected && (
        <nav className="bottom-nav" ref={navRef} aria-label="Primary">
          {TABS.map((item) => (
            <button
              type="button"
              className="nav-item"
              key={item}
              aria-current={tab === item ? 'page' : undefined}
              onClick={() => {
                setPersonalInformationOpen(false)
                setTab(item)
              }}
            >
              <Icon name={TAB_ICON[item]} size="lg" />
              <span>{item}</span>
            </button>
          ))}
        </nav>
      )}

      {toast && (
        <div className="toast" role="status" aria-live="polite">
          {toast}
        </div>
      )}

      {sheetOpen && (
        <div className="sheet-backdrop" onClick={closeLayer}>
          <form
            className="sheet"
            key={editingMed?.id ?? 'new-medication'}
            ref={sheetRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="sheet-title"
            onClick={(event) => event.stopPropagation()}
            onSubmit={saveMedication}
          >
            <div className="sheet-grip" aria-hidden="true" />
            <div className="sheet-head">
              <div>
                <p className="overline">Build your routine</p>
                <h2 id="sheet-title">{editingMed ? 'Edit medication' : 'Add medication'}</h2>
              </div>
              <button type="button" className="icon-button" onClick={closeLayer} aria-label="Close without saving">
                <Icon name="close" size="lg" />
              </button>
            </div>

            <label className="field">
              Medication name
              <input name="name" placeholder="e.g. Vitamin D" defaultValue={editingMed?.name ?? ''} required autoFocus autoComplete="off" enterKeyHint="next" />
            </label>

            <label className="field">
              Category
              <select name="category" defaultValue={editingMed?.category ?? 'Prescription'}>
                <option value="Prescription">Prescription medicine</option>
                <option value="OTC">Over-the-counter</option>
                <option value="Supplement">Supplement / vitamin</option>
                <option value="Herbal/Traditional">Herbal / traditional</option>
                <option value="Other">Other</option>
              </select>
            </label>

            <div className="field-row">
              <label className="field">
                Dose each time
                <input name="dosage" placeholder="1 tablet · 500 mg" defaultValue={editingMed?.dosage ?? ''} autoComplete="off" enterKeyHint="next" />
              </label>
            </div>

            <fieldset className="dose-times-field">
              <legend>Times per day</legend>
              <div className="dose-times-list">
                {draftTimes.map((time, index) => (
                  <div className="dose-time-input" key={index}>
                    <label className="field">
                      Dose {index + 1}
                      <input type="time" name="times" value={time} required onChange={(event) => setDraftTimes((current) => current.map((item, itemIndex) => itemIndex === index ? event.target.value : item))} />
                    </label>
                    {draftTimes.length > 1 && <button type="button" className="remove-dose-time" onClick={() => setDraftTimes((current) => current.filter((_, itemIndex) => itemIndex !== index))} aria-label={`Remove dose ${index + 1}`}>−</button>}
                  </div>
                ))}
              </div>
              {draftTimes.length < 8 && <button type="button" className="add-dose-time" onClick={() => setDraftTimes((current) => [...current, ['08:00', '12:00', '18:00', '22:00'][current.length] ?? ''])}>+ Add another dose time</button>}
            </fieldset>

            <label className="field">
              Indication / reason for taking
              <input name="indication" placeholder="e.g. blood pressure, allergy symptoms" defaultValue={editingMed?.indication ?? ''} autoComplete="off" enterKeyHint="next" />
            </label>

            <label className="field">
              Instructions
              <input name="instructions" placeholder="Take with breakfast" defaultValue={editingMed?.instructions ?? ''} autoComplete="off" enterKeyHint="done" />
            </label>

            <label className="field">
              Clinician-approved timing flexibility
              <select name="approved_shift_minutes" defaultValue={editingMed?.approved_shift_minutes ?? 0}>
                <option value="0">Fixed time</option>
                <option value="15">Up to 15 minutes early or late</option>
                <option value="30">Up to 30 minutes early or late</option>
                <option value="60">Up to 1 hour early or late</option>
                <option value="120">Up to 2 hours early or late</option>
              </select>
              <small>Only choose a window confirmed by your prescriber, pharmacist, or label.</small>
            </label>

            <p className="form-note">AI planning never changes dose or frequency. Fixed time is the default.</p>

            <button className="primary-button" type="submit">
              <span>{editingMed ? 'Save changes' : 'Save to my routine'}</span>
              <Icon name="arrow-right" size="md" />
            </button>
          </form>
        </div>
      )}
    </main>
  )
}
