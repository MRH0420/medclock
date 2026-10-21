import { useEffect, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { personalInformationApi } from './api'
import type { PersonalInformation } from './api'
import { Icon } from './icons'

type Props = { onBack: () => void; notify: (message: string) => void }
type ProfileDraft = Omit<PersonalInformation, 'birth_year' | 'sex'> & {
  birth_year: number | null
  sex: PersonalInformation['sex'] | ''
}

const blankProfile: ProfileDraft = {
  birth_year: null,
  sex: '',
  pregnancy: null,
  breakfast: null,
  lunch: null,
  dinner: null,
  wake: null,
  sleep: null,
  medical_history: '',
  medication_limits: '',
}

const RHYTHM_FIELDS: { key: 'breakfast' | 'lunch' | 'dinner' | 'wake' | 'sleep'; label: string }[] = [
  { key: 'breakfast', label: 'Breakfast' },
  { key: 'lunch', label: 'Lunch' },
  { key: 'dinner', label: 'Dinner' },
  { key: 'wake', label: 'Wake up' },
  { key: 'sleep', label: 'Bedtime' },
]

function compactBytes(memory: string | null): number {
  return memory ? new TextEncoder().encode(memory).length : 0
}

export function PersonalInformationScreen({ onBack, notify }: Props) {
  const [profile, setProfile] = useState<ProfileDraft>(blankProfile)
  const [memory, setMemory] = useState<string | null>(null)
  const [deepseekReady, setDeepseekReady] = useState(false)
  const [consent, setConsent] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [summaryStatus, setSummaryStatus] = useState('')

  useEffect(() => {
    let cancelled = false
    personalInformationApi.get().then((data) => {
      if (cancelled) return
      setDeepseekReady(data.deepseek_configured)
      if (data.profile) setProfile(data.profile)
      setMemory(data.memory)
      if (data.memory) setSummaryStatus(`AI memory ready · ${data.memory_bytes || compactBytes(data.memory)} bytes`)
    }).catch((reason: unknown) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : 'Could not load saved profile')
    }).finally(() => {
      if (!cancelled) setLoading(false)
    })
    return () => { cancelled = true }
  }, [])

  const update = (key: keyof PersonalInformation, value: string | number | null) => {
    setProfile((current) => {
      const next = { ...current, [key]: value } as ProfileDraft
      if (key === 'sex' && value === '男') next.pregnancy = null
      return next
    })
  }

  const save = async (): Promise<boolean> => {
    if (!profile.birth_year || !profile.sex) {
      setError('Select your birth year and sex before saving.')
      return false
    }
    setSaving(true)
    setError('')
    try {
      const result = await personalInformationApi.save({
        ...profile,
        birth_year: profile.birth_year,
        sex: profile.sex,
      })
      setProfile(result.profile)
      setMemory(result.memory)
      setSummaryStatus(result.memory
        ? `Profile saved · AI memory ready · ${result.memory_bytes} bytes`
        : 'Profile saved encrypted · generate a new AI memory for future hints.')
      return true
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not save Personal information')
      return false
    } finally {
      setSaving(false)
    }
  }

  const saveOnly = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (await save()) notify('Personal information saved on this device')
  }

  const generate = async () => {
    if (!consent) {
      setError('Check the consent box first. Health information is sent to DeepSeek only when you request a summary.')
      return
    }
    if (!(await save())) return
    setSaving(true)
    setError('')
    try {
      const result = await personalInformationApi.summarize()
      setMemory(result.memory)
      setSummaryStatus(`DeepSeek memory saved · ${result.memory_bytes} bytes`)
      notify('Short AI memory saved')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'DeepSeek summary failed; your local profile is still saved')
    } finally {
      setSaving(false)
    }
  }

  const handleText = (key: 'medical_history' | 'medication_limits') => (event: ChangeEvent<HTMLTextAreaElement>) =>
    update(key, event.target.value)

  return (
    <div className="view personal-information">
      <div className="screen-head">
        <div>
          <p className="overline">Your profile</p>
          <h1>Personal information</h1>
        </div>
        <button type="button" className="icon-button" onClick={onBack} aria-label="Back to profile">
          <Icon name="back" size="lg" />
        </button>
      </div>
      <p className="date-line">Add your routine and medication limits.</p>

      {loading ? <section className="panel"><p className="detail-text">Loading saved information…</p></section> : (
        <form className="personal-form" onSubmit={saveOnly}>
          <section className="panel personal-section">
            <p className="overline">ABOUT YOU</p>
            <label className="personal-field">
              Birth year <span>Required</span>
              <select value={profile.birth_year ?? ''} onChange={(event) => update('birth_year', Number(event.target.value) || null)} required>
                <option value="" disabled>Select birth year</option>
                {Array.from({ length: new Date().getFullYear() - 1900 + 1 }, (_, i) => new Date().getFullYear() - i).map((year) => (
                  <option key={year} value={year}>{year}</option>
                ))}
              </select>
            </label>
            <fieldset className="personal-field">
              <legend>Sex <span>Required</span></legend>
              <div className="choice-row" role="radiogroup" aria-label="Sex">
                {(['男', '女'] as const).map((sex) => (
                  <label className={`choice-option${profile.sex === sex ? ' is-selected' : ''}`} key={sex}>
                    <input type="radio" name="sex" value={sex} checked={profile.sex === sex} required onChange={() => update('sex', sex)} />
                    {sex === '男' ? 'Male' : 'Female'}
                  </label>
                ))}
              </div>
            </fieldset>
            {profile.sex === '女' ? (
              <label className="personal-field">
                Pregnancy / lactation <small>Used only for medicine-safety reminders.</small>
                <select value={profile.pregnancy ?? ''} onChange={(event) => update('pregnancy', event.target.value || null)}>
                  <option value="">Choose if applicable</option>
                  <option value="妊娠期">Pregnant</option>
                  <option value="哺乳期">Breastfeeding</option>
                  <option value="否">Neither</option>
                </select>
              </label>
            ) : (
              <div className="personal-field is-muted" aria-live="polite">
                Pregnancy / lactation <small>Not applicable · saved as null</small>
              </div>
            )}
          </section>

          <section className="panel personal-section">
            <p className="overline">YOUR DAILY RHYTHM</p>
            <p className="detail-text">Optional times help fit reminders to your routine.</p>
            <div className="rhythm-grid">
              {RHYTHM_FIELDS.map(({ key, label }) => (
                <label className="personal-field" key={key}>
                  {label}
                  <input type="time" step="1800" value={profile[key] ?? ''} onChange={(event) => update(key, event.target.value || null)} />
                </label>
              ))}
            </div>
          </section>

          <section className="panel personal-section">
            <p className="overline">HEALTH & MEDICATION LIMITS</p>
            <label className="personal-field">
              Past medical history
              <small>Conditions, allergies, and long-term medicines. Write “None” if none; leave blank if unknown.</small>
              <textarea rows={3} maxLength={1200} value={profile.medical_history} onChange={handleText('medical_history')} placeholder="e.g. asthma; penicillin allergy; long-term medication…" />
            </label>
            <label className="personal-field">
              Medication restrictions
              <small>Known contraindications, prior reactions, or clinician instructions only.</small>
              <textarea rows={3} maxLength={1200} value={profile.medication_limits} onChange={handleText('medication_limits')} placeholder="e.g. avoid a medicine per clinician instructions…" />
            </label>
          </section>

          <aside className="note personal-privacy">
            <span className="note-icon" aria-hidden="true"><Icon name="shield" size="sm" /></span>
            <div>
              <b>Private by default</b>
              <p>Answers stay encrypted on this device. DeepSeek receives them only after consent and generation. The short memory is stored locally.</p>
            </div>
          </aside>

          <section className="panel personal-section ai-summary-panel">
            <p className="overline">SHORT AI MEMORY</p>
            <p className="detail-text">Create a short memory for future medication hints.</p>
            {summaryStatus && <p className="persona-status" role="status">{summaryStatus}</p>}
            {!deepseekReady ? (
              <p className="form-note">DeepSeek key is blank. Your profile is saved encrypted; no AI memory is generated. Add <code>DEEPSEEK_API_KEY</code> to <code>backend/.env</code> and restart FastAPI to enable generation.</p>
            ) : (
              <>
                <label className="privacy-consent">
                  <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
                  <span>I agree to send these answers to DeepSeek for this summary only.</span>
                </label>
                <button className="ghost-button is-block" type="button" onClick={() => void generate()} disabled={saving || !consent}>
                  {saving ? 'Creating…' : 'Save & generate short memory'}
                </button>
              </>
            )}
            <button className="primary-button is-block" type="submit" disabled={saving}>
              {saving ? 'Saving…' : 'Save personal information'}
            </button>
          </section>

          {error && <p className="personal-error" role="alert">{error}</p>}
          {memory && <div className="form-note persona-footnote"><p>Memory file: <code>backend/.data/persona.txt</code> · {compactBytes(memory)} bytes</p><blockquote>{memory}</blockquote></div>}
        </form>
      )}
    </div>
  )
}
