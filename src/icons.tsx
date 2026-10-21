/**
 * One icon family, one geometry: 24×24 box, 1.6 stroke, round caps and joins.
 * Every structural icon in the app comes from here — no emoji, no text glyphs —
 * so weight, corner treatment and optical size stay consistent everywhere.
 */
import type { ReactElement } from 'react'

export type IconName =
  | 'today' | 'calendar' | 'pill' | 'profile'
  | 'plus' | 'back' | 'next' | 'close' | 'check' | 'arrow-right' | 'arrow-down'
  | 'sun' | 'cap' | 'utensils' | 'activity' | 'moon'
  | 'shield' | 'link' | 'heart-pulse' | 'map-pin' | 'clock' | 'info'

const GLYPHS: Record<IconName, ReactElement> = {
  today: (<><path d="M3 10.6 12 3.2l9 7.4" /><path d="M5.6 9.3V20.8h12.8V9.3" /><path d="M9.9 20.8v-5.6h4.2v5.6" /></>),
  calendar: (<><rect x="3.2" y="5.2" width="17.6" height="15.6" rx="2" /><path d="M3.2 10.2h17.6M8 3.2v4M16 3.2v4" /></>),
  pill: (<><rect x="3" y="8.5" width="18" height="7" rx="3.5" transform="rotate(-45 12 12)" /><path d="m8.6 8.6 6.8 6.8" /></>),
  profile: (<><circle cx="12" cy="8.2" r="3.6" /><path d="M4.6 20.6a7.4 7.4 0 0 1 14.8 0" /></>),
  plus: (<path d="M12 5.2v13.6M5.2 12h13.6" />),
  back: (<path d="M14.4 6.4 8.8 12l5.6 5.6" />),
  next: (<path d="m9.6 6.4 5.6 5.6-5.6 5.6" />),
  close: (<path d="M6.4 6.4l11.2 11.2M17.6 6.4 6.4 17.6" />),
  check: (<path d="m4.8 12.6 4.9 4.9L19.4 6.9" />),
  'arrow-right': (<path d="M4.4 12h15.2M13.6 5.9l6.1 6.1-6.1 6.1" />),
  'arrow-down': (<path d="M12 4.4v15.2M5.9 13.6l6.1 6.1 6.1-6.1" />),
  sun: (<><circle cx="12" cy="12" r="4" /><path d="M12 2.6v2.3M12 19.1v2.3M2.6 12h2.3M19.1 12h2.3M5.3 5.3l1.7 1.7M17 17l1.7 1.7M18.7 5.3 17 7M7 17l-1.7 1.7" /></>),
  cap: (<><path d="m12 3.6 8.8 4.4-8.8 4.4-8.8-4.4z" /><path d="M6.8 10.2v5.6c0 1.4 2.3 2.6 5.2 2.6s5.2-1.2 5.2-2.6v-5.6" /><path d="M20.8 8v5.4" /></>),
  utensils: (<><path d="M6 3.2v6.4a2.4 2.4 0 0 0 4.8 0V3.2" /><path d="M8.4 12.2v8.6" /><path d="M17.6 20.8V3.2c1.8 1.4 2.8 3.4 2.8 5.6 0 2.2-1 3.8-2.8 4.4" /></>),
  activity: (<path d="M3.2 12h3.6l2.4-6.2 4.2 13 2.4-6.8h5" />),
  moon: (<path d="M20.2 14.6A8.6 8.6 0 0 1 9.4 3.8a8.6 8.6 0 1 0 10.8 10.8z" />),
  shield: (<><path d="M12 3.2 5.4 5.9v5.4c0 4.2 2.7 7.4 6.6 9.1 3.9-1.7 6.6-4.9 6.6-9.1V5.9z" /><path d="m9.2 11.8 2.1 2.1 3.9-4.2" /></>),
  link: (<><path d="M10.4 13.6a3.8 3.8 0 0 0 5.4 0l2.6-2.6a3.8 3.8 0 1 0-5.4-5.4l-1.5 1.5" /><path d="M13.6 10.4a3.8 3.8 0 0 0-5.4 0l-2.6 2.6a3.8 3.8 0 1 0 5.4 5.4l1.5-1.5" /></>),
  'heart-pulse': (<><path d="M12 20.4C12 20.4 3.6 15.2 3.6 9.4a4.7 4.7 0 0 1 8.4-2.9 4.7 4.7 0 0 1 8.4 2.9c0 5.8-8.4 11-8.4 11z" /><path d="M6.6 11.6h2.7l1.3-2.4 1.9 4.6 1.3-2.2h3.1" /></>),
  'map-pin': (<><path d="M12 21.2s6.8-5.6 6.8-11a6.8 6.8 0 1 0-13.6 0c0 5.4 6.8 11 6.8 11z" /><circle cx="12" cy="10.2" r="2.5" /></>),
  clock: (<><circle cx="12" cy="12" r="8.4" /><path d="M12 7.4V12l3.2 2" /></>),
  info: (<><circle cx="12" cy="12" r="8.4" /><path d="M12 11.2v5.4" /><path d="M12 7.6h.01" /></>),
}

type IconSize = 'sm' | 'md' | 'lg'

export function Icon({ name, size = 'md', className }: { name: IconName; size?: IconSize; className?: string }) {
  return (
    <svg
      className={`icon icon-${size}${className ? ` ${className}` : ''}`}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {GLYPHS[name]}
    </svg>
  )
}

/** Decorative status-bar marks (signal / Wi-Fi / battery) — same family, filled tips. */
export function StatusGlyphs() {
  return (
    <span className="status-icons" aria-hidden="true">
      <svg viewBox="0 0 24 24" className="icon icon-sm" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        <path d="M4 19v-3.4M9.3 19v-6.6M14.7 19V8.6M20 19V4.6" />
      </svg>
      <svg viewBox="0 0 24 24" className="icon icon-sm" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
        <path d="M2.8 8.9a13.4 13.4 0 0 1 18.4 0M6.4 12.4a8.4 8.4 0 0 1 11.2 0M9.6 15.6a3.8 3.8 0 0 1 4.8 0" />
        <path d="M12 18.9h.01" strokeWidth="2.2" />
      </svg>
      <svg viewBox="0 0 24 24" className="icon icon-sm" fill="none" stroke="currentColor" strokeWidth="1.6">
        <rect x="2.4" y="7.6" width="17.2" height="8.8" rx="2.2" />
        <path d="M21.6 10.6v2.8" strokeLinecap="round" />
        <rect x="4.4" y="9.6" width="10.8" height="4.8" rx="1" fill="currentColor" stroke="none" />
      </svg>
    </span>
  )
}
