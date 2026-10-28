// Single source of truth for colors, labels, and formatters. All colors
// come from the ink / iris / mint / sun / ember / crimson palette so
// visual language stays coherent everywhere.

// ---------------------------------------------------------------------------
// Semantic palettes
// ---------------------------------------------------------------------------

export const TIER_COLORS = {
  Normal:   { bg: 'bg-mint-500/10',    text: 'text-mint-400',    border: 'border-mint-500/30',    dot: 'bg-mint-400',    hex: '#3ecf8e' },
  Elevated: { bg: 'bg-sun-500/10',     text: 'text-sun-400',     border: 'border-sun-500/30',     dot: 'bg-sun-400',     hex: '#ffb454' },
  High:     { bg: 'bg-ember-500/10',   text: 'text-ember-400',   border: 'border-ember-500/30',   dot: 'bg-ember-400',   hex: '#ff9558' },
  Critical: { bg: 'bg-crimson-500/10', text: 'text-crimson-400', border: 'border-crimson-500/30', dot: 'bg-crimson-400', hex: '#f85149' },
}

export const DECISION_COLORS = {
  ALLOW: { bg: 'bg-mint-500/10',    text: 'text-mint-400',    border: 'border-mint-500/30',    hex: '#3ecf8e' },
  FLAG:  { bg: 'bg-sun-500/10',     text: 'text-sun-400',     border: 'border-sun-500/30',     hex: '#ffb454' },
  BLOCK: { bg: 'bg-crimson-500/10', text: 'text-crimson-400', border: 'border-crimson-500/30', hex: '#f85149' },
}

export const ATTACK_TYPE_LABELS = {
  prompt_injection:  'Prompt Injection',
  jailbreak:         'Jailbreak',
  data_exfiltration: 'Data Exfiltration',
  model_extraction:  'Model Extraction',
}

export const ATTACK_TYPE_COLORS = {
  prompt_injection:  '#ffb454',
  jailbreak:         '#ff9558',
  data_exfiltration: '#f85149',
  model_extraction:  '#b585ff',
}

export const PII_TYPE_LABELS = {
  EMAIL:         'Email',
  PHONE:         'Phone',
  CREDIT_CARD:   'Credit Card',
  AADHAAR:       'Aadhaar',
  PASSPORT:      'Passport',
  IP_ADDRESS:    'IP Address',
  SENSITIVE_URL: 'Sensitive URL',
  DATE_OF_BIRTH: 'Date of Birth',
  PERSON:        'Person Name',
  ADDRESS:       'Address',
}

// ---------------------------------------------------------------------------
// Score -> color mapping (continuous)
// ---------------------------------------------------------------------------

export function scoreColor(score) {
  if (score >= 0.75) return '#f85149'    // crimson - block territory
  if (score >= 0.55) return '#ffb454'    // sun - flag territory
  if (score >= 0.30) return '#ffc472'    // sun light - warning
  return '#3ecf8e'                        // mint - safe
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

export function formatTimestamp(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleTimeString(undefined, {
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    })
  } catch { return iso }
}

export function formatDateTime(iso) {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch { return iso }
}

export function formatDuration(startIso) {
  if (!startIso) return '0s'
  const start = new Date(startIso).getTime()
  const now = Date.now()
  const secs = Math.max(0, Math.floor((now - start) / 1000))
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  const rem = secs % 60
  if (mins < 60) return `${mins}m ${rem}s`
  const hours = Math.floor(mins / 60)
  return `${hours}h ${mins % 60}m`
}

export function truncate(text, max = 80) {
  if (!text) return ''
  if (text.length <= max) return text
  return text.slice(0, max) + '…'
}

// Session IDs come from the backend as "session-abcdef123456". Break them
// into readable groups for a more distinctive presentation.
export function formatSessionId(sid) {
  if (!sid) return '—'
  const clean = sid.replace(/^session-/, '')
  const chunks = clean.match(/.{1,4}/g) || [clean]
  return chunks.join('·').toUpperCase()
}
