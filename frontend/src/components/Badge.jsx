import { TIER_COLORS, DECISION_COLORS } from '../constants'

// TIER: pill with a live dot and monospace label.
export function TierBadge({ tier, size = 'md' }) {
  const c = TIER_COLORS[tier] || TIER_COLORS.Normal
  const sizeCls =
    size === 'lg' ? 'px-3 py-1.5 text-[12px]' :
    size === 'sm' ? 'px-2 py-0.5 text-[10px]' :
                    'px-2.5 py-1 text-[11px]'
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border ${c.bg} ${c.text} ${c.border} ${sizeCls} font-medium tracking-wide uppercase`}>
      <span className="relative flex h-1.5 w-1.5">
        <span className={`absolute inset-0 rounded-full ${c.dot} animate-pulse-soft opacity-70`} />
        <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${c.dot}`} />
      </span>
      {tier}
    </span>
  )
}

// DECISION: chunkier, more editorial. Uses caret-style bracket accents.
export function DecisionBadge({ decision, size = 'md' }) {
  const c = DECISION_COLORS[decision] || DECISION_COLORS.ALLOW
  const sizeCls =
    size === 'lg' ? 'px-3.5 py-1.5 text-[12px]' :
    size === 'sm' ? 'px-2 py-0.5 text-[10px]' :
                    'px-3 py-1 text-[11px]'
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md border font-semibold uppercase tracking-[0.14em] ${c.bg} ${c.text} ${c.border} ${sizeCls}`}>
      {decision}
    </span>
  )
}

// Attack / PII pill: takes an arbitrary color.
export function Pill({ label, color = '#a1a7b5', icon }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10.5px] font-medium tracking-wide"
      style={{
        color,
        backgroundColor: color + '15',
        border: `1px solid ${color}30`,
      }}
    >
      {icon}
      {label}
    </span>
  )
}
