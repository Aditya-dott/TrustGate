import { scoreColor } from '../constants'

// Horizontal 0-1 gauge. Gradient fill, optional threshold markers for
// flag (0.55) and block (0.75).
export default function ScoreGauge({
  value, label, showTicks = false, showNumber = true,
  height = 'md',
}) {
  const clamped = Math.max(0, Math.min(1, Number(value) || 0))
  const percent = clamped * 100
  const color = scoreColor(clamped)
  const heightCls = height === 'sm' ? 'h-1.5' : height === 'lg' ? 'h-3' : 'h-2'

  return (
    <div className="w-full">
      {label && (
        <div className="flex items-center justify-between mb-1.5">
          <span className="label-eyebrow">{label}</span>
          {showNumber && (
            <span className="num-display text-[12px]">{clamped.toFixed(3)}</span>
          )}
        </div>
      )}
      <div className={`relative w-full rounded-full bg-ink-800 overflow-hidden ${heightCls}`}>
        <div
          className={`${heightCls} rounded-full transition-all duration-500`}
          style={{
            width: `${percent}%`,
            backgroundColor: color,
            boxShadow: `0 0 10px ${color}80`,
          }}
        />
        {showTicks && (
          <>
            <div className="absolute top-0 bottom-0 w-px bg-sun-500/40" style={{ left: '55%' }} title="Flag @ 0.55" />
            <div className="absolute top-0 bottom-0 w-px bg-crimson-500/50" style={{ left: '75%' }} title="Block @ 0.75" />
          </>
        )}
      </div>
      {!label && showNumber && (
        <div className="mt-1 text-right num-display text-[11px] text-ink-200">
          {clamped.toFixed(3)}
        </div>
      )}
    </div>
  )
}
