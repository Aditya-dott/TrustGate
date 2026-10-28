import { useEffect, useState } from 'react'
import { CheckCircle2, Flag, ShieldOff, Zap, Clock, Radar } from 'lucide-react'
import { TierBadge } from './Badge.jsx'
import ScoreGauge from './ScoreGauge.jsx'
import { formatDuration, formatSessionId, TIER_COLORS, formatDateTime } from '../constants.js'

function Metric({ eyebrow, value, sublabel, icon: Icon, tone = 'text-ink-100' }) {
  return (
    <div className="flex-1 min-w-0 rounded-xl border border-ink-600/50 bg-ink-800/40 px-4 py-3.5">
      <div className="flex items-center gap-1.5 label-eyebrow">
        {Icon && <Icon className="w-3 h-3" />}
        {eyebrow}
      </div>
      <div className={`mt-1.5 font-display text-[26px] font-semibold leading-none num-display ${tone}`}>
        {value}
      </div>
      {sublabel && (
        <div className="mt-1 text-[10.5px] text-ink-300 leading-tight">
          {sublabel}
        </div>
      )}
    </div>
  )
}

export default function SessionOverview({ state, analytics, sessionId }) {
  // Re-render duration once per second
  const [, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 1000)
    return () => clearInterval(id)
  }, [])

  const tier = state?.tier || 'Normal'
  const c = TIER_COLORS[tier]
  const ema = state?.ema || 0
  const total = analytics?.total_messages || 0
  const streak = state?.consecutive_suspicious || 0
  const dc = analytics?.decision_counts || {}

  return (
    <section className="panel overflow-hidden">
      <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr]">
        {/* LEFT — big tier hero */}
        <div className={`relative p-6 border-r border-ink-600/40 ${c.bg}`}>
          <div className="absolute inset-0 opacity-10 pointer-events-none"
               style={{ background: `radial-gradient(circle at 30% 30%, ${c.hex}, transparent 65%)` }} />
          <div className="relative">
            <div className="flex items-center gap-2 mb-3">
              <Radar className="w-3 h-3 text-ink-200" />
              <span className="label-section text-ink-100">Session Overview</span>
            </div>
            <div className="text-[10.5px] font-mono tracking-wider text-ink-200 mb-1">
              {formatSessionId(sessionId)}
            </div>
            <div className="flex items-center gap-1.5 text-[10.5px] text-ink-200 mb-6">
              <Clock className="w-2.5 h-2.5" />
              Started {formatDateTime(state?.started_at)} · Active {formatDuration(state?.started_at)}
            </div>

            <div className="label-eyebrow mb-1">Risk Tier</div>
            <div className={`font-display text-[46px] font-semibold leading-none mb-3 ${c.text}`}>
              {tier}
            </div>
            <TierBadge tier={tier} size="md" />

            <div className="mt-5">
              <ScoreGauge value={ema} label="Session EMA · risk accumulator" showTicks height="lg" />
            </div>

            <div className="mt-5 flex items-center justify-between text-[10.5px] text-ink-200">
              <div>
                <div className="label-eyebrow">Streak</div>
                <div className="num-display text-ink-100 text-[16px] font-semibold mt-0.5">
                  {streak}
                  <span className="text-ink-300 font-normal text-[10.5px] ml-1.5 uppercase tracking-wider">
                    consecutive
                  </span>
                </div>
              </div>
              <div className="text-right">
                <div className="label-eyebrow">Message #</div>
                <div className="num-display text-ink-100 text-[16px] font-semibold mt-0.5">
                  {state?.message_count || 0}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT — metric grid */}
        <div className="p-6 flex flex-col justify-between">
          <div>
            <div className="label-section text-ink-200 mb-4">Live Counters</div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <Metric
                eyebrow="Messages"
                value={total}
                sublabel="processed this session"
              />
              <Metric
                eyebrow="Allowed"
                value={dc.ALLOW || 0}
                tone="text-mint-400"
                icon={CheckCircle2}
                sublabel="reached the LLM"
              />
              <Metric
                eyebrow="Flagged"
                value={dc.FLAG || 0}
                tone="text-sun-400"
                icon={Flag}
                sublabel="allowed with warning"
              />
              <Metric
                eyebrow="Blocked"
                value={dc.BLOCK || 0}
                tone="text-crimson-400"
                icon={ShieldOff}
                sublabel="never reached the LLM"
              />
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="rounded-xl border border-ink-600/50 bg-ink-800/40 px-4 py-3.5 flex items-center justify-between">
              <div>
                <div className="label-eyebrow flex items-center gap-1.5">
                  <Zap className="w-3 h-3" />
                  Escalation streak
                </div>
                <div className="text-[10.5px] text-ink-300 mt-0.5">
                  Consecutive scores ≥ 0.4
                </div>
              </div>
              <div className="num-display text-[24px] font-semibold text-ink-100">
                {streak}
              </div>
            </div>
            <div className="col-span-2 rounded-xl border border-iris-500/25 bg-iris-500/5 px-4 py-3.5">
              <div className="label-eyebrow text-iris-300">Live · Auto-polling every 2s</div>
              <div className="mt-1 text-[12px] text-ink-200 leading-relaxed">
                All charts and records below refresh automatically as new prompts
                are processed on the Chat page.
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
