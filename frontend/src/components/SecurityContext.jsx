import { Radar, Activity, Shield, Timer, Layers } from 'lucide-react'
import { TierBadge, DecisionBadge, Pill } from './Badge.jsx'
import ScoreGauge from './ScoreGauge.jsx'
import {
  ATTACK_TYPE_LABELS, ATTACK_TYPE_COLORS,
  formatSessionId, formatDuration, scoreColor,
} from '../constants.js'

function Meter({ label, value }) {
  const v = Number(value) || 0
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="label-eyebrow">{label}</span>
        <span className="num-display text-[11px] text-ink-100">{v.toFixed(3)}</span>
      </div>
      <div className="h-1 rounded-full bg-ink-800 overflow-hidden">
        <div
          className="h-1 rounded-full transition-all duration-500"
          style={{
            width: `${Math.min(100, v * 100)}%`,
            background: scoreColor(v),
          }}
        />
      </div>
    </div>
  )
}

function Section({ eyebrow, icon: Icon, children }) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-2.5">
        {Icon && <Icon className="w-3 h-3 text-ink-300" />}
        <span className="label-section text-ink-200">{eyebrow}</span>
      </div>
      {children}
    </div>
  )
}

export default function SecurityContext({ sessionId, tier, state, latest }) {
  const analysis = latest?.analysis

  return (
    <aside className="hidden lg:flex flex-col gap-5 w-[340px] flex-shrink-0">
      {/* Session identifier */}
      <div className="panel p-5">
        <div className="flex items-center gap-2 mb-3">
          <Radar className="w-3 h-3 text-ink-300" />
          <span className="label-section text-ink-200">Session</span>
        </div>
        <div className="font-mono text-[13px] text-ink-100 tracking-wider break-all mb-1">
          {formatSessionId(sessionId)}
        </div>
        <div className="text-[10.5px] text-ink-300 flex items-center gap-1.5">
          <Timer className="w-2.5 h-2.5" />
          Active for {formatDuration(state?.started_at)}
        </div>
      </div>

      {/* Live tier */}
      <div className="panel p-5">
        <Section eyebrow="Session Risk Tier" icon={Shield}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="font-display text-[28px] font-semibold leading-none text-ink-100 mb-1">
                {tier || 'Normal'}
              </div>
              <TierBadge tier={tier || 'Normal'} size="sm" />
            </div>
          </div>
          <Meter label="EMA · exponential moving average" value={state?.ema || 0} />
          <div className="flex items-center justify-between mt-3 text-[10.5px] text-ink-300">
            <span className="uppercase tracking-wider">Streak</span>
            <span className="num-display text-ink-100">
              {state?.consecutive_suspicious || 0} consecutive
            </span>
          </div>
        </Section>
      </div>

      {/* Latest analysis */}
      <div className="panel p-5">
        <Section eyebrow="Latest Pipeline Analysis" icon={Activity}>
          {!analysis ? (
            <div className="text-[12px] text-ink-300 leading-relaxed">
              Send a message to see the classifier scores, semantic signals, and
              decision breakdown here in real time.
            </div>
          ) : (
            <div className="space-y-3.5">
              <div className="flex items-center justify-between">
                <span className="label-eyebrow">Decision</span>
                <DecisionBadge decision={analysis.decision} size="sm" />
              </div>
              {analysis.detected_attack_type && (
                <div className="flex items-center justify-between">
                  <span className="label-eyebrow">Attack type</span>
                  <Pill
                    label={ATTACK_TYPE_LABELS[analysis.detected_attack_type] || analysis.detected_attack_type}
                    color={ATTACK_TYPE_COLORS[analysis.detected_attack_type]}
                  />
                </div>
              )}
              <div className="hairline pt-3 space-y-2.5">
                <Meter label="Classifier suspicion" value={analysis.classifier_score} />
                <Meter label="Combined score" value={analysis.final_combined_score} />
              </div>
            </div>
          )}
        </Section>
      </div>

      {/* Pipeline stages */}
      <div className="panel p-5">
        <Section eyebrow="Pipeline" icon={Layers}>
          <div className="space-y-2 text-[12px] text-ink-200">
            {[
              ['01', 'Prompt Guard', 'Classifier + semantic analyzer'],
              ['02', 'Supervisor', 'Gated combination + tier penalty'],
              ['03', 'LLM', 'Only if not blocked'],
              ['04', 'Privacy Sieve', 'PII detection + masking'],
            ].map(([n, name, sub]) => (
              <div key={n} className="flex items-start gap-3 py-1.5">
                <span className="num-display text-[10.5px] text-ink-300 mt-0.5">{n}</span>
                <div className="min-w-0">
                  <div className="text-ink-100 leading-tight">{name}</div>
                  <div className="text-[10.5px] text-ink-300 leading-tight">{sub}</div>
                </div>
              </div>
            ))}
          </div>
        </Section>
      </div>
    </aside>
  )
}
