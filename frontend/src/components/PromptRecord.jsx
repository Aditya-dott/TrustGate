import { useState } from 'react'
import {
  ChevronDown, ChevronUp, FileText, Shield, Cog, Sparkles, Lock, MessageCircle,
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip } from 'recharts'
import { TierBadge, DecisionBadge, Pill } from './Badge.jsx'
import ScoreGauge from './ScoreGauge.jsx'
import {
  ATTACK_TYPE_LABELS, ATTACK_TYPE_COLORS, PII_TYPE_LABELS,
  formatTimestamp, scoreColor,
} from '../constants.js'

function Section({ icon: Icon, title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="subpanel">
      <button
        className="w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-ink-700/40 transition-colors rounded-t-xl"
        onClick={() => setOpen(o => !o)}
      >
        <div className="flex items-center gap-2">
          <Icon className="w-3 h-3 text-iris-400" />
          <span className="label-section text-ink-200">{title}</span>
        </div>
        {open ? <ChevronUp className="w-3.5 h-3.5 text-ink-300" /> : <ChevronDown className="w-3.5 h-3.5 text-ink-300" />}
      </button>
      {open && <div className="px-4 pb-4 pt-1 space-y-3.5 animate-fade-in">{children}</div>}
    </div>
  )
}

function KV({ label, children }) {
  return (
    <div>
      <div className="label-eyebrow mb-1">{label}</div>
      <div className="text-[12.5px] text-ink-100">{children}</div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Per-chunk mini bar chart
// ---------------------------------------------------------------------------
function ChunkScoreChart({ scores }) {
  if (!scores || scores.length <= 1) return null
  const data = scores.map((s, i) => ({ chunk: `#${i + 1}`, score: s, color: scoreColor(s) }))
  return (
    <div>
      <div className="label-eyebrow mb-1.5">Per-chunk suspicion</div>
      <div className="h-28 rounded-lg bg-ink-950/60 border border-ink-600/40 px-1 py-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 0, right: 8, bottom: 0, left: -20 }}>
            <XAxis dataKey="chunk" stroke="#575d6e" fontSize={9} tickLine={false} axisLine={false} />
            <YAxis domain={[0, 1]} stroke="#575d6e" fontSize={9} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ backgroundColor: 'rgba(14,15,19,0.95)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '6px', fontSize: '10.5px' }}
              itemStyle={{ color: '#e2e6ee' }}
              labelStyle={{ color: '#a1a7b5' }}
              formatter={v => v.toFixed(3)}
            />
            <Bar dataKey="score" radius={[3, 3, 0, 0]}>
              {data.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main record card
// ---------------------------------------------------------------------------
export default function PromptRecord({ record, highlighted, refCallback }) {
  const [expanded, setExpanded] = useState(false)
  const pg = record.prompt_guard_report || {}
  const sd = record.supervisor_decision || {}
  const ps = record.privacy_sieve_report
  const decisionColor =
    record.decision === 'BLOCK' ? '#f85149' :
    record.decision === 'FLAG'  ? '#ffb454' : '#3ecf8e'

  return (
    <div
      ref={refCallback}
      id={`record-${record.id}`}
      className={`relative panel overflow-hidden transition-all ${highlighted ? 'ring-2 ring-iris-500/40' : ''}`}
    >
      {/* Left accent bar showing decision color */}
      <div className="absolute left-0 top-0 bottom-0 w-1" style={{ backgroundColor: decisionColor }} />

      {/* Header (always visible) */}
      <div className="pl-5 pr-4 py-3.5 flex items-center gap-4">
        <div className="flex-shrink-0 w-9 h-9 rounded-lg bg-ink-800 border border-ink-600/60 flex items-center justify-center num-display text-[13px] text-ink-200">
          {record.message_number}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-[13.5px] text-ink-100 truncate leading-tight">
            {record.original_prompt}
          </div>
          <div className="mt-1 flex items-center gap-2 text-[10.5px] text-ink-300">
            <span className="font-mono">{formatTimestamp(record.timestamp)}</span>
            <span className="text-ink-400">·</span>
            <span>Suspicion <span className="num-display text-ink-200">{pg.suspicion_score?.toFixed(3) ?? '—'}</span></span>
            {pg.num_chunks > 1 && (
              <>
                <span className="text-ink-400">·</span>
                <span className="num-display text-ink-200">{pg.num_chunks}</span> chunks
              </>
            )}
            {record.pii_entities_found > 0 && (
              <>
                <span className="text-ink-400">·</span>
                <span className="num-display text-ink-200">{record.pii_entities_found}</span> PII redacted
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <DecisionBadge decision={record.decision} />
          <button
            onClick={() => setExpanded(e => !e)}
            className="focus-ring text-ink-200 hover:text-ink-100 transition-colors"
            aria-label={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Expanded body */}
      {expanded && (
        <div className="pl-5 pr-4 pb-4 space-y-3 animate-fade-in">
          <Section icon={FileText} title="Prompt Information">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <KV label="Session"><span className="font-mono text-[11px]">{record.session_id}</span></KV>
              <KV label="Message #">#{record.message_number}</KV>
              <KV label="Timestamp">{formatTimestamp(record.timestamp)}</KV>
            </div>
            <KV label="Full prompt">
              <div className="mt-1 p-3 rounded-lg bg-ink-950/70 border border-ink-600/40 whitespace-pre-wrap break-words text-[12.5px] leading-relaxed">
                {record.original_prompt}
              </div>
            </KV>
          </Section>

          <Section icon={Shield} title="Prompt Guard Report">
            <ScoreGauge value={pg.suspicion_score} label="Overall suspicion" showTicks />
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <KV label="Top attack class">
                {pg.top_attack_class ? (
                  <Pill label={ATTACK_TYPE_LABELS[pg.top_attack_class] || pg.top_attack_class}
                        color={ATTACK_TYPE_COLORS[pg.top_attack_class]} />
                ) : '—'}
              </KV>
              <KV label="Detected attack type">
                {pg.detected_attack_type ? (
                  <Pill label={ATTACK_TYPE_LABELS[pg.detected_attack_type] || pg.detected_attack_type}
                        color={ATTACK_TYPE_COLORS[pg.detected_attack_type]} />
                ) : <span className="text-ink-300">None</span>}
              </KV>
              <KV label="Attack similarity"><span className="num-display">{pg.attack_similarity?.toFixed(3) ?? '—'}</span></KV>
              <KV label="Deviation score"><span className="num-display">{pg.deviation_score?.toFixed(3) ?? '—'}</span></KV>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <KV label="Confidence"><span className="num-display">{pg.confidence?.toFixed(3) ?? '—'}</span></KV>
              <KV label="Raw attack sim."><span className="num-display">{pg.raw_attack_similarity?.toFixed(3) ?? '—'}</span></KV>
              <KV label="Noise baseline"><span className="num-display">{pg.noise_baseline?.toFixed(3) ?? '—'}</span></KV>
            </div>
            <ChunkScoreChart scores={pg.per_chunk_scores} />
          </Section>

          <Section icon={Cog} title="Supervisor Decision">
            <div className="flex items-center justify-between rounded-lg bg-ink-950/70 border border-ink-600/40 px-3 py-2.5">
              <span className="label-eyebrow">Final decision</span>
              <DecisionBadge decision={record.decision} />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <KV label="Raw combined"><span className="num-display">{sd.raw_combined_score?.toFixed(3) ?? '—'}</span></KV>
              <KV label="Tier penalty">
                <span className="num-display">+{sd.tier_penalty_applied?.toFixed(3) ?? '0.000'}</span>
                {sd.session_tier_at_evaluation && (
                  <span className="ml-1.5 text-[10.5px] text-ink-300">from {sd.session_tier_at_evaluation}</span>
                )}
              </KV>
              <KV label="Final combined"><span className="num-display font-semibold">{sd.final_combined_score?.toFixed(3) ?? '—'}</span></KV>
              <KV label="Session EMA"><span className="num-display">{sd.session_ema_after?.toFixed(3) ?? '—'}</span></KV>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 items-end">
              <KV label="Semantic gate"><span className="num-display">{sd.semantic_gate?.toFixed(3) ?? '—'}</span></KV>
              <KV label="Semantic boost"><span className="num-display">+{sd.semantic_boost_applied?.toFixed(3) ?? '0.000'}</span></KV>
              <KV label="Streak after"><span className="num-display">{sd.consecutive_suspicious_after ?? 0}</span></KV>
              <KV label="Tier after"><TierBadge tier={sd.session_tier_after} size="sm" /></KV>
            </div>
          </Section>

          <Section icon={Sparkles} title="LLM Section" defaultOpen={false}>
            {record.decision === 'BLOCK' ? (
              <div className="rounded-lg bg-crimson-500/5 border border-crimson-500/25 p-3 text-[12.5px] text-crimson-300">
                Prompt never reached the LLM — blocked by the Supervisor.
              </div>
            ) : record.llm_error ? (
              <div className="rounded-lg bg-sun-500/5 border border-sun-500/25 p-3 text-[12.5px] text-sun-300">
                LLM error: {record.llm_error}
              </div>
            ) : (
              <>
                <div className="grid grid-cols-3 gap-3">
                  <KV label="Backend"><span className="font-mono text-[11px]">{record.llm_backend || '—'}</span></KV>
                  <KV label="Model"><span className="font-mono text-[11px]">{record.llm_model || '—'}</span></KV>
                  <KV label="Latency"><span className="num-display">{record.llm_latency_ms} ms</span></KV>
                </div>
                <KV label="Raw LLM response (before Privacy Sieve)">
                  <div className="mt-1 p-3 rounded-lg bg-ink-950/70 border border-ink-600/40 whitespace-pre-wrap break-words text-[12.5px] leading-relaxed max-h-56 overflow-auto">
                    {record.llm_response}
                  </div>
                </KV>
              </>
            )}
          </Section>

          <Section icon={Lock} title="Privacy Sieve Report" defaultOpen={false}>
            {record.decision === 'BLOCK' ? (
              <div className="text-[12.5px] text-ink-300 italic">
                Not applicable — response never generated.
              </div>
            ) : !ps ? (
              <div className="text-[12.5px] text-ink-300 italic">No response to scan.</div>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <span className="label-eyebrow">Status</span>
                  {ps.pii_found
                    ? <Pill label={`${ps.entities?.length || 0} PII redacted`} color="#ffb454" />
                    : <Pill label="Clean · no PII detected" color="#3ecf8e" />}
                </div>
                {ps.entities?.length > 0 && (
                  <div className="rounded-lg border border-ink-600/40 overflow-hidden">
                    <table className="w-full text-[11.5px]">
                      <thead className="bg-ink-800/60 border-b border-ink-600/40">
                        <tr className="text-ink-300 uppercase tracking-wider">
                          <th className="text-left px-3 py-2 font-medium">Type</th>
                          <th className="text-left px-3 py-2 font-medium">Original</th>
                          <th className="text-left px-3 py-2 w-20 font-medium">Position</th>
                          <th className="text-left px-3 py-2 font-medium">Replacement</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ps.entities.map((e, i) => (
                          <tr key={i} className="border-b border-ink-600/30 last:border-0">
                            <td className="px-3 py-2 text-ink-100">{PII_TYPE_LABELS[e.entity_type] || e.entity_type}</td>
                            <td className="px-3 py-2 font-mono text-ink-200">{e.original_text}</td>
                            <td className="px-3 py-2 font-mono text-ink-300">{e.start}–{e.end}</td>
                            <td className="px-3 py-2 font-mono text-sun-300">{e.replacement}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <KV label="Cleaned response">
                  <div className="mt-1 p-3 rounded-lg bg-ink-950/70 border border-ink-600/40 text-[12.5px] leading-relaxed whitespace-pre-wrap break-words max-h-56 overflow-auto">
                    {ps.cleaned_text || record.llm_response}
                  </div>
                </KV>
              </>
            )}
          </Section>

          <Section icon={MessageCircle} title="Final Output (as shown to user)">
            <div className="p-3 rounded-lg bg-ink-950/70 border border-ink-600/40 text-[13px] text-ink-100 whitespace-pre-wrap break-words leading-relaxed">
              {record.final_output}
            </div>
          </Section>
        </div>
      )}
    </div>
  )
}
