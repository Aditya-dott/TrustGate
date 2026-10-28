import {
  PieChart, Pie, Cell, ResponsiveContainer,
  AreaChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  BarChart, Bar,
} from 'recharts'
import {
  DECISION_COLORS, ATTACK_TYPE_LABELS, ATTACK_TYPE_COLORS,
} from '../constants'

const tooltipStyle = {
  backgroundColor: 'rgba(14,15,19,0.95)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: '8px',
  fontSize: '11px',
  boxShadow: '0 12px 28px -12px rgba(0,0,0,0.6)',
  padding: '8px 10px',
}
const tooltipItem = { color: '#e2e6ee' }
const tooltipLabel = { color: '#a1a7b5', marginBottom: 4 }

function ChartCard({ title, subtitle, children }) {
  return (
    <div className="panel p-5">
      <div className="mb-4">
        <div className="font-display text-[14px] font-semibold text-ink-100 leading-tight">
          {title}
        </div>
        {subtitle && (
          <div className="text-[10.5px] text-ink-300 mt-1 leading-tight">
            {subtitle}
          </div>
        )}
      </div>
      {children}
    </div>
  )
}

function Empty({ message }) {
  return (
    <div className="flex items-center justify-center text-ink-300 text-[12px]" style={{ height: 200 }}>
      {message}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 1. Decision distribution donut
// ---------------------------------------------------------------------------
function DecisionDonut({ decisionCounts }) {
  const entries = ['ALLOW', 'FLAG', 'BLOCK']
    .map(k => ({ name: k, value: decisionCounts[k] || 0 }))
  const filtered = entries.filter(d => d.value > 0)
  const total = entries.reduce((s, e) => s + e.value, 0)

  if (total === 0) return <Empty message="No messages yet" />

  return (
    <div className="flex items-center gap-6">
      <div className="relative" style={{ width: 180, height: 180 }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={filtered}
              dataKey="value"
              nameKey="name"
              innerRadius="62%"
              outerRadius="90%"
              paddingAngle={3}
              strokeWidth={0}
            >
              {filtered.map((entry) => (
                <Cell key={entry.name} fill={DECISION_COLORS[entry.name].hex} />
              ))}
            </Pie>
            <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipItem} labelStyle={tooltipLabel} />
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <div className="num-display text-[28px] font-semibold text-ink-100 leading-none">
            {total}
          </div>
          <div className="text-[10px] text-ink-300 uppercase tracking-wider mt-1">
            Total
          </div>
        </div>
      </div>
      <div className="space-y-3">
        {entries.map(e => (
          <div key={e.name} className="flex items-center gap-3">
            <span className="inline-block w-2.5 h-2.5 rounded-sm flex-shrink-0"
              style={{ backgroundColor: DECISION_COLORS[e.name].hex }} />
            <span className="text-[12px] uppercase tracking-wider text-ink-200 font-medium w-14">
              {e.name}
            </span>
            <span className="num-display text-[16px] font-semibold text-ink-100">{e.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// 2. Score timeline with EMA overlay
// ---------------------------------------------------------------------------
function ScoreTimeline({ timeline }) {
  if (!timeline || timeline.length === 0) return <Empty message="No messages yet" />
  const data = timeline.map(t => ({
    msg: `#${t.message_number}`,
    combined: t.final_combined_score,
    ema: t.session_ema,
  }))
  return (
    <div style={{ width: '100%', height: 220 }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -10 }}>
          <defs>
            <linearGradient id="gradCombined" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#7c73f0" stopOpacity={0.45} />
              <stop offset="100%" stopColor="#7c73f0" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="2 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis
            dataKey="msg" stroke="#6b7280" fontSize={10} tickLine={false}
            axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
          />
          <YAxis
            domain={[0, 1]} stroke="#6b7280" fontSize={10} tickLine={false}
            axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
            tickFormatter={v => v.toFixed(1)}
          />
          <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipItem} labelStyle={tooltipLabel}
            formatter={v => v?.toFixed(3)} />
          <Area
            type="monotone" dataKey="combined" stroke="#7c73f0" strokeWidth={2}
            fill="url(#gradCombined)" name="Combined score" dot={{ r: 3, fill: '#7c73f0', strokeWidth: 0 }}
          />
          <Line
            type="monotone" dataKey="ema" stroke="#5eb8ff" strokeWidth={1.5}
            strokeDasharray="4 3" name="Session EMA" dot={{ r: 2, fill: '#5eb8ff', strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// 3. Attack types horizontal bar
// ---------------------------------------------------------------------------
function AttackTypeBar({ attackCounts }) {
  const entries = Object.entries(attackCounts || {})
    .filter(([, c]) => c > 0)
    .map(([k, v]) => ({
      type: ATTACK_TYPE_LABELS[k] || k,
      color: ATTACK_TYPE_COLORS[k] || '#a1a7b5',
      count: v,
    }))
  if (entries.length === 0) return <Empty message="No attacks detected yet" />

  const chartHeight = Math.max(120, entries.length * 40 + 30)
  return (
    <div style={{ width: '100%', height: chartHeight }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={entries} layout="vertical" margin={{ left: 10, right: 20, top: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="2 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
          <XAxis type="number" stroke="#6b7280" fontSize={10} tickLine={false}
            axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} allowDecimals={false} />
          <YAxis type="category" dataKey="type" stroke="#a1a7b5" fontSize={11}
            tickLine={false} axisLine={false} width={120} />
          <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipItem} labelStyle={tooltipLabel} />
          <Bar dataKey="count" radius={[0, 5, 5, 0]} barSize={18}>
            {entries.map((e, i) => <Cell key={i} fill={e.color} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Layout — simple 2-col grid, timeline full width below
// ---------------------------------------------------------------------------
export default function AnalyticsCharts({ analytics }) {
  const timeline = analytics?.score_timeline || []
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <ChartCard title="Decision Distribution" subtitle="Allow · Flag · Block breakdown">
        <DecisionDonut decisionCounts={analytics?.decision_counts || {}} />
      </ChartCard>
      <ChartCard title="Attack Types Detected" subtitle="Distribution across blocked and flagged prompts">
        <AttackTypeBar attackCounts={analytics?.attack_type_counts || {}} />
      </ChartCard>
      <div className="lg:col-span-2">
        <ChartCard title="Score & EMA Timeline" subtitle="Combined score per message with session EMA overlay">
          <ScoreTimeline timeline={timeline} />
        </ChartCard>
      </div>
    </div>
  )
}
