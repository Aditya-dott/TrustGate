import { DecisionBadge } from './Badge.jsx'
import { formatTimestamp, truncate, ATTACK_TYPE_LABELS } from '../constants.js'

export default function SessionLogTable({ records, onRowClick, highlightId }) {
  if (!records || records.length === 0) {
    return (
      <div className="panel p-10 text-center">
        <div className="text-[12.5px] text-ink-300">
          No messages yet. Send a message on the Chat page to see records here.
        </div>
      </div>
    )
  }

  return (
    <div className="panel overflow-hidden">
      <div className="px-5 py-3.5 border-b border-ink-600/40 flex items-center justify-between">
        <div>
          <div className="font-display text-[14px] font-semibold text-ink-100">Full session log</div>
          <div className="text-[10.5px] text-ink-300 mt-0.5">Click a row to jump to its full record above</div>
        </div>
        <div className="text-[10.5px] text-ink-300 uppercase tracking-wider">
          <span className="num-display text-ink-100 text-[12px] mr-1">{records.length}</span>
          {records.length === 1 ? 'record' : 'records'}
        </div>
      </div>
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full text-[12px]">
          <thead className="sticky top-0 z-10 bg-ink-900/95 backdrop-blur border-b border-ink-600/40">
            <tr className="text-ink-300 text-[10px] uppercase tracking-[0.12em]">
              <th className="text-left px-4 py-2.5 font-medium w-12">#</th>
              <th className="text-left px-3 py-2.5 font-medium">Prompt</th>
              <th className="text-left px-3 py-2.5 font-medium w-20">Suspicion</th>
              <th className="text-left px-3 py-2.5 font-medium w-24">Decision</th>
              <th className="text-left px-3 py-2.5 font-medium w-36">Attack Type</th>
              <th className="text-left px-3 py-2.5 font-medium w-14">PII</th>
              <th className="text-left px-3 py-2.5 font-medium w-24">Time</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => {
              const highlighted = highlightId === r.id
              return (
                <tr
                  key={r.id}
                  onClick={() => onRowClick?.(r.id)}
                  className={`border-b border-ink-600/30 cursor-pointer transition-colors ${
                    highlighted ? 'bg-iris-500/10' : 'hover:bg-ink-800/40'
                  }`}
                >
                  <td className="px-4 py-2.5 num-display text-ink-300">{r.message_number}</td>
                  <td className="px-3 py-2.5 text-ink-100">{truncate(r.original_prompt, 90)}</td>
                  <td className="px-3 py-2.5 num-display text-ink-100">
                    {r.classifier_score?.toFixed(3)}
                  </td>
                  <td className="px-3 py-2.5"><DecisionBadge decision={r.decision} size="sm" /></td>
                  <td className="px-3 py-2.5 text-ink-200 text-[11.5px]">
                    {r.detected_attack_type
                      ? (ATTACK_TYPE_LABELS[r.detected_attack_type] || r.detected_attack_type)
                      : <span className="text-ink-400">—</span>}
                  </td>
                  <td className="px-3 py-2.5 num-display text-ink-200">{r.pii_entities_found || 0}</td>
                  <td className="px-3 py-2.5 text-ink-300 num-display text-[10.5px]">
                    {formatTimestamp(r.timestamp)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
