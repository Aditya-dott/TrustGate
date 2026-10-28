import { useEffect, useRef, useState } from 'react'
import { AlertCircle, Wifi, WifiOff, Activity } from 'lucide-react'
import SessionOverview from '../components/SessionOverview.jsx'
import AnalyticsCharts from '../components/AnalyticsCharts.jsx'
import PromptRecord from '../components/PromptRecord.jsx'
import SessionLogTable from '../components/SessionLogTable.jsx'
import {
  ensureSession, fetchAnalytics, fetchRecords, fetchSessionState,
} from '../api.js'

const POLL_INTERVAL_MS = 2000

export default function MonitorPage() {
  const [sessionId, setSessionId] = useState(null)
  const [state, setState] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [records, setRecords] = useState([])
  const [error, setError] = useState('')
  const [connected, setConnected] = useState(true)
  const [highlightId, setHighlightId] = useState(null)
  const [lastPoll, setLastPoll] = useState(0)
  const recordRefs = useRef({})

  useEffect(() => {
    let cancelled = false
    ensureSession()
      .then((sid) => { if (!cancelled) setSessionId(sid) })
      .catch((e) => { if (!cancelled) setError(`Backend unreachable: ${e.message}`) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false

    async function poll() {
      try {
        const [s, a, r] = await Promise.all([
          fetchSessionState(sessionId),
          fetchAnalytics(sessionId),
          fetchRecords(sessionId),
        ])
        if (cancelled) return
        setState(s); setAnalytics(a); setRecords(r.records || [])
        setConnected(true); setError('')
        setLastPoll(Date.now())
      } catch (e) {
        if (cancelled) return
        setConnected(false); setError(`Poll failed: ${e.message}`)
      }
    }

    poll()
    const id = setInterval(poll, POLL_INTERVAL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [sessionId])

  function handleJumpToRecord(recordId) {
    setHighlightId(recordId)
    const el = recordRefs.current[recordId]
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setTimeout(() => setHighlightId(null), 2000)
  }

  return (
    <div className="flex-1 max-w-[1400px] mx-auto w-full px-6 py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-display text-[22px] font-semibold text-ink-100 leading-none tracking-tight">
            Monitor
          </h1>
          <div className="mt-1.5 text-[11.5px] text-ink-300">
            Live security operations dashboard
          </div>
        </div>
        <div className="flex items-center gap-4 text-[11px]">
          <div className="flex items-center gap-1.5">
            <Activity className="w-3 h-3 text-ink-300" />
            <span className="text-ink-200">Poll interval</span>
            <span className="num-display text-ink-100 ml-1">2.0s</span>
          </div>
          <div className={`flex items-center gap-1.5 ${connected ? 'text-mint-400' : 'text-crimson-400'}`}>
            {connected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            <span className="uppercase tracking-wider font-medium">
              {connected ? 'Live' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="rounded-lg border border-crimson-500/30 bg-crimson-500/5 p-3 flex items-start gap-2.5 text-[12.5px] text-crimson-300">
          <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
          <div>{error}</div>
        </div>
      )}

      {/* Session overview hero */}
      <SessionOverview state={state} analytics={analytics} sessionId={sessionId} />

      {/* Analytics bento */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="label-section text-ink-200">Live Analytics</div>
            <div className="text-[11px] text-ink-300 mt-0.5">
              Charts refresh automatically as new prompts are processed
            </div>
          </div>
        </div>
        <AnalyticsCharts analytics={analytics} />
      </section>

      {/* Per-prompt records */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="label-section text-ink-200">
              Per-Prompt Analysis ({records.length})
            </div>
            <div className="text-[11px] text-ink-300 mt-0.5">
              Click any card to expand the full pipeline breakdown
            </div>
          </div>
        </div>
        {records.length === 0 ? (
          <div className="panel p-10 text-center">
            <div className="text-[12.5px] text-ink-300">
              No prompts yet. Head to the Chat tab and try a message to see records here.
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {records.map((r) => (
              <PromptRecord
                key={r.id}
                record={r}
                highlighted={highlightId === r.id}
                refCallback={(el) => { recordRefs.current[r.id] = el }}
              />
            ))}
          </div>
        )}
      </section>

      {/* Session log */}
      <section>
        <div className="label-section text-ink-200 mb-3">Session Log</div>
        <SessionLogTable
          records={records}
          onRowClick={handleJumpToRecord}
          highlightId={highlightId}
        />
      </section>

      <div className="h-4" />
    </div>
  )
}
