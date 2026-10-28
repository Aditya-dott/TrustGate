import { ShieldAlert, CheckCircle2, Flag, AlertTriangle, Sparkles } from 'lucide-react'
import { ATTACK_TYPE_LABELS } from '../constants'

// User's own message.
function UserBubble({ text }) {
  return (
    <div className="flex justify-end animate-slide-up">
      <div className="max-w-[75%] relative">
        <div className="rounded-2xl rounded-tr-md bg-gradient-to-br from-iris-500/15 to-iris-600/10 border border-iris-500/25 px-4 py-2.5 text-[14px] leading-relaxed text-ink-100">
          <div className="whitespace-pre-wrap break-words">{text}</div>
        </div>
      </div>
    </div>
  )
}

// Allowed / flagged assistant response.
function AssistantBubble({ text, decision, piiFound }) {
  const isFlag = decision === 'FLAG'
  return (
    <div className="flex justify-start animate-slide-up">
      <div className="flex gap-2.5 max-w-[85%]">
        <div className="flex-shrink-0 w-7 h-7 rounded-md bg-gradient-to-br from-iris-400/20 to-sky-500/20 border border-iris-500/25 flex items-center justify-center mt-0.5">
          <Sparkles className="w-3.5 h-3.5 text-iris-300" />
        </div>
        <div className={`min-w-0 relative rounded-2xl rounded-tl-md border px-4 py-2.5 text-[14px] leading-relaxed ${
          isFlag
            ? 'bg-sun-500/5 border-sun-500/30 text-ink-100'
            : 'bg-ink-800/60 border-ink-600/50 text-ink-100'
        }`}>
          {isFlag && (
            <div className="flex items-center gap-1.5 mb-2 text-[10.5px] text-sun-400 uppercase tracking-wider font-medium">
              <Flag className="w-3 h-3" />
              Response flagged for review
            </div>
          )}
          <div className="whitespace-pre-wrap break-words">{text}</div>
          {piiFound > 0 && (
            <div className="mt-2.5 pt-2 flex items-center gap-1.5 text-[10.5px] text-ink-200 border-t border-ink-600/40">
              <CheckCircle2 className="w-3 h-3 text-mint-400" />
              <span className="uppercase tracking-wider">
                {piiFound} PII {piiFound === 1 ? 'entity' : 'entities'} redacted
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Blocked response.
function BlockedBubble({ attackType }) {
  const pretty = ATTACK_TYPE_LABELS[attackType] || 'Malicious Content'
  return (
    <div className="flex justify-start animate-slide-up">
      <div className="flex gap-2.5 max-w-[85%]">
        <div className="flex-shrink-0 w-7 h-7 rounded-md bg-crimson-500/15 border border-crimson-500/30 flex items-center justify-center mt-0.5">
          <ShieldAlert className="w-3.5 h-3.5 text-crimson-400" />
        </div>
        <div className="min-w-0 relative rounded-2xl rounded-tl-md border border-crimson-500/30 bg-crimson-500/5 px-4 py-2.5">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10.5px] uppercase tracking-[0.14em] text-crimson-400 font-semibold">
              Blocked · {pretty}
            </span>
          </div>
          <div className="text-[13.5px] text-ink-100 leading-relaxed">
            This prompt was intercepted by TrustGate before it reached the AI model.
          </div>
          <div className="mt-1.5 text-[11.5px] text-ink-200">
            Open the Monitor tab for the full pipeline analysis.
          </div>
        </div>
      </div>
    </div>
  )
}

function ErrorBubble({ text }) {
  return (
    <div className="flex justify-start animate-slide-up">
      <div className="max-w-[75%] rounded-2xl rounded-tl-md bg-ink-800/60 border border-ink-600/50 px-4 py-2.5">
        <div className="flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-sun-400 mt-0.5 flex-shrink-0" />
          <div className="text-[13.5px] text-ink-200 leading-relaxed">{text}</div>
        </div>
      </div>
    </div>
  )
}

export default function ChatBubble({ message }) {
  if (message.role === 'user') return <UserBubble text={message.text} />
  if (message.decision === 'BLOCK') return <BlockedBubble attackType={message.attackType} />
  if (message.error) return <ErrorBubble text={message.text} />
  return <AssistantBubble text={message.text} decision={message.decision} piiFound={message.piiFound || 0} />
}
