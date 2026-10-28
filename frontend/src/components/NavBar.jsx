import { NavLink } from 'react-router-dom'
import { useEffect, useState } from 'react'
import Logo from './Logo.jsx'
import { fetchHealth } from '../api.js'

export default function NavBar() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const h = await fetchHealth()
        if (!cancelled) setHealth(h)
      } catch { if (!cancelled) setHealth({ ok: false }) }
    }
    tick()
    const id = setInterval(tick, 10000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const linkBase = 'group relative flex items-center gap-2 px-3.5 py-1.5 rounded-md text-[13px] font-medium transition-colors'
  const linkIdle = 'text-ink-200 hover:text-ink-100'
  const linkActive = 'text-ink-100'

  return (
    <header className="relative z-20 border-b border-ink-600/40 bg-ink-950/70 backdrop-blur-lg">
      <div className="max-w-[1400px] mx-auto flex items-center justify-between px-6 py-3.5">
        <div className="flex items-center gap-3">
          <Logo size={30} />
          <div className="leading-tight">
            <div className="font-display text-[15px] font-semibold text-ink-100 tracking-tight">
              TrustGate
            </div>
            <div className="text-[10.5px] text-ink-300 tracking-wider uppercase font-medium">
              Zero-Trust AI Middleware
            </div>
          </div>
        </div>

        <nav className="flex items-center gap-1 p-1 rounded-lg border border-ink-600/50 bg-ink-900/50">
          <NavLink to="/chat" className={({ isActive }) =>
            `${linkBase} ${isActive ? linkActive : linkIdle}`}>
            {({ isActive }) => (
              <>
                {isActive && (
                  <span className="absolute inset-0 rounded-md bg-gradient-to-b from-ink-700/80 to-ink-700/30 border border-ink-500/60 shadow-sm" />
                )}
                <span className="relative">Chat</span>
              </>
            )}
          </NavLink>
          <NavLink to="/monitor" className={({ isActive }) =>
            `${linkBase} ${isActive ? linkActive : linkIdle}`}>
            {({ isActive }) => (
              <>
                {isActive && (
                  <span className="absolute inset-0 rounded-md bg-gradient-to-b from-ink-700/80 to-ink-700/30 border border-ink-500/60 shadow-sm" />
                )}
                <span className="relative">Monitor</span>
              </>
            )}
          </NavLink>
        </nav>

        <div className="flex items-center gap-2.5 text-[11px]">
          <span className="relative flex h-2 w-2">
            <span className={`absolute inset-0 rounded-full ${health?.ok ? 'bg-mint-400' : 'bg-crimson-400'} animate-pulse-soft opacity-70`} />
            <span className={`relative inline-flex rounded-full h-2 w-2 ${health?.ok ? 'bg-mint-500' : 'bg-crimson-500'}`} />
          </span>
          <span className="text-ink-200 font-medium tracking-wide">
            {health?.ok ? 'BACKEND ONLINE' : 'OFFLINE'}
          </span>
          {health?.llm_backend && (
            <span className="text-ink-300 font-mono">· {health.llm_backend}</span>
          )}
        </div>
      </div>
    </header>
  )
}
