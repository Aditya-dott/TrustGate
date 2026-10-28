// Thin fetch wrappers, one per backend endpoint. All requests go
// through Vite's dev-server proxy (see vite.config.js), so the base
// URL is relative and CORS never enters the picture.

const jsonHeaders = { 'Content-Type': 'application/json' }

async function jsonFetch(url, options = {}) {
  const res = await fetch(url, options)
  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      detail = body?.detail || JSON.stringify(body)
    } catch { detail = res.statusText }
    throw new Error(`HTTP ${res.status}: ${detail}`)
  }
  return res.json()
}

export function fetchHealth() {
  return jsonFetch('/api/health')
}
export function createSession() {
  return jsonFetch('/api/session/new', { method: 'POST' })
}
export function sendChatMessage(prompt, sessionId) {
  return jsonFetch('/api/chat', {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ prompt, session_id: sessionId }),
  })
}
export function fetchSessionState(sessionId) {
  return jsonFetch(`/api/session/${encodeURIComponent(sessionId)}/state`)
}
export function fetchRecords(sessionId, sinceId = 0) {
  return jsonFetch(
    `/api/session/${encodeURIComponent(sessionId)}/records?since_id=${sinceId}`
  )
}
export function fetchAnalytics(sessionId) {
  return jsonFetch(`/api/session/${encodeURIComponent(sessionId)}/analytics`)
}
export function fetchRecord(recordId) {
  return jsonFetch(`/api/records/${encodeURIComponent(recordId)}`)
}
export function deleteSession(sessionId) {
  return jsonFetch(`/api/session/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
}

// Session id persisted across page reloads and page switches so the
// chat page and the monitor page look at the same conversation.
const SESSION_KEY = 'trustgate.session_id'

export function getStoredSessionId() {
  return localStorage.getItem(SESSION_KEY)
}
export function storeSessionId(sessionId) {
  localStorage.setItem(SESSION_KEY, sessionId)
}
export function clearStoredSessionId() {
  localStorage.removeItem(SESSION_KEY)
}
export async function ensureSession() {
  let sid = getStoredSessionId()
  if (!sid) {
    const res = await createSession()
    sid = res.session_id
    storeSessionId(sid)
  }
  return sid
}
