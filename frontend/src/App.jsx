import { Routes, Route, Navigate } from 'react-router-dom'
import NavBar from './components/NavBar.jsx'
import ChatPage from './pages/ChatPage.jsx'
import MonitorPage from './pages/MonitorPage.jsx'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <NavBar />
      <main className="flex-1 flex flex-col relative">
        {/* Ambient grid backdrop */}
        <div className="absolute inset-0 grid-pattern pointer-events-none opacity-40" />
        <div className="relative flex-1 flex flex-col">
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/monitor" element={<MonitorPage />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
