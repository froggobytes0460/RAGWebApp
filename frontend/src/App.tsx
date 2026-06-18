import { Routes, Route } from 'react-router-dom'
import { RedirectPage } from './pages/RedirectPage'
import { ChatPage } from './pages/ChatPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<RedirectPage />} />
      <Route path="/chat/:sessionId" element={<ChatPage />} />
    </Routes>
  )
}

export default App
