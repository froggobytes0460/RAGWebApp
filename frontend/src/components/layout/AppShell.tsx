import { useState } from 'react'
import { FolderOpen } from 'lucide-react'
import { SessionSidebar } from './SessionSidebar'
import { DocumentDrawer } from './DocumentDrawer'
import { ChatContainer } from '../chat/ChatContainer'
import { useSessions } from '../../context/SessionContext'
import { useParams } from 'react-router-dom'

export function AppShell() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const { sessions } = useSessions()
  const [drawerOpen, setDrawerOpen] = useState(false)

  if (!sessionId) return null

  const session = sessions.find((s) => s.id === sessionId)

  return (
    <div className="flex h-full overflow-hidden bg-white">
      <SessionSidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex items-center justify-between border-b border-gray-100 px-4 py-2.5">
          <h1 className="truncate text-sm font-semibold text-gray-800">
            {session?.label ?? 'Chat'}
          </h1>
          <button
            onClick={() => setDrawerOpen((o) => !o)}
            className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50 transition-colors"
          >
            <FolderOpen size={14} />
            Documents
          </button>
        </header>
        <ChatContainer sessionId={sessionId} />
      </div>
      <DocumentDrawer
        sessionId={sessionId}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  )
}
