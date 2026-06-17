import { useNavigate, useParams } from 'react-router-dom'
import { Plus, Trash2, MessageSquare } from 'lucide-react'
import { useSessions } from '../../context/SessionContext'

export function SessionSidebar() {
  const { sessions, createSession, deleteSession } = useSessions()
  const navigate = useNavigate()
  const { sessionId } = useParams<{ sessionId: string }>()

  const handleNew = () => {
    const session = createSession()
    navigate(`/chat/${session.id}`)
  }

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-gray-100 bg-gray-50">
      <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3">
        <span className="text-sm font-semibold text-gray-700">Chats</span>
        <button
          onClick={handleNew}
          className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-200 hover:text-gray-700 transition-colors"
          title="New chat"
        >
          <Plus size={16} />
        </button>
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 overflow-y-auto p-2">
        {sessions.length === 0 && (
          <p className="px-2 py-3 text-center text-xs text-gray-400">No chats yet.</p>
        )}
        {sessions.map((session) => {
          const isActive = session.id === sessionId
          return (
            <div
              key={session.id}
              className={`group flex items-center gap-2 rounded-lg px-2 py-2 cursor-pointer transition-colors
                ${isActive ? 'bg-violet-100 text-violet-800' : 'text-gray-600 hover:bg-gray-200'}`}
              onClick={() => navigate(`/chat/${session.id}`)}
            >
              <MessageSquare size={14} className="shrink-0 opacity-60" />
              <span className="flex-1 truncate text-sm" title={session.label}>
                {session.label}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  deleteSession(session.id)
                  if (isActive) navigate('/')
                }}
                className="hidden shrink-0 rounded p-0.5 hover:text-red-500 group-hover:block"
                title="Delete chat"
              >
                <Trash2 size={12} />
              </button>
            </div>
          )
        })}
      </nav>
    </aside>
  )
}
