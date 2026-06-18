import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import { generateUUID } from '../lib/uuid'

export interface Session {
  id: string
  label: string
  createdAt: string
}

interface SessionContextValue {
  sessions: Session[]
  createSession: () => Session
  deleteSession: (id: string) => void
  renameSession: (id: string, label: string) => void
}

const STORAGE_KEY = 'rag_sessions'

function loadSessions(): Session[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as Session[]) : []
  } catch {
    return []
  }
}

function saveSessions(sessions: Session[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
}

const SessionContext = createContext<SessionContextValue | null>(null)

export function SessionProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<Session[]>(loadSessions)

  const update = useCallback((next: Session[]) => {
    setSessions(next)
    saveSessions(next)
  }, [])

  const createSession = useCallback((): Session => {
    const session: Session = {
      id: generateUUID(),
      label: 'New Chat',
      createdAt: new Date().toISOString(),
    }
    setSessions((prev) => {
      const next = [session, ...prev]
      saveSessions(next)
      return next
    })
    return session
  }, [])

  const deleteSession = useCallback(
    (id: string) => update(sessions.filter((s) => s.id !== id)),
    [sessions, update],
  )

  const renameSession = useCallback(
    (id: string, label: string) =>
      update(sessions.map((s) => (s.id === id ? { ...s, label } : s))),
    [sessions, update],
  )

  return (
    <SessionContext.Provider value={{ sessions, createSession, deleteSession, renameSession }}>
      {children}
    </SessionContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useSessions(): SessionContextValue {
  const ctx = useContext(SessionContext)
  if (!ctx) throw new Error('useSessions must be used inside SessionProvider')
  return ctx
}
