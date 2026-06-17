import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { type ReactNode } from 'react'
import { SessionProvider, useSessions } from '../context/SessionContext'

const STORAGE_KEY = 'rag_sessions'

function wrapper({ children }: { children: ReactNode }) {
  return <SessionProvider>{children}</SessionProvider>
}

beforeEach(() => {
  localStorage.clear()
})

describe('useSessions', () => {
  it('starts with empty sessions when localStorage is empty', () => {
    const { result } = renderHook(() => useSessions(), { wrapper })
    expect(result.current.sessions).toEqual([])
  })

  it('loads persisted sessions from localStorage on mount', () => {
    const stored = [{ id: 'abc', label: 'Old Chat', createdAt: '2024-01-01T00:00:00Z' }]
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stored))

    const { result } = renderHook(() => useSessions(), { wrapper })
    expect(result.current.sessions).toEqual(stored)
  })

  it('createSession adds a new session and persists it', () => {
    const { result } = renderHook(() => useSessions(), { wrapper })

    let session: ReturnType<typeof result.current.createSession>
    act(() => {
      session = result.current.createSession()
    })

    expect(result.current.sessions).toHaveLength(1)
    expect(result.current.sessions[0].label).toBe('New Chat')
    expect(result.current.sessions[0].id).toBe(session!.id)

    const persisted = JSON.parse(localStorage.getItem(STORAGE_KEY)!)
    expect(persisted).toHaveLength(1)
  })

  it('createSession prepends the newest session', () => {
    const { result } = renderHook(() => useSessions(), { wrapper })

    act(() => { result.current.createSession() })
    act(() => { result.current.createSession() })

    expect(result.current.sessions).toHaveLength(2)
    // newest first: the second created session should be at index 0
    const [first, second] = result.current.sessions
    expect(new Date(first.createdAt) >= new Date(second.createdAt)).toBe(true)
  })

  it('deleteSession removes the correct session', () => {
    const { result } = renderHook(() => useSessions(), { wrapper })

    act(() => { result.current.createSession() })
    act(() => { result.current.createSession() })

    const idToDelete = result.current.sessions[1].id

    act(() => { result.current.deleteSession(idToDelete) })

    expect(result.current.sessions).toHaveLength(1)
    expect(result.current.sessions.find((s) => s.id === idToDelete)).toBeUndefined()
  })

  it('deleteSession persists change to localStorage', () => {
    const { result } = renderHook(() => useSessions(), { wrapper })

    act(() => { result.current.createSession() })
    const id = result.current.sessions[0].id
    act(() => { result.current.deleteSession(id) })

    const persisted = JSON.parse(localStorage.getItem(STORAGE_KEY)!)
    expect(persisted).toHaveLength(0)
  })

  it('renameSession updates the label', () => {
    const { result } = renderHook(() => useSessions(), { wrapper })

    act(() => { result.current.createSession() })
    const id = result.current.sessions[0].id

    act(() => { result.current.renameSession(id, 'My renamed chat') })

    expect(result.current.sessions[0].label).toBe('My renamed chat')
  })

  it('renameSession persists change to localStorage', () => {
    const { result } = renderHook(() => useSessions(), { wrapper })

    act(() => { result.current.createSession() })
    const id = result.current.sessions[0].id
    act(() => { result.current.renameSession(id, 'Renamed') })

    const persisted = JSON.parse(localStorage.getItem(STORAGE_KEY)!)
    expect(persisted[0].label).toBe('Renamed')
  })

  it('throws when used outside SessionProvider', () => {
    expect(() => renderHook(() => useSessions())).toThrow(
      'useSessions must be used inside SessionProvider',
    )
  })
})
