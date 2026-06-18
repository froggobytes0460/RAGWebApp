import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { type ReactNode } from 'react'
import { useMessages } from '../hooks/useMessages'
import { api } from '../lib/api'

vi.mock('../lib/api', () => ({
  api: {
    listMessages: vi.fn(),
  },
}))

function makeWrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  )
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('useMessages', () => {
  it('returns message history on success', async () => {
    const messages = [
      { id: 1, role: 'user' as const, content: 'Hello', created_at: '2024-01-01T00:00:00Z' },
      { id: 2, role: 'ai' as const, content: 'Hi there!', created_at: '2024-01-01T00:00:01Z' },
    ]
    vi.mocked(api.listMessages).mockResolvedValue(messages)

    const { result } = renderHook(() => useMessages('sess1'), { wrapper: makeWrapper() })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual(messages)
  })

  it('returns empty array for new sessions (404)', async () => {
    vi.mocked(api.listMessages).mockResolvedValue([])

    const { result } = renderHook(() => useMessages('new-session'), { wrapper: makeWrapper() })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual([])
  })

  it('exposes error state on failure', async () => {
    vi.mocked(api.listMessages).mockRejectedValue(new Error('fetch failed'))

    const { result } = renderHook(() => useMessages('sess1'), { wrapper: makeWrapper() })

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect((result.current.error as Error).message).toBe('fetch failed')
  })

  it('does not refetch after data is loaded (staleTime: Infinity)', async () => {
    vi.mocked(api.listMessages).mockResolvedValue([])

    const { result } = renderHook(() => useMessages('sess1'), { wrapper: makeWrapper() })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    // isStale should be false because staleTime is Infinity
    expect(result.current.isStale).toBe(false)
    expect(api.listMessages).toHaveBeenCalledTimes(1)
  })
})
