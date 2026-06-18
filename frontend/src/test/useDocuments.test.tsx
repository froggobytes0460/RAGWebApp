import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { type ReactNode } from 'react'
import { useDocuments } from '../hooks/useDocuments'
import { api } from '../lib/api'

vi.mock('../lib/api', () => ({
  api: {
    listDocuments: vi.fn(),
    uploadDocument: vi.fn(),
    deleteDocument: vi.fn(),
  },
}))

function makeWrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return {
    qc,
    wrapper: ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={qc}>{children}</QueryClientProvider>
    ),
  }
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('useDocuments', () => {
  it('fetches and returns document list', async () => {
    vi.mocked(api.listDocuments).mockResolvedValue([{ filename: 'doc1.pdf' }, { filename: 'doc2.pdf' }])
    const { wrapper } = makeWrapper()

    const { result } = renderHook(() => useDocuments('sess1'), { wrapper })

    await waitFor(() => expect(result.current.query.isSuccess).toBe(true))
    expect(result.current.query.data).toEqual([{ filename: 'doc1.pdf' }, { filename: 'doc2.pdf' }])
  })

  it('exposes error state when fetch fails', async () => {
    vi.mocked(api.listDocuments).mockRejectedValue(new Error('Network error'))
    const { wrapper } = makeWrapper()

    const { result } = renderHook(() => useDocuments('sess1'), { wrapper })

    await waitFor(() => expect(result.current.query.isError).toBe(true))
    expect((result.current.query.error as Error).message).toBe('Network error')
  })

  it('uploadMutation calls api.uploadDocument and invalidates query', async () => {
    vi.mocked(api.listDocuments).mockResolvedValue([])
    vi.mocked(api.uploadDocument).mockResolvedValue({ document_id: 'x', doc_count: 1 })

    const { wrapper, qc } = makeWrapper()
    const invalidateSpy = vi.spyOn(qc, 'invalidateQueries')

    const { result } = renderHook(() => useDocuments('sess1'), { wrapper })
    await waitFor(() => expect(result.current.query.isSuccess).toBe(true))

    const file = new File([''], 'test.pdf')
    await act(async () => {
      await result.current.uploadMutation.mutateAsync(file)
    })

    expect(api.uploadDocument).toHaveBeenCalledWith('sess1', file, expect.any(Function))
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['documents', 'sess1'] })
  })

  it('uploadProgress resets to null after successful upload', async () => {
    vi.mocked(api.listDocuments).mockResolvedValue([])
    vi.mocked(api.uploadDocument).mockResolvedValue({ document_id: 'x', doc_count: 1 })

    const { wrapper } = makeWrapper()
    const { result } = renderHook(() => useDocuments('sess1'), { wrapper })
    await waitFor(() => expect(result.current.query.isSuccess).toBe(true))

    await act(async () => {
      await result.current.uploadMutation.mutateAsync(new File([''], 'f.pdf'))
    })

    expect(result.current.uploadProgress).toBeNull()
  })

  it('deleteMutation calls api.deleteDocument and invalidates query', async () => {
    vi.mocked(api.listDocuments).mockResolvedValue([{ filename: 'file.pdf' }])
    vi.mocked(api.deleteDocument).mockResolvedValue(undefined)

    const { wrapper, qc } = makeWrapper()
    const invalidateSpy = vi.spyOn(qc, 'invalidateQueries')

    const { result } = renderHook(() => useDocuments('sess1'), { wrapper })
    await waitFor(() => expect(result.current.query.isSuccess).toBe(true))

    await act(async () => {
      await result.current.deleteMutation.mutateAsync('file.pdf')
    })

    expect(api.deleteDocument).toHaveBeenCalledWith('sess1', 'file.pdf')
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['documents', 'sess1'] })
  })
})
