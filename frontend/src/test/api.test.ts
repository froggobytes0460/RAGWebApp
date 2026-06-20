import { describe, it, expect, vi, beforeEach } from 'vitest'
import { api } from '../lib/api'

beforeEach(() => {
  vi.restoreAllMocks()
})

function mockFetch(status: number, body: unknown, ok = status >= 200 && status < 300) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok,
      status,
      statusText: 'Status ' + status,
      json: vi.fn().mockResolvedValue(body),
    }),
  )
}

describe('api.listDocuments', () => {
  it('returns parsed list on success', async () => {
    mockFetch(200, ['a.pdf', 'b.docx'])
    await expect(api.listDocuments('sess1')).resolves.toEqual(['a.pdf', 'b.docx'])
  })

  it('returns empty array on 404', async () => {
    mockFetch(404, { detail: 'not found' }, false)
    await expect(api.listDocuments('sess1')).resolves.toEqual([])
  })

  it('throws on other error status', async () => {
    mockFetch(500, { detail: 'server boom' }, false)
    await expect(api.listDocuments('sess1')).rejects.toThrow('server boom')
  })
})

describe('api.listMessages', () => {
  it('returns messages on success', async () => {
    const messages = [{ id: 1, role: 'user', content: 'Hi', created_at: '2024-01-01T00:00:00Z' }]
    mockFetch(200, messages)
    await expect(api.listMessages('sess1')).resolves.toEqual(messages)
  })

  it('returns empty array on 404', async () => {
    mockFetch(404, {}, false)
    await expect(api.listMessages('sess1')).resolves.toEqual([])
  })
})

describe('api.deleteDocument', () => {
  it('resolves on 200', async () => {
    mockFetch(200, {})
    await expect(api.deleteDocument('sess1', 'file.pdf')).resolves.toBeUndefined()
  })

  it('url-encodes the filename', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: vi.fn() })
    vi.stubGlobal('fetch', fetchMock)
    await api.deleteDocument('sess1', 'my file.pdf')
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/chats/sess1/documents/my%20file.pdf',
      expect.objectContaining({ method: 'DELETE' }),
    )
  })

  it('throws with detail message on error', async () => {
    mockFetch(403, { detail: 'Forbidden' }, false)
    await expect(api.deleteDocument('sess1', 'file.pdf')).rejects.toThrow('Forbidden')
  })
})

describe('api.streamJobProgress', () => {
  function makeSSEResponse(chunks: string[], status = 200) {
    let chunkIndex = 0
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      pull(controller) {
        if (chunkIndex < chunks.length) {
          controller.enqueue(encoder.encode(chunks[chunkIndex++]))
        } else {
          controller.close()
        }
      },
    })
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: status < 400, status, body: stream }),
    )
  }

  it('calls onProgress for progress events', async () => {
    const onProgress = vi.fn()
    const onDone = vi.fn()
    const onError = vi.fn()
    makeSSEResponse([
      'data: {"status":"processing","progress":50}\n\n',
      'data: {"status":"done","progress":100}\n\n',
    ])
    await api.streamJobProgress('sess1', 'job-1', { onProgress, onDone, onError })
    expect(onProgress).toHaveBeenCalledWith({ status: 'processing', progress: 50 })
    expect(onDone).toHaveBeenCalledWith({ status: 'done', progress: 100 })
    expect(onError).not.toHaveBeenCalled()
  })

  it('calls onDone for failed status', async () => {
    const onDone = vi.fn()
    makeSSEResponse(['data: {"status":"failed","progress":0}\n\n'])
    await api.streamJobProgress('sess1', 'job-1', { onProgress: vi.fn(), onDone, onError: vi.fn() })
    expect(onDone).toHaveBeenCalledWith({ status: 'failed', progress: 0 })
  })

  it('calls onError on non-ok response', async () => {
    const onError = vi.fn()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500, body: null }))
    await api.streamJobProgress('sess1', 'job-1', { onProgress: vi.fn(), onDone: vi.fn(), onError })
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'HTTP 500' }))
  })

  it('ignores malformed SSE frames', async () => {
    const onProgress = vi.fn()
    const onDone = vi.fn()
    makeSSEResponse([
      'data: not-json\n\n',
      'data: {"status":"done","progress":100}\n\n',
    ])
    await api.streamJobProgress('sess1', 'job-1', { onProgress, onDone, onError: vi.fn() })
    expect(onProgress).not.toHaveBeenCalled()
    expect(onDone).toHaveBeenCalledWith({ status: 'done', progress: 100 })
  })

  it('does not call onError when aborted', async () => {
    const onError = vi.fn()
    const abortError = new Error('aborted')
    abortError.name = 'AbortError'
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(abortError))
    await api.streamJobProgress('sess1', 'job-1', { onProgress: vi.fn(), onDone: vi.fn(), onError })
    expect(onError).not.toHaveBeenCalled()
  })
})

describe('api.uploadDocument', () => {
  interface XHRInstance {
    upload: { onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null }
    onload: (() => void) | null
    onerror: (() => void) | null
    open: ReturnType<typeof vi.fn>
    send: ReturnType<typeof vi.fn>
    status: number
    responseText: string
  }

  function makeXHRMock(status: number, responseText: string, sendImpl?: (instance: XHRInstance) => void) {
    let instance!: XHRInstance
    class MockXHR {
      upload = { onprogress: null as XHRInstance['upload']['onprogress'] }
      onload: (() => void) | null = null
      onerror: (() => void) | null = null
      status = status
      responseText = responseText
      open = vi.fn()
      send = vi.fn().mockImplementation(() => {
        if (sendImpl) {
          sendImpl(this as unknown as XHRInstance)
        } else {
          this.onload?.()
        }
      })
      constructor() { instance = this as unknown as XHRInstance }
    }
    vi.stubGlobal('XMLHttpRequest', MockXHR)
    return { getInstance: () => instance }
  }

  it('resolves with IngestJobResponse on 202', async () => {
    const payload = { job_id: 'job-1' }
    makeXHRMock(202, JSON.stringify(payload))
    await expect(api.uploadDocument('sess1', new File([''], 'test.pdf'))).resolves.toEqual(payload)
  })

  it('rejects with detail on non-201', async () => {
    makeXHRMock(413, JSON.stringify({ detail: 'File too large' }))
    await expect(api.uploadDocument('sess1', new File([''], 'big.pdf'))).rejects.toThrow('File too large')
  })

  it('reports upload progress', async () => {
    const onProgress = vi.fn()
    makeXHRMock(202, JSON.stringify({ job_id: 'job-1' }), (xhr) => {
      xhr.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 100 })
      xhr.onload?.()
    })
    await api.uploadDocument('sess1', new File([''], 'test.pdf'), onProgress)
    expect(onProgress).toHaveBeenCalledWith(50)
  })
})
