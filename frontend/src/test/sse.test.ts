import { describe, it, expect, vi, beforeEach } from 'vitest'
import { streamMessage } from '../lib/sse'
import type { SSEHandlers } from '../lib/sse'
import type { MessageRequest } from '../types/api'

const SESSION_ID = 'test-session'
const BODY: MessageRequest = { question: 'Hello' }

function makeHandlers(): SSEHandlers & { chunks: string[]; doneArg: unknown; errorArg: string } {
  const chunks: string[] = []
  let doneArg: unknown = null
  let errorArg = ''
  return {
    chunks,
    get doneArg() { return doneArg },
    get errorArg() { return errorArg },
    onChunk: (t) => chunks.push(t),
    onDone: (c) => { doneArg = c },
    onError: (d) => { errorArg = d },
  }
}

function encodeSSE(events: Array<{ event: string; data: unknown }>): ReadableStream<Uint8Array> {
  const text = events.map(({ event, data }) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join('')
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(text))
      controller.close()
    },
  })
}

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('streamMessage – happy path', () => {
  it('dispatches chunk and done events', async () => {
    const stream = encodeSSE([
      { event: 'chunk', data: { text: 'Hello ' } },
      { event: 'chunk', data: { text: 'world' } },
      { event: 'done', data: { retrieved_chunks: [{ content: 'ctx', score: 0.9, filename: 'f.pdf', page_number: 1 }] } },
    ])

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body: stream }))

    const h = makeHandlers()
    await streamMessage(SESSION_ID, BODY, h, new AbortController().signal)

    expect(h.chunks).toEqual(['Hello ', 'world'])
    expect(h.doneArg).toEqual([{ content: 'ctx', score: 0.9, filename: 'f.pdf', page_number: 1 }])
    expect(h.errorArg).toBe('')
  })

  it('sends correct fetch options', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, body: encodeSSE([]) })
    vi.stubGlobal('fetch', fetchMock)

    await streamMessage(SESSION_ID, BODY, makeHandlers(), new AbortController().signal)

    expect(fetchMock).toHaveBeenCalledWith(
      `/api/v1/chats/${SESSION_ID}/messages`,
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'Content-Type': 'application/json', Accept: 'text/event-stream' }),
        body: JSON.stringify(BODY),
      }),
    )
  })
})

describe('streamMessage – error handling', () => {
  it('calls onError for non-ok response with detail', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Entity',
      json: vi.fn().mockResolvedValue({ detail: 'Bad question' }),
    }))

    const h = makeHandlers()
    await streamMessage(SESSION_ID, BODY, h, new AbortController().signal)

    expect(h.errorArg).toBe('Bad question')
    expect(h.chunks).toHaveLength(0)
  })

  it('falls back to statusText when json parse fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: vi.fn().mockRejectedValue(new Error('not json')),
    }))

    const h = makeHandlers()
    await streamMessage(SESSION_ID, BODY, h, new AbortController().signal)

    expect(h.errorArg).toBe('Internal Server Error')
  })

  it('calls onError when network throws', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('net::ERR_FAILED')))

    const h = makeHandlers()
    await streamMessage(SESSION_ID, BODY, h, new AbortController().signal)

    expect(h.errorArg).toBe('Network error')
  })

  it('silently returns on AbortError', async () => {
    const err = Object.assign(new Error('aborted'), { name: 'AbortError' })
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(err))

    const h = makeHandlers()
    await streamMessage(SESSION_ID, BODY, h, new AbortController().signal)

    expect(h.errorArg).toBe('')
    expect(h.chunks).toHaveLength(0)
  })

  it('dispatches onError event from SSE error event', async () => {
    const stream = encodeSSE([{ event: 'error', data: { detail: 'LLM quota exceeded' } }])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body: stream }))

    const h = makeHandlers()
    await streamMessage(SESSION_ID, BODY, h, new AbortController().signal)

    expect(h.errorArg).toBe('LLM quota exceeded')
  })
})
