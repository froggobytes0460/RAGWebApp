import type { MessageRequest, RetrievedChunk } from '../types/api'

export interface SseFrame {
  eventType: string
  data: string
}

export async function parseSseStream(
  body: ReadableStream<Uint8Array>,
  onFrame: (frame: SseFrame) => boolean | void,
  signal?: AbortSignal,
): Promise<void> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''

      for (const eventBlock of events) {
        const lines = eventBlock.split('\n')
        let eventType = ''
        let dataLine = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.slice(7).trim()
          if (line.startsWith('data: ')) dataLine = line.slice(6).trim()
        }
        if (!dataLine) continue
        const stop = onFrame({ eventType, data: dataLine })
        if (stop) return
      }
    }
  } catch (err) {
    if ((err as Error).name !== 'AbortError') throw err
  } finally {
    if (signal?.aborted) reader.cancel().catch(() => {})
  }
}

export interface SSEHandlers {
  onChunk: (text: string) => void
  onDone: (chunks: RetrievedChunk[]) => void
  onError: (detail: string) => void
}

export async function streamMessage(
  sessionId: string,
  body: MessageRequest,
  handlers: SSEHandlers,
  signal: AbortSignal,
): Promise<void> {
  let response: Response
  try {
    response = await fetch(`/api/v1/chats/${sessionId}/messages/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(body),
      signal,
    })
  } catch (err) {
    if ((err as Error).name === 'AbortError') return
    handlers.onError('Network error')
    return
  }

  if (!response.ok) {
    const errBody = await response.json().catch(() => ({ detail: response.statusText }))
    handlers.onError((errBody as { detail?: string }).detail ?? `HTTP ${response.status}`)
    return
  }

  try {
    await parseSseStream(
      response.body!,
      ({ eventType, data }) => {
        const payload = JSON.parse(data) as Record<string, unknown>
        if (eventType === 'chunk') handlers.onChunk(payload.text as string)
        if (eventType === 'done')
          handlers.onDone((payload.retrieved_chunks as RetrievedChunk[]) ?? [])
        if (eventType === 'error') handlers.onError(payload.detail as string)
      },
      signal,
    )
  } catch {
    handlers.onError('Stream interrupted')
  }
}
