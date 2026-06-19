import type {
  DocumentListItem,
  IngestJobResponse,
  JobProgressEvent,
  MessageHistoryItem,
} from '../types/api'

const BASE = '/api/v1/chats'

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  listDocuments: async (sessionId: string): Promise<DocumentListItem[]> => {
    const res = await fetch(`${BASE}/${sessionId}/documents/`)
    if (res.status === 404) return []
    return handleResponse<DocumentListItem[]>(res)
  },

  uploadDocument: (
    sessionId: string,
    file: File,
    onProgress?: (pct: number) => void,
  ): Promise<IngestJobResponse> => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      const form = new FormData()
      form.append('file', file)
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      }
      xhr.onload = () => {
        if (xhr.status === 202) {
          resolve(JSON.parse(xhr.responseText) as IngestJobResponse)
        } else {
          const body = JSON.parse(xhr.responseText) as { detail?: string }
          reject(new Error(body.detail ?? `HTTP ${xhr.status}`))
        }
      }
      xhr.onerror = () => reject(new Error('Network error'))
      xhr.open('POST', `${BASE}/${sessionId}/documents/`)
      xhr.send(form)
    })
  },

  streamJobProgress: (
    sessionId: string,
    jobId: string,
    handlers: {
      onProgress: (event: JobProgressEvent) => void
      onDone: (event: JobProgressEvent) => void
      onError: (err: Error) => void
    },
    signal?: AbortSignal,
  ): Promise<void> => {
    return fetch(`${BASE}/${sessionId}/documents/jobs/${jobId}/progress`, { signal })
      .then((res) => {
        if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        const pump = (): Promise<void> =>
          reader.read().then(({ done, value }) => {
            if (done) return
            buffer += decoder.decode(value, { stream: true })
            const parts = buffer.split('\n\n')
            buffer = parts.pop() ?? ''
            for (const part of parts) {
              const dataLine = part.split('\n').find((l) => l.startsWith('data:'))
              if (!dataLine) continue
              try {
                const evt = JSON.parse(dataLine.slice(5).trim()) as JobProgressEvent
                if (evt.status === 'done' || evt.status === 'failed') {
                  handlers.onDone(evt)
                  return
                }
                handlers.onProgress(evt)
              } catch {
                // ignore malformed frames
              }
            }
            return pump()
          })

        return pump()
      })
      .catch((err: unknown) => {
        if (err instanceof Error && err.name !== 'AbortError') {
          handlers.onError(err)
        }
      })
  },

  deleteDocument: async (sessionId: string, filename: string): Promise<void> => {
    const res = await fetch(
      `${BASE}/${sessionId}/documents/${encodeURIComponent(filename)}`,
      { method: 'DELETE' },
    )
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(body.detail ?? `HTTP ${res.status}`)
    }
  },

  listMessages: async (sessionId: string): Promise<MessageHistoryItem[]> => {
    const res = await fetch(`${BASE}/${sessionId}/messages/`)
    if (res.status === 404) return []
    return handleResponse<MessageHistoryItem[]>(res)
  },
}
