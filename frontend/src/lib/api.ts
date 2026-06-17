import type { IngestResponse, MessageHistoryItem } from '../types/api'

const BASE = '/api/v1/chats'

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  listDocuments: async (sessionId: string): Promise<string[]> => {
    const res = await fetch(`${BASE}/${sessionId}/documents`)
    if (res.status === 404) return []
    return handleResponse<string[]>(res)
  },

  uploadDocument: (
    sessionId: string,
    file: File,
    onProgress?: (pct: number) => void,
  ): Promise<IngestResponse> => {
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
        if (xhr.status === 201) {
          resolve(JSON.parse(xhr.responseText) as IngestResponse)
        } else {
          const body = JSON.parse(xhr.responseText) as { detail?: string }
          reject(new Error(body.detail ?? `HTTP ${xhr.status}`))
        }
      }
      xhr.onerror = () => reject(new Error('Network error'))
      xhr.open('POST', `${BASE}/${sessionId}/documents`)
      xhr.send(form)
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
    const res = await fetch(`${BASE}/${sessionId}/messages`)
    if (res.status === 404) return []
    return handleResponse<MessageHistoryItem[]>(res)
  },
}
