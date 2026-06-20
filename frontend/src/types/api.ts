export interface IngestResponse {
  document_id: string
  doc_count: number
}

export interface RetrievedChunk {
  content: string
  score: number
  filename: string
  page_number: number | null
}

export interface MetadataFilter {
  filenames?: string[]
  page_min?: number
  page_max?: number
}

export interface MessageRequest {
  question: string
  top_k?: number
  score_threshold?: number
  filters?: MetadataFilter
}

export interface MessageHistoryItem {
  id: number | null
  role: 'user' | 'ai'
  content: string
  created_at: string
  retrieved_chunks?: RetrievedChunk[]
}

export interface DocumentListItem {
  filename: string
  [key: string]: string
}

export interface IngestJobResponse {
  job_id: string
  status: 'queued'
}

export interface JobProgressEvent {
  job_id: string
  filename: string
  status: 'queued' | 'processing' | 'done' | 'failed'
  progress: number
  chunk_count: number | null
  error: string | null
}
