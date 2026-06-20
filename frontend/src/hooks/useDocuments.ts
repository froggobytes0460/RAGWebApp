import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState, useCallback } from 'react'
import { api } from '../lib/api'
import type { JobProgressEvent } from '../types/api'

export interface ActiveJob {
  jobId: string
  filename: string
  status: JobProgressEvent['status']
  progress: number
  error: string | null
}

export function useDocuments(sessionId: string) {
  const qc = useQueryClient()

  const query = useQuery({
    queryKey: ['documents', sessionId],
    queryFn: () => api.listDocuments(sessionId),
  })

  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [activeJobs, setActiveJobs] = useState<Map<string, ActiveJob>>(new Map())

  const upsertJob = useCallback((job: ActiveJob) => {
    setActiveJobs((prev) => new Map(prev).set(job.jobId, job))
  }, [])

  const removeJob = useCallback((jobId: string) => {
    setActiveJobs((prev) => {
      const next = new Map(prev)
      next.delete(jobId)
      return next
    })
  }, [])

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const jobResp = await api.uploadDocument(sessionId, file, (pct) =>
        setUploadProgress(pct),
      )
      setUploadProgress(null)

      upsertJob({
        jobId: jobResp.job_id,
        filename: file.name,
        status: 'queued',
        progress: 0,
        error: null,
      })

      await api.streamJobProgress(
        sessionId,
        jobResp.job_id,
        {
          onProgress: (evt) => {
            upsertJob({
              jobId: evt.job_id,
              filename: evt.filename,
              status: evt.status,
              progress: evt.progress,
              error: evt.error,
            })
          },
          onDone: (evt) => {
            if (evt.status === 'done') {
              upsertJob({
                jobId: evt.job_id,
                filename: evt.filename,
                status: 'done',
                progress: 100,
                error: null,
              })
              setTimeout(() => {
                removeJob(evt.job_id)
                void qc.invalidateQueries({ queryKey: ['documents', sessionId] })
              }, 2000)
            } else {
              upsertJob({
                jobId: evt.job_id,
                filename: evt.filename,
                status: 'failed',
                progress: evt.progress,
                error: evt.error,
              })
            }
          },
          onError: (err) => {
            setActiveJobs((prev) => {
              const next = new Map(prev)
              const existing = next.get(jobResp.job_id)
              if (existing) {
                next.set(jobResp.job_id, { ...existing, status: 'failed', error: err.message })
              }
              return next
            })
          },
        },
      )

      return jobResp
    },
    onError: () => setUploadProgress(null),
  })

  const deleteMutation = useMutation({
    mutationFn: (filename: string) => api.deleteDocument(sessionId, filename),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['documents', sessionId] }),
  })

  return { query, uploadMutation, deleteMutation, uploadProgress, activeJobs }
}
