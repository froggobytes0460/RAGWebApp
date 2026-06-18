import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../lib/api'

export function useDocuments(sessionId: string) {
  const qc = useQueryClient()

  const query = useQuery({
    queryKey: ['documents', sessionId],
    queryFn: () => api.listDocuments(sessionId),
  })

  const [uploadProgress, setUploadProgress] = useState<number | null>(null)

  const uploadMutation = useMutation({
    mutationFn: (file: File) =>
      api.uploadDocument(sessionId, file, (pct) => setUploadProgress(pct)),
    onSuccess: () => {
      setUploadProgress(null)
      void qc.invalidateQueries({ queryKey: ['documents', sessionId] })
    },
    onError: () => setUploadProgress(null),
  })

  const deleteMutation = useMutation({
    mutationFn: (filename: string) => api.deleteDocument(sessionId, filename),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['documents', sessionId] }),
  })

  return { query, uploadMutation, deleteMutation, uploadProgress }
}
