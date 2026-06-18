import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export function useMessages(sessionId: string) {
  return useQuery({
    queryKey: ['messages', sessionId],
    queryFn: () => api.listMessages(sessionId),
    staleTime: Infinity,
  })
}
