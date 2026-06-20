import { useState, useRef, useCallback, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import { ErrorBanner } from '../ui/ErrorBanner'
import { useMessages } from '../../hooks/useMessages'
import { streamMessage } from '../../lib/sse'
import { useSessions } from '../../context/SessionContext'
import type { RetrievedChunk } from '../../types/api'

interface ChatContainerProps {
  sessionId: string
}

export function ChatContainer({ sessionId }: ChatContainerProps) {
  const qc = useQueryClient()
  const { data: messages = [], isLoading } = useMessages(sessionId)
  const { renameSession, sessions } = useSessions()

  const [streamingContent, setStreamingContent] = useState('')
  const [streamingSources, setStreamingSources] = useState<RetrievedChunk[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingDone, setStreamingDone] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const messageCountAtSendRef = useRef<number>(0)
  const streamingDoneRef = useRef(false)

  useEffect(() => {
    if (streamingDoneRef.current && messages.length > messageCountAtSendRef.current) {
      const lastMsg = messages[messages.length - 1]
      if (lastMsg?.role === 'ai') {
        streamingDoneRef.current = false
        setStreamingDone(false)
        setStreamingContent('')
      }
    }
  }, [messages])

  const handleSend = useCallback(
    async (question: string, topK: number, scoreThreshold: number | undefined) => {
      abortRef.current = new AbortController()
      messageCountAtSendRef.current = messages.length
      setIsStreaming(true)
      setStreamingDone(false)
      setStreamingContent('')
      setStreamingSources([])
      setError(null)

      // Auto-label the session with the first question
      const session = sessions.find((s) => s.id === sessionId)
      if (session?.label === 'New Chat') {
        renameSession(sessionId, question.slice(0, 40))
      }

      await streamMessage(
        sessionId,
        { question, top_k: topK, score_threshold: scoreThreshold },
        {
          onChunk: (text) => setStreamingContent((prev) => prev + text),
          onDone: (chunks) => {
            setStreamingSources(chunks)
            streamingDoneRef.current = true
            setStreamingDone(true)
            setIsStreaming(false)
            void qc.invalidateQueries({ queryKey: ['messages', sessionId] })
          },
          onError: (detail) => {
            setError(detail)
            setIsStreaming(false)
            setStreamingDone(false)
          },
        },
        abortRef.current.signal,
      )
    },
    [sessionId, sessions, renameSession, qc, messages.length],
  )

  const handleStop = useCallback(() => {
    abortRef.current?.abort()
    setIsStreaming(false)
    setStreamingDone(false)
    setStreamingContent('')
    setStreamingSources([])
  }, [])

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <MessageList
        messages={messages}
        isLoading={isLoading}
        streamingContent={streamingContent}
        streamingSources={streamingSources}
        isStreaming={isStreaming}
        streamingDone={streamingDone}
      />
      <div className="flex flex-col gap-2 px-4 pb-0">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
      </div>
      <ChatInput onSend={handleSend} isStreaming={isStreaming} onStop={handleStop} />
    </div>
  )
}
