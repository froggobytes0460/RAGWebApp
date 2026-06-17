import { useEffect, useRef } from 'react'
import { MessageBubble } from './MessageBubble'
import { StreamingBubble } from './StreamingBubble'
import { Spinner } from '../ui/Spinner'
import type { MessageHistoryItem, RetrievedChunk } from '../../types/api'

interface MessageListProps {
  messages: MessageHistoryItem[]
  isLoading: boolean
  streamingContent: string
  streamingSources: RetrievedChunk[]
  isStreaming: boolean
  streamingDone: boolean
}

export function MessageList({
  messages,
  isLoading,
  streamingContent,
  streamingSources,
  isStreaming,
  streamingDone,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, streamingContent])

  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner className="h-6 w-6 text-gray-300" />
      </div>
    )
  }

  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center text-center px-8">
        <p className="text-2xl font-semibold text-gray-600">Ask anything</p>
        <p className="mt-2 text-sm text-gray-500">Upload documents, then start a conversation.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 py-6">
      {messages.map((msg) => (
        <MessageBubble key={msg.id ?? msg.created_at} message={msg} />
      ))}
      {(isStreaming || streamingDone) && (
        <StreamingBubble
          content={streamingContent}
          sources={streamingSources}
          isDone={streamingDone}
        />
      )}
      <div ref={bottomRef} />
    </div>
  )
}
