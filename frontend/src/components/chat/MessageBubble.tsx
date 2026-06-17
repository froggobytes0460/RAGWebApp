import { SourceChunks } from './SourceChunks'
import type { MessageHistoryItem } from '../../types/api'

interface MessageBubbleProps {
  message: MessageHistoryItem
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[75%] ${isUser ? 'order-2' : ''}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap
            ${isUser
              ? 'rounded-tr-sm bg-violet-600 text-white'
              : 'rounded-tl-sm bg-white border border-gray-100 text-gray-800 shadow-sm'
            }`}
        >
          {message.content}
        </div>
        {!isUser && message.retrieved_chunks && message.retrieved_chunks.length > 0 && (
          <div className="mt-1 px-1">
            <SourceChunks chunks={message.retrieved_chunks} />
          </div>
        )}
      </div>
    </div>
  )
}
