import { SourceChunks } from './SourceChunks'
import type { RetrievedChunk } from '../../types/api'

interface StreamingBubbleProps {
  content: string
  sources: RetrievedChunk[]
  isDone: boolean
}

export function StreamingBubble({ content, sources, isDone }: StreamingBubbleProps) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[75%]">
        <div className="rounded-2xl rounded-tl-sm bg-[#1a1d27] border border-gray-700 px-4 py-2.5 text-sm leading-relaxed text-gray-200 shadow-sm whitespace-pre-wrap">
          {content}
          {!isDone && (
            <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-gray-500 align-middle" />
          )}
        </div>
        {isDone && sources.length > 0 && (
          <div className="mt-1 px-1">
            <SourceChunks chunks={sources} />
          </div>
        )}
      </div>
    </div>
  )
}
