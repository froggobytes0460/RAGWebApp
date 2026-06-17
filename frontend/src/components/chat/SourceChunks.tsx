import { Collapsible } from '../ui/Collapsible'
import { Badge } from '../ui/Badge'
import type { RetrievedChunk } from '../../types/api'

interface SourceChunksProps {
  chunks: RetrievedChunk[]
}

export function SourceChunks({ chunks }: SourceChunksProps) {
  if (chunks.length === 0) return null

  return (
    <Collapsible trigger={`${chunks.length} source${chunks.length > 1 ? 's' : ''}`} className="mt-2">
      <div className="flex flex-col gap-2">
        {chunks.map((chunk, i) => (
          <div key={i} className="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 text-xs text-gray-600">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <span className="font-medium text-gray-700 truncate">{chunk.filename}</span>
              {chunk.page_number != null && (
                <span className="text-gray-400">p.{chunk.page_number}</span>
              )}
              <Badge value={chunk.score} />
            </div>
            <p className="leading-relaxed line-clamp-4">{chunk.content}</p>
          </div>
        ))}
      </div>
    </Collapsible>
  )
}
