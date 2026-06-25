import { useState, useRef, useCallback } from 'react'
import { Send, Settings2, X } from 'lucide-react'

interface ChatInputProps {
  onSend: (question: string, topK: number, scoreThreshold: number | undefined) => void
  isStreaming: boolean
  onStop: () => void
}

export function ChatInput({ onSend, isStreaming, onStop }: ChatInputProps) {
  const [text, setText] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [topK, setTopK] = useState(4)
  const [scoreThreshold, setScoreThreshold] = useState<number | undefined>(undefined)
  const [useThreshold, setUseThreshold] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = useCallback(() => {
    const q = text.trim()
    if (!q || isStreaming) return
    onSend(q, topK, useThreshold ? scoreThreshold : undefined)
    setText('')
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }, 0)
  }, [text, isStreaming, onSend, topK, scoreThreshold, useThreshold])

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const onInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`
  }

  return (
    <div className="border-t border-gray-800 bg-[#13151e] px-4 py-3">
      {showSettings && (
        <div className="mb-3 rounded-xl border border-gray-700 bg-[#1a1d27] px-4 py-3 text-sm">
          <div className="flex items-center justify-between mb-3">
            <span className="font-medium text-gray-200">Retrieval settings</span>
            <button onClick={() => setShowSettings(false)} className="text-gray-500 hover:text-gray-300">
              <X size={14} />
            </button>
          </div>
          <div className="flex flex-col gap-3">
            <label className="flex flex-col gap-1">
              <div className="flex justify-between text-xs text-gray-400">
                <span>Top K chunks (after reranking)</span>
                <span className="font-medium text-gray-200">{topK}</span>
              </div>
              <input
                type="range" min={1} max={25} value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="accent-violet-600"
              />
            </label>
            <label className="flex flex-col gap-1">
              <div className="flex items-center justify-between text-xs text-gray-400">
                <span>Score threshold</span>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox" checked={useThreshold}
                    onChange={(e) => setUseThreshold(e.target.checked)}
                    className="accent-violet-500"
                  />
                  <span className="font-medium text-gray-200">
                    {useThreshold ? (scoreThreshold ?? 0.5).toFixed(2) : 'off'}
                  </span>
                </div>
              </div>
              {useThreshold && (
                <input
                  type="range" min={0} max={100} value={Math.round((scoreThreshold ?? 0.5) * 100)}
                  onChange={(e) => setScoreThreshold(Number(e.target.value) / 100)}
                  className="accent-violet-600"
                />
              )}
            </label>
          </div>
        </div>
      )}

      <div className="flex items-end gap-2">
        <button
          onClick={() => setShowSettings((s) => !s)}
          className={`shrink-0 rounded-lg p-2 transition-colors ${showSettings ? 'bg-violet-900/50 text-violet-400' : 'text-gray-500 hover:bg-gray-800 hover:text-gray-300'}`}
          title="Retrieval settings"
        >
          <Settings2 size={18} />
        </button>
        <textarea
          ref={textareaRef}
          value={text}
          onChange={onInput}
          onKeyDown={onKeyDown}
          placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
          rows={1}
          className="flex-1 resize-none rounded-xl border border-gray-700 bg-[#1a1d27] px-3 py-2 text-sm text-gray-200 placeholder-gray-600 outline-none focus:border-violet-500 focus:bg-[#1f2233] transition-colors"
          style={{ maxHeight: '160px' }}
        />
        {isStreaming ? (
          <button
            onClick={onStop}
            className="shrink-0 rounded-xl bg-red-500 px-3 py-2 text-white hover:bg-red-600 transition-colors text-xs font-medium"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!text.trim()}
            className="shrink-0 rounded-xl bg-violet-600 p-2 text-white hover:bg-violet-700 disabled:opacity-40 transition-colors"
          >
            <Send size={18} />
          </button>
        )}
      </div>
    </div>
  )
}
