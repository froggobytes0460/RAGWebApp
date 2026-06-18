import { Trash2, FileText } from 'lucide-react'

interface DocumentItemProps {
  filename: string
  onDelete: () => void
  isDeleting?: boolean
}

export function DocumentItem({ filename, onDelete, isDeleting = false }: DocumentItemProps) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-gray-700 bg-[#1a1d27] px-3 py-2 text-sm">
      <FileText size={14} className="shrink-0 text-gray-500" />
      <span className="flex-1 truncate text-gray-300" title={filename}>
        {filename}
      </span>
      <button
        onClick={onDelete}
        disabled={isDeleting}
        className="shrink-0 rounded p-0.5 text-gray-500 hover:bg-red-900/40 hover:text-red-400 disabled:opacity-50 transition-colors"
        title="Delete document"
      >
        <Trash2 size={14} />
      </button>
    </div>
  )
}
