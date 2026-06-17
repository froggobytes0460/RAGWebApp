import { Trash2, FileText } from 'lucide-react'

interface DocumentItemProps {
  filename: string
  onDelete: () => void
  isDeleting?: boolean
}

export function DocumentItem({ filename, onDelete, isDeleting = false }: DocumentItemProps) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-gray-100 bg-white px-3 py-2 text-sm">
      <FileText size={14} className="shrink-0 text-gray-400" />
      <span className="flex-1 truncate text-gray-700" title={filename}>
        {filename}
      </span>
      <button
        onClick={onDelete}
        disabled={isDeleting}
        className="shrink-0 rounded p-0.5 text-gray-400 hover:bg-red-50 hover:text-red-500 disabled:opacity-50 transition-colors"
        title="Delete document"
      >
        <Trash2 size={14} />
      </button>
    </div>
  )
}
