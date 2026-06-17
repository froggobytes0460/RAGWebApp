import { X } from 'lucide-react'
import { UploadZone } from '../documents/UploadZone'
import { DocumentList } from '../documents/DocumentList'
import { useDocuments } from '../../hooks/useDocuments'

interface DocumentDrawerProps {
  sessionId: string
  open: boolean
  onClose: () => void
}

export function DocumentDrawer({ sessionId, open, onClose }: DocumentDrawerProps) {
  const { query, uploadMutation, deleteMutation, uploadProgress } = useDocuments(sessionId)

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-10 bg-black/10"
          onClick={onClose}
        />
      )}
      <aside
        className={`fixed right-0 top-0 z-20 flex h-full w-72 flex-col border-l border-gray-800 bg-[#13151e] shadow-xl transition-transform duration-200
          ${open ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <span className="text-sm font-semibold text-gray-200">Documents</span>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
            <X size={16} />
          </button>
        </div>
        <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-4">
          <UploadZone uploadMutation={uploadMutation} uploadProgress={uploadProgress} />
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-500">
              Uploaded files
            </p>
            <DocumentList query={query} deleteMutation={deleteMutation} />
          </div>
        </div>
      </aside>
    </>
  )
}
