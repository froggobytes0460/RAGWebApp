import { useRef, useState, useCallback } from 'react'
import { Upload } from 'lucide-react'
import { ErrorBanner } from '../ui/ErrorBanner'
import type { UseMutationResult } from '@tanstack/react-query'
import type { IngestJobResponse } from '../../types/api'

const ACCEPTED = '.pdf,.docx,.md,.xlsx'
const MAX_MB = 50

interface UploadZoneProps {
  uploadMutation: UseMutationResult<IngestJobResponse, Error, File>
  uploadProgress: number | null
}

export function UploadZone({ uploadMutation, uploadProgress }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)

  const handleFile = useCallback(
    (file: File) => {
      setValidationError(null)
      if (file.size > MAX_MB * 1024 * 1024) {
        setValidationError(`File exceeds ${MAX_MB}MB limit.`)
        return
      }
      uploadMutation.mutate(file)
    },
    [uploadMutation],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    e.target.value = ''
  }

  const isUploading = uploadMutation.isPending
  const error = validationError ?? (uploadMutation.isError ? uploadMutation.error.message : null)

  return (
    <div className="flex flex-col gap-2">
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={() => !isUploading && inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors
          ${isDragging ? 'border-violet-500 bg-violet-900/20' : 'border-gray-700 bg-[#1a1d27] hover:border-gray-600'}
          ${isUploading ? 'pointer-events-none opacity-60' : ''}`}
      >
        <Upload size={20} className="mb-2 text-gray-500" />
        <p className="text-sm text-gray-400">
          {isUploading ? 'Uploading…' : 'Drop a file or click to upload'}
        </p>
        <p className="mt-1 text-xs text-gray-500">PDF, DOCX, MD, XLSX · max {MAX_MB}MB</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className="hidden"
          onChange={onInputChange}
          disabled={isUploading}
        />
      </div>

      {uploadProgress !== null && (
        <div className="h-1.5 w-full rounded-full bg-gray-700 overflow-hidden">
          <div
            className="h-full rounded-full bg-violet-500 transition-all duration-150"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}

      {error && (
        <ErrorBanner message={error} onDismiss={() => { setValidationError(null); uploadMutation.reset() }} />
      )}
    </div>
  )
}
