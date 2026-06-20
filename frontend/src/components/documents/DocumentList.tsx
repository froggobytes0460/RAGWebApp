import { DocumentItem } from './DocumentItem'
import { Spinner } from '../ui/Spinner'
import type { UseMutationResult, UseQueryResult } from '@tanstack/react-query'
import type { DocumentListItem } from '../../types/api'
import type { ActiveJob } from '../../hooks/useDocuments'

interface DocumentListProps {
  query: UseQueryResult<DocumentListItem[]>
  deleteMutation: UseMutationResult<void, Error, string>
  activeJobs: Map<string, ActiveJob>
}

export function DocumentList({ query, deleteMutation, activeJobs }: DocumentListProps) {
  if (query.isLoading) {
    return (
      <div className="flex justify-center py-4">
        <Spinner className="h-5 w-5 text-gray-400" />
      </div>
    )
  }

  if (query.isError) {
    return <p className="text-sm text-red-400">Failed to load documents.</p>
  }

  const docs = query.data ?? []
  const jobs = Array.from(activeJobs.values())

  return (
    <div className="flex flex-col gap-1.5">
      {jobs.map((job) => (
        <div key={job.jobId} className="rounded-md bg-gray-900 px-3 py-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-300 truncate max-w-[75%]">{job.filename}</span>
            {job.status !== 'failed' && (
              <span className="text-xs text-gray-500">{job.progress}%</span>
            )}
          </div>
          {job.status === 'failed' ? (
            <p className="text-xs text-red-400">{job.error ?? 'Ingestion failed'}</p>
          ) : (
            <div className="h-1.5 w-full rounded-full bg-gray-700">
              <div
                className="h-1.5 rounded-full bg-violet-500 transition-all duration-300"
                style={{ width: `${job.progress}%` }}
              />
            </div>
          )}
        </div>
      ))}

      {docs.length === 0 && jobs.length === 0 && (
        <p className="text-center text-sm text-gray-500 py-3">No documents uploaded yet.</p>
      )}

      {docs.map(({ filename }) => (
        <DocumentItem
          key={filename}
          filename={filename}
          onDelete={() => deleteMutation.mutate(filename)}
          isDeleting={deleteMutation.isPending && deleteMutation.variables === filename}
        />
      ))}
    </div>
  )
}
