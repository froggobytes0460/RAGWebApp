import { DocumentItem } from './DocumentItem'
import { Spinner } from '../ui/Spinner'
import type { UseMutationResult, UseQueryResult } from '@tanstack/react-query'
import type { DocumentListItem } from '../../types/api'

interface DocumentListProps {
  query: UseQueryResult<DocumentListItem[]>
  deleteMutation: UseMutationResult<void, Error, string>
}

export function DocumentList({ query, deleteMutation }: DocumentListProps) {
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

  if (docs.length === 0) {
    return (
      <p className="text-center text-sm text-gray-500 py-3">
        No documents uploaded yet.
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-1.5">
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
