import { X } from 'lucide-react'

interface ErrorBannerProps {
  message: string
  onDismiss?: () => void
}

export function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-red-800 bg-red-900/30 px-3 py-2 text-sm text-red-400">
      <span className="flex-1">{message}</span>
      {onDismiss && (
        <button onClick={onDismiss} className="shrink-0 hover:text-red-900">
          <X size={14} />
        </button>
      )}
    </div>
  )
}
