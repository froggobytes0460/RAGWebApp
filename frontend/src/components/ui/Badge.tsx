interface BadgeProps {
  value: number
  label?: string
  rank?: number
  className?: string
}

export function Badge({ value, label, rank, className = '' }: BadgeProps) {
  const color =
    rank === 1 ? 'bg-green-900/50 text-green-400' :
    rank === 2 ? 'bg-yellow-900/50 text-yellow-400' :
    rank != null && rank <= 4 ? 'bg-blue-900/50 text-blue-400' :
    'bg-gray-800/50 text-gray-400'

  const display = label ?? `${Math.round(value * 100)}%`

  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${color} ${className}`}>
      {display}
    </span>
  )
}
