interface BadgeProps {
  value: number
  label?: string
  className?: string
}

export function Badge({ value, label, className = '' }: BadgeProps) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 80 ? 'bg-green-900/50 text-green-400' :
    pct >= 60 ? 'bg-yellow-900/50 text-yellow-400' :
    'bg-red-900/50 text-red-400'

  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${color} ${className}`}>
      {label ?? `${pct}%`}
    </span>
  )
}
