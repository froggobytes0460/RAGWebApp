interface BadgeProps {
  value: number
  label?: string
  className?: string
}

export function Badge({ value, label, className = '' }: BadgeProps) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 80 ? 'bg-green-100 text-green-800' :
    pct >= 60 ? 'bg-yellow-100 text-yellow-800' :
    'bg-red-100 text-red-800'

  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${color} ${className}`}>
      {label ?? `${pct}%`}
    </span>
  )
}
