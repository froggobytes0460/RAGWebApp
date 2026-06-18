import { useState, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'

interface CollapsibleProps {
  trigger: ReactNode
  children: ReactNode
  defaultOpen?: boolean
  className?: string
}

export function Collapsible({ trigger, children, defaultOpen = false, className = '' }: CollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className={className}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1 text-left text-sm font-medium text-gray-500 hover:text-gray-300 transition-colors"
      >
        <ChevronDown
          size={14}
          className={`shrink-0 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        />
        {trigger}
      </button>
      <div
        className={`overflow-hidden transition-all duration-200 ${open ? 'max-h-[2000px] opacity-100 mt-2' : 'max-h-0 opacity-0'}`}
      >
        {children}
      </div>
    </div>
  )
}
