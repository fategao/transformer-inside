import type { ReactNode } from 'react'

type Entry = { label: ReactNode; value: number; note?: string; color?: string }

export function BarList({ entries, max, format }: { entries: Entry[]; max?: number; format?: (value: number) => string }) {
  const ceiling = max ?? Math.max(...entries.map((entry) => entry.value), 1e-9)
  return (
    <div className="bar-list">
      {entries.map((entry, index) => (
        <div className="bar-list__row" key={index}>
          <span className="bar-list__label">{entry.label}</span>
          <span className="bar-list__track">
            <span
              className="bar-list__fill"
              style={{
                width: `${Math.max(2, (entry.value / ceiling) * 100)}%`,
                background: entry.color ?? 'var(--accent)',
              }}
            />
          </span>
          <span className="bar-list__value">{format ? format(entry.value) : entry.value.toFixed(3)}</span>
          {entry.note && <span className="bar-list__note">{entry.note}</span>}
        </div>
      ))}
    </div>
  )
}
