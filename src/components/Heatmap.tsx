import { heatColor } from '../lib/colors'

type Props = {
  values: number[][]
  max?: number
  label?: string
  rowLabels?: string[]
  columnLabels?: string[]
}

export function Heatmap({ values, max = 1, label, rowLabels, columnLabels }: Props) {
  const columns = values[0]?.length ?? 0
  return (
    <div className="heatmap-wrap">
      {columnLabels && columns > 0 && (
        <div className="heatmap-axis heatmap-axis--top" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
          {columnLabels.map((item, index) => (
            <span key={`${item}-${index}`}>{item}</span>
          ))}
        </div>
      )}
      <div className="heatmap-row">
        {rowLabels && (
          <div className="heatmap-axis heatmap-axis--left">
            {rowLabels.map((item, index) => (
              <span key={`${item}-${index}`}>{item}</span>
            ))}
          </div>
        )}
        <div
          className="heatmap"
          style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
          role="img"
          aria-label={label ?? '数值热力图'}
        >
          {values.flatMap((row, rowIndex) =>
            row.map((value, columnIndex) => (
              <span
                key={`${rowIndex}-${columnIndex}`}
                className="heatmap__cell"
                style={{ background: heatColor(value, max) }}
                title={`第 ${rowIndex + 1} 行，第 ${columnIndex + 1} 列：${value.toFixed(3)}`}
              />
            )),
          )}
        </div>
      </div>
    </div>
  )
}

export function MiniMatrix({
  values,
  rowLabels,
  label,
}: {
  values: number[][]
  rowLabels?: string[]
  label?: string
}) {
  let max = 0
  values.forEach((row) => row.forEach((value) => (max = Math.max(max, Math.abs(value)))))
  return (
    <div className="mini-matrix-wrap">
      {rowLabels && (
        <div className="mini-matrix-labels">
          {rowLabels.map((item, index) => (
            <span key={`${item}-${index}`}>{item}</span>
          ))}
        </div>
      )}
      <div
        className="mini-matrix"
        style={{ gridTemplateColumns: `repeat(${values[0]?.length ?? 1}, minmax(0, 1fr))` }}
        role="img"
        aria-label={label ?? '向量矩阵'}
      >
        {values.flatMap((row, rowIndex) =>
          row.map((value, columnIndex) => {
            const t = max > 0 ? value / max : 0
            const alpha = Math.min(1, Math.abs(t))
            const color = t >= 0 ? `rgba(47,111,237,${alpha})` : `rgba(224,82,99,${alpha})`
            return (
              <span
                key={`${rowIndex}-${columnIndex}`}
                style={{ background: color }}
                title={value.toFixed(3)}
              />
            )
          }),
        )}
      </div>
    </div>
  )
}
