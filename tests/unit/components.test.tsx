import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { BarList } from '../../src/components/BarList'
import { Heatmap } from '../../src/components/Heatmap'
import { chapters } from '../../src/content/chapters'
import { formatBytes, formatFlops } from '../../src/engine/kv-cache'

describe('content model', () => {
  it('has nine ordered chapters with unique ids', () => {
    expect(chapters).toHaveLength(9)
    expect(new Set(chapters.map((chapter) => chapter.id)).size).toBe(9)
    chapters.forEach((chapter, index) => expect(chapter.index).toBe(index))
  })

  it('renders a labeled heatmap', () => {
    const { container } = render(
      <Heatmap
        values={[
          [1, 0],
          [0.5, 0.25],
        ]}
        label="测试注意力"
      />,
    )
    expect(screen.getByRole('img', { name: '测试注意力' })).toBeInTheDocument()
    expect(container.querySelectorAll('.heatmap__cell')).toHaveLength(4)
  })

  it('renders probability bars with formatted values', () => {
    render(
      <BarList
        entries={[
          { label: 'A', value: 0.5 },
          { label: 'B', value: 0.25 },
        ]}
        format={(value) => `${(value * 100).toFixed(0)}%`}
      />,
    )
    expect(screen.getByText('A')).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()
  })
})

describe('format helpers', () => {
  it('formats bytes and FLOPs for the ledger', () => {
    expect(formatBytes(262144)).toBe('256.0 KB')
    expect(formatFlops(1_500_000)).toBe('1.50 MFLOPs')
    expect(formatFlops(2_000_000_000)).toBe('2.00 GFLOPs')
  })
})
