export function heatColor(value: number, max = 1): string {
  const t = Math.max(0, Math.min(1, value / (max || 1)))
  const low = [248, 246, 239]
  const high = [47, 111, 237]
  const mix = low.map((channel, index) => Math.round(channel + (high[index] - channel) * Math.pow(t, 0.72)))
  return `rgb(${mix.join(',')})`
}

export function signedColor(value: number, max = 1): string {
  const t = Math.max(-1, Math.min(1, value / (max || 1)))
  if (t >= 0) {
    const mix = [255, 255, 255].map((channel, index) => Math.round(channel + ([47, 111, 237][index] - channel) * t))
    return `rgb(${mix.join(',')})`
  }
  const mix = [255, 255, 255].map((channel, index) => Math.round(channel + ([224, 82, 99][index] - channel) * -t))
  return `rgb(${mix.join(',')})`
}

export function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs >= 1e6) return `${(value / 1e6).toFixed(digits)}M`
  if (abs >= 1e3) return `${(value / 1e3).toFixed(digits)}k`
  return value.toFixed(digits)
}
