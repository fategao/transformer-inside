import type { SceneData } from './SceneController'

export function makePlaceholderData(prompt = 'ROMEO: what light'): SceneData {
  const tokens = Array.from(prompt)
  const attention = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, (_, head) =>
      Array.from({ length: tokens.length }, (_, query) => {
        const row = Array.from({ length: tokens.length }, () => 0)
        const span = query + 1
        for (let key = 0; key <= query; key += 1) {
          const distance = query - key
          row[key] = Math.exp(-distance * (0.22 + head * 0.12)) / span
        }
        const sum = row.reduce((acc, value) => acc + value, 0)
        return row.map((value) => (sum > 0 ? value / sum : 0))
      }),
    ),
  )
  const probabilities = Array.from({ length: 65 }, (_, index) => {
    if (index === 1) return 0.32
    if (index === 46) return 0.14
    return 0.54 / 63
  })
  return {
    tokens,
    attention,
    probabilities,
    vocab: [],
    embedding: tokens.map((_, token) =>
      Array.from({ length: 8 }, (_, dim) => Math.sin((token + 1) * (dim + 2) * 0.7)),
    ),
  }
}
