import { temperatureProbabilities } from '../engine/sampling'
import type { ForwardTrace, ModelManifest } from '../engine/types'
import type { SceneData } from '../scene/SceneController'
import { makePlaceholderData } from '../scene/placeholder'

export function buildSceneData(trace: ForwardTrace | null, manifest: ModelManifest | null): SceneData {
  if (!trace || !manifest) {
    const placeholder = makePlaceholderData()
    return { ...placeholder, vocab: manifest?.vocab ?? [] }
  }
  const seq = trace.seq
  const attention = trace.layers.map((layer) =>
    Array.from({ length: manifest.config.nHead }, (_, head) =>
      Array.from({ length: seq }, (_, query) =>
        Array.from({ length: seq }, (_, key) => layer.attention[head * seq * seq + query * seq + key]),
      ),
    ),
  )
  return {
    tokens: trace.tokens,
    attention,
    probabilities: temperatureProbabilities(trace.logits, 1, manifest.config.vocabSize),
    vocab: manifest.vocab,
    embedding: Array.from({ length: seq }, (_, token) => {
      const values = Array.from({ length: 8 }, (_, dim) => trace.embeddings[token * trace.dim + dim])
      const max = Math.max(...values.map(Math.abs), 1e-6)
      return values.map((value) => value / max)
    }),
  }
}
