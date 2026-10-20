import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { TransformerModel, parseWeights } from '../../src/engine/model'
import { sampleFromLogits, temperatureProbabilities } from '../../src/engine/sampling'
import { createTokenizer } from '../../src/engine/tokenizer'
import { estimateKvCache } from '../../src/engine/kv-cache'
import type { ModelManifest } from '../../src/engine/types'

const root = process.cwd()
const manifest = JSON.parse(readFileSync(resolve(root, 'public/model/model.json'), 'utf8')) as ModelManifest
const reference = JSON.parse(readFileSync(resolve(root, 'public/model/reference.json'), 'utf8')) as {
  prompt: string
  tokenIds: number[]
  logits: number[]
  attention: number[][][][]
}
const weightBuffer = readFileSync(resolve(root, 'public/model/weights.bin'))
const weights = parseWeights(
  manifest,
  weightBuffer.buffer.slice(weightBuffer.byteOffset, weightBuffer.byteOffset + weightBuffer.byteLength),
)
const model = new TransformerModel(manifest, weights)

describe('browser inference', () => {
  it('matches the PyTorch reference logits', () => {
    const trace = model.forward(reference.tokenIds)
    let maxDiff = 0
    for (let i = 0; i < trace.logits.length; i += 1) {
      maxDiff = Math.max(maxDiff, Math.abs(trace.logits[i] - reference.logits[i]))
    }
    expect(maxDiff).toBeLessThan(2e-4)
  })

  it('matches the PyTorch reference attention weights', () => {
    const trace = model.forward(reference.tokenIds)
    let maxDiff = 0
    reference.attention.forEach((layer, layerIndex) => {
      layer.forEach((head, headIndex) => {
        const actual = trace.layers[layerIndex].attention.slice(
          headIndex * trace.seq * trace.seq,
          (headIndex + 1) * trace.seq * trace.seq,
        )
        head.forEach((row, query) => {
          row.forEach((value, key) => {
            maxDiff = Math.max(maxDiff, Math.abs(actual[query * trace.seq + key] - value))
          })
        })
      })
    })
    expect(maxDiff).toBeLessThan(2e-4)
  })

  it('keeps causal attention rows normalized', () => {
    const trace = model.forward(reference.tokenIds)
    const layer = trace.layers[0]
    for (let head = 0; head < manifest.config.nHead; head += 1) {
      for (let query = 0; query < trace.seq; query += 1) {
        let sum = 0
        for (let key = 0; key < trace.seq; key += 1) {
          const value = layer.attention[head * trace.seq * trace.seq + query * trace.seq + key]
          sum += value
          if (key > query) expect(value).toBe(0)
        }
        expect(sum).toBeCloseTo(1, 5)
      }
    }
  })
})

describe('sampling', () => {
  it('is deterministic for a fixed seed', () => {
    const trace = model.forward(reference.tokenIds)
    const first = sampleFromLogits(trace.logits, { temperature: 0.8, topK: 20, seed: 7 })
    const second = sampleFromLogits(trace.logits, { temperature: 0.8, topK: 20, seed: 7 })
    expect(first.chosenIndex).toBe(second.chosenIndex)
  })

  it('only keeps top-k candidates', () => {
    const probabilities = temperatureProbabilities([1, 2, 3, 4], 1, 2)
    expect(probabilities.filter((value) => value > 0)).toHaveLength(2)
    expect(probabilities[3]).toBeGreaterThan(probabilities[2])
    expect(probabilities[2]).toBeGreaterThan(0)
    expect(probabilities[1]).toBe(0)
    expect(probabilities[0]).toBe(0)
  })
})

describe('tokenizer and KV cache', () => {
  it('round-trips text and drops unknown characters', () => {
    const tokenizer = createTokenizer(manifest.vocab)
    expect(tokenizer.decode(tokenizer.encode('ROMEO: '))).toBe('ROMEO: ')
    expect(tokenizer.sanitize('你好ROMEO')).toBe('ROMEO')
  })

  it('shows the cache saving compute as context grows', () => {
    const short = estimateKvCache(8, 4, 4, 32)
    const long = estimateKvCache(64, 4, 4, 32)
    expect(long.cacheBytes).toBeGreaterThan(short.cacheBytes)
    expect(long.savedFraction).toBeGreaterThan(short.savedFraction)
    expect(long.cachedFlopsPerStep).toBeLessThan(long.fullFlopsPerStep)
  })
})
