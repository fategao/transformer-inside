import { argmax, mulberry32, normalize } from './math'
import type { SampleResult, SamplingOptions } from './types'

export function temperatureProbabilities(logits: ArrayLike<number>, temperature: number, topK: number): number[] {
  const scaled = Array.from(logits, (value) => value / Math.max(temperature, 1e-6))
  const ranked = Array.from(scaled.keys()).sort((a, b) => scaled[b] - scaled[a])
  const keep = new Set(ranked.slice(0, Math.max(1, Math.min(topK, scaled.length))))
  const filtered = scaled.map((value, index) => (keep.has(index) ? value : -Infinity))
  return normalize(filtered)
}

export function sampleFromLogits(logits: ArrayLike<number>, options: SamplingOptions): SampleResult {
  const probabilities = temperatureProbabilities(logits, options.temperature, options.topK)
  const random = mulberry32(options.seed)
  const draw = random()
  let cumulative = 0
  let chosenIndex = probabilities.length - 1
  for (let i = 0; i < probabilities.length; i += 1) {
    cumulative += probabilities[i]
    if (draw <= cumulative) {
      chosenIndex = i
      break
    }
  }
  return { probabilities, chosenIndex, greedyIndex: argmax(logits) }
}

export function topEntries(probabilities: number[], count = 8): Array<{ index: number; probability: number }> {
  return Array.from(probabilities.keys())
    .sort((a, b) => probabilities[b] - probabilities[a])
    .slice(0, count)
    .map((index) => ({ index, probability: probabilities[index] }))
}
