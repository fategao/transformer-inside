export function linear(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
  rows: number,
  inDim: number,
  outDim: number,
): Float32Array {
  const output = new Float32Array(rows * outDim)
  for (let row = 0; row < rows; row += 1) {
    const inBase = row * inDim
    const outBase = row * outDim
    for (let out = 0; out < outDim; out += 1) {
      let sum = bias[out]
      const weightBase = out * inDim
      for (let i = 0; i < inDim; i += 1) {
        sum += input[inBase + i] * weight[weightBase + i]
      }
      output[outBase + out] = sum
    }
  }
  return output
}

export function layerNorm(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
  rows: number,
  dim: number,
  eps = 1e-5,
): Float32Array {
  const output = new Float32Array(rows * dim)
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim
    let mean = 0
    for (let i = 0; i < dim; i += 1) mean += input[base + i]
    mean /= dim
    let variance = 0
    for (let i = 0; i < dim; i += 1) {
      const diff = input[base + i] - mean
      variance += diff * diff
    }
    variance /= dim
    const scale = 1 / Math.sqrt(variance + eps)
    for (let i = 0; i < dim; i += 1) {
      output[base + i] = (input[base + i] - mean) * scale * weight[i] + bias[i]
    }
  }
  return output
}

export function gelu(x: number): number {
  const c = Math.sqrt(2 / Math.PI)
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)))
}

export function softmaxInPlace(values: Float32Array, length = values.length): void {
  let max = -Infinity
  for (let i = 0; i < length; i += 1) {
    if (values[i] > max) max = values[i]
  }
  let sum = 0
  for (let i = 0; i < length; i += 1) {
    const value = Math.exp(values[i] - max)
    values[i] = value
    sum += value
  }
  const inv = 1 / sum
  for (let i = 0; i < length; i += 1) values[i] *= inv
}

export function dot(left: Float32Array, leftOffset: number, right: Float32Array, rightOffset: number, length: number): number {
  let sum = 0
  for (let i = 0; i < length; i += 1) sum += left[leftOffset + i] * right[rightOffset + i]
  return sum
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state |= 0
    state = (state + 0x6d2b79f5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function argmax(values: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[best]) best = i
  }
  return best
}

export function normalize(values: ArrayLike<number>): number[] {
  let max = -Infinity
  for (let i = 0; i < values.length; i += 1) max = Math.max(max, values[i])
  const exps = Array.from(values, (value) => Math.exp(value - max))
  const sum = exps.reduce((acc, value) => acc + value, 0)
  return exps.map((value) => value / sum)
}

export function norm(values: Float32Array, offset = 0, length = values.length): number {
  let sum = 0
  for (let i = 0; i < length; i += 1) sum += values[offset + i] * values[offset + i]
  return Math.sqrt(sum)
}
