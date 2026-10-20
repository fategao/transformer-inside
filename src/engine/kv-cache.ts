export type KvCacheEstimate = {
  contextLength: number
  layers: number
  heads: number
  headDim: number
  fullFlopsPerStep: number
  cachedFlopsPerStep: number
  cacheBytes: number
  savedFraction: number
}

export function estimateKvCache(contextLength: number, layers: number, heads: number, headDim: number): KvCacheEstimate {
  const dModel = heads * headDim
  const attentionProjection = 4 * contextLength * dModel * dModel
  const attentionScores = 4 * heads * contextLength * contextLength * headDim
  // The MLP is position-wise, so a full recompute pays for every context
  // position while a cached decode step pays for the new token only.
  const mlpFull = 2 * contextLength * dModel * (4 * dModel)
  const fullFlopsPerStep = layers * (attentionProjection + attentionScores + mlpFull)
  const cachedProjection = layers * (4 * dModel * dModel)
  const cachedScores = layers * (4 * heads * contextLength * headDim)
  const cachedMlp = layers * (2 * dModel * (4 * dModel))
  const cachedFlopsPerStep = cachedProjection + cachedScores + cachedMlp
  const cacheBytes = 2 * layers * contextLength * dModel * 4
  return {
    contextLength,
    layers,
    heads,
    headDim,
    fullFlopsPerStep,
    cachedFlopsPerStep,
    cacheBytes,
    savedFraction: 1 - cachedFlopsPerStep / fullFlopsPerStep,
  }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export function formatFlops(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} GFLOPs`
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} MFLOPs`
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)} KFLOPs`
  return `${value.toFixed(0)} FLOPs`
}
