import { dot, gelu, layerNorm, linear, softmaxInPlace } from './math'
import type { ForwardTrace, LayerTrace, ModelConfig, ModelManifest } from './types'

type WeightMap = Map<string, Float32Array>

export class TransformerModel {
  readonly manifest: ModelManifest
  readonly config: ModelConfig
  private readonly weights: WeightMap

  constructor(manifest: ModelManifest, weights: WeightMap) {
    this.manifest = manifest
    this.config = manifest.config
    this.weights = weights
  }

  private weight(name: string): Float32Array {
    const value = this.weights.get(name)
    if (!value) throw new Error(`missing tensor: ${name}`)
    return value
  }

  forward(tokenIds: number[]): ForwardTrace {
    const { nLayer, nHead, dModel, dFF, contextLength } = this.config
    const ids = tokenIds.slice(-contextLength)
    const seq = ids.length
    if (seq === 0) throw new Error('tokenIds must not be empty')
    const headDim = dModel / nHead
    const tokEmb = this.weight('tok_emb.weight')
    const posEmb = this.weight('pos_emb.weight')

    let x = new Float32Array(seq * dModel)
    for (let t = 0; t < seq; t += 1) {
      const tokenBase = ids[t] * dModel
      const posBase = t * dModel
      for (let d = 0; d < dModel; d += 1) {
        x[t * dModel + d] = tokEmb[tokenBase + d] + posEmb[posBase + d]
      }
    }
    const embeddings = x.slice()
    const layers: LayerTrace[] = []

    for (let layer = 0; layer < nLayer; layer += 1) {
      const prefix = `blocks.${layer}`
      const residualIn = x.slice()
      const ln1 = layerNorm(x, this.weight(`${prefix}.ln1.weight`), this.weight(`${prefix}.ln1.bias`), seq, dModel)
      const qkv = linear(ln1, this.weight(`${prefix}.attn.qkv.weight`), this.weight(`${prefix}.attn.qkv.bias`), seq, dModel, dModel * 3)
      const attention = new Float32Array(nHead * seq * seq)
      const attnOut = new Float32Array(seq * dModel)

      for (let head = 0; head < nHead; head += 1) {
        const headBase = head * headDim
        for (let query = 0; query < seq; query += 1) {
          const scores = new Float32Array(seq)
          for (let key = 0; key <= query; key += 1) {
            let score = 0
            for (let i = 0; i < headDim; i += 1) {
              score +=
                qkv[query * dModel * 3 + headBase + i] *
                qkv[key * dModel * 3 + dModel + headBase + i]
            }
            scores[key] = score / Math.sqrt(headDim)
          }
          for (let key = query + 1; key < seq; key += 1) scores[key] = -Infinity
          softmaxInPlace(scores)
          attention.set(scores, head * seq * seq + query * seq)
          for (let i = 0; i < headDim; i += 1) {
            let value = 0
            for (let key = 0; key <= query; key += 1) {
              value +=
                scores[key] *
                qkv[key * dModel * 3 + dModel * 2 + headBase + i]
            }
            attnOut[query * dModel + headBase + i] = value
          }
        }
      }

      const attnProjected = linear(attnOut, this.weight(`${prefix}.attn.proj.weight`), this.weight(`${prefix}.attn.proj.bias`), seq, dModel, dModel)
      const residualMid = new Float32Array(seq * dModel)
      for (let i = 0; i < residualMid.length; i += 1) residualMid[i] = residualIn[i] + attnProjected[i]

      const ln2 = layerNorm(residualMid, this.weight(`${prefix}.ln2.weight`), this.weight(`${prefix}.ln2.bias`), seq, dModel)
      const preActivation = linear(ln2, this.weight(`${prefix}.fc.weight`), this.weight(`${prefix}.fc.bias`), seq, dModel, dFF)
      const mlpHidden = new Float32Array(preActivation.length)
      for (let i = 0; i < preActivation.length; i += 1) mlpHidden[i] = gelu(preActivation[i])
      const mlpOut = linear(mlpHidden, this.weight(`${prefix}.mlp_proj.weight`), this.weight(`${prefix}.mlp_proj.bias`), seq, dFF, dModel)
      const residualOut = new Float32Array(seq * dModel)
      for (let i = 0; i < residualOut.length; i += 1) residualOut[i] = residualMid[i] + mlpOut[i]

      layers.push({ ln1, qkv, attention, attnOut, residualIn, residualMid, ln2, mlpHidden, mlpOut, residualOut })
      x = residualOut
    }

    const finalNorm = layerNorm(x, this.weight('ln_f.weight'), this.weight('ln_f.bias'), seq, dModel)
    const lastBase = (seq - 1) * dModel
    const logits = new Float32Array(this.config.vocabSize)
    for (let token = 0; token < this.config.vocabSize; token += 1) {
      logits[token] = dot(finalNorm, lastBase, tokEmb, token * dModel, dModel)
    }

    return {
      tokens: ids.map((id) => this.manifest.vocab[id] ?? ''),
      tokenIds: ids,
      seq,
      dim: dModel,
      layers,
      embeddings,
      finalNorm,
      logits,
    }
  }

  probabilities(trace: ForwardTrace, temperature = 1): number[] {
    const scaled = new Float32Array(trace.logits.length)
    for (let i = 0; i < trace.logits.length; i += 1) scaled[i] = trace.logits[i] / Math.max(temperature, 1e-6)
    softmaxInPlace(scaled)
    return Array.from(scaled)
  }
}

export function parseWeights(manifest: ModelManifest, buffer: ArrayBuffer): WeightMap {
  const map: WeightMap = new Map()
  for (const tensor of manifest.weights.tensors) {
    const [rows, cols] = tensor.shape.length === 2 ? tensor.shape : [tensor.count, 1]
    map.set(tensor.name, new Float32Array(buffer, tensor.offset * 4, rows * cols))
  }
  return map
}

export function attentionFor(trace: ForwardTrace, layer: number, head: number): Float32Array {
  const seq = trace.seq
  return trace.layers[layer].attention.slice(head * seq * seq, (head + 1) * seq * seq)
}
