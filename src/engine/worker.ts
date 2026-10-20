/// <reference lib="webworker" />
import { TransformerModel, parseWeights } from './model'
import { sampleFromLogits } from './sampling'
import { createTokenizer } from './tokenizer'
import type { ModelManifest, WorkerRequest, WorkerResponse } from './types'

let model: TransformerModel | null = null
let tokenizer: ReturnType<typeof createTokenizer> | null = null

async function init(baseUrl: string) {
  const manifestResponse = await fetch(`${baseUrl}model.json`)
  if (!manifestResponse.ok) throw new Error(`无法加载 model.json (${manifestResponse.status})`)
  const manifest = (await manifestResponse.json()) as ModelManifest
  const weightResponse = await fetch(`${baseUrl}${manifest.weights.file}`)
  if (!weightResponse.ok) throw new Error(`无法加载权重 (${weightResponse.status})`)
  const buffer = await weightResponse.arrayBuffer()
  model = new TransformerModel(manifest, parseWeights(manifest, buffer))
  tokenizer = createTokenizer(manifest.vocab)
  return manifest
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  try {
    if (request.type === 'init') {
      const manifest = await init(request.baseUrl)
      const response: WorkerResponse = { id: request.id, ok: true, type: 'init', manifest }
      self.postMessage(response)
      return
    }
    if (!model || !tokenizer) throw new Error('模型尚未初始化')
    if (request.type === 'forward') {
      const trace = model.forward(request.tokenIds)
      const response: WorkerResponse = { id: request.id, ok: true, type: 'forward', trace }
      const transfer = [
        trace.embeddings.buffer,
        trace.finalNorm.buffer,
        trace.logits.buffer,
        ...trace.layers.flatMap((layer) => [
          layer.ln1.buffer,
          layer.qkv.buffer,
          layer.attention.buffer,
          layer.attnOut.buffer,
          layer.residualIn.buffer,
          layer.residualMid.buffer,
          layer.ln2.buffer,
          layer.mlpHidden.buffer,
          layer.mlpOut.buffer,
          layer.residualOut.buffer,
        ]),
      ]
      self.postMessage(response, { transfer })
      return
    }
    if (request.type === 'sample') {
      const trace = model.forward(request.tokenIds)
      const result = sampleFromLogits(trace.logits, request.options)
      const response: WorkerResponse = { id: request.id, ok: true, type: 'sample', result }
      self.postMessage(response)
      return
    }
    if (request.type === 'generate') {
      const ids = request.tokenIds.slice(-model.config.contextLength)
      for (let step = 0; step < request.maxNewTokens; step += 1) {
        const trace = model.forward(ids)
        const result = sampleFromLogits(trace.logits, { ...request.options, seed: request.options.seed + step })
        ids.push(result.chosenIndex)
        if (ids.length > model.config.contextLength) ids.shift()
      }
      const response: WorkerResponse = {
        id: request.id,
        ok: true,
        type: 'generate',
        result: { tokenIds: ids, text: tokenizer.decode(ids) },
      }
      self.postMessage(response)
    }
  } catch (error) {
    const response: WorkerResponse = {
      id: request.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }
    self.postMessage(response)
  }
}

