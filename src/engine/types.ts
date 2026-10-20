export type ModelConfig = {
  vocabSize: number
  contextLength: number
  nLayer: number
  nHead: number
  dModel: number
  dFF: number
}

export type TensorManifest = {
  name: string
  shape: number[]
  offset: number
  count: number
}

export type TrainingMetrics = {
  steps: number
  batchSize: number
  blockSize: number
  learningRate: number
  seed: number
  device: string
  trainLoss: number
  valLoss: number
  parameterCount: number
  corpusCharacters: number
  corpusSha256: string
  trainingSeconds: number
}

export type ModelManifest = {
  name: string
  config: ModelConfig
  vocab: string[]
  tokenizer: 'character'
  weights: {
    file: string
    dtype: 'float32'
    sha256: string
    tensors: TensorManifest[]
  }
  training: TrainingMetrics
  sample: string
}

export type LayerTrace = {
  ln1: Float32Array
  qkv: Float32Array
  attention: Float32Array
  attnOut: Float32Array
  residualIn: Float32Array
  residualMid: Float32Array
  ln2: Float32Array
  mlpHidden: Float32Array
  mlpOut: Float32Array
  residualOut: Float32Array
}

export type ForwardTrace = {
  tokens: string[]
  tokenIds: number[]
  seq: number
  dim: number
  layers: LayerTrace[]
  embeddings: Float32Array
  finalNorm: Float32Array
  logits: Float32Array
}

export type SamplingOptions = {
  temperature: number
  topK: number
  seed: number
}

export type SampleResult = {
  probabilities: number[]
  chosenIndex: number
  greedyIndex: number
}

export type GenerateResult = {
  tokenIds: number[]
  text: string
}

export type WorkerRequest =
  | { id: number; type: 'init'; baseUrl: string }
  | { id: number; type: 'forward'; tokenIds: number[] }
  | { id: number; type: 'sample'; tokenIds: number[]; options: SamplingOptions }
  | { id: number; type: 'generate'; tokenIds: number[]; maxNewTokens: number; options: SamplingOptions }

export type WorkerResponse =
  | { id: number; ok: true; type: 'init'; manifest: ModelManifest }
  | { id: number; ok: true; type: 'forward'; trace: ForwardTrace }
  | { id: number; ok: true; type: 'sample'; result: SampleResult }
  | { id: number; ok: true; type: 'generate'; result: GenerateResult }
  | { id: number; ok: false; error: string }
