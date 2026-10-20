import { useMemo, useState } from 'react'
import { sampleFromLogits, temperatureProbabilities, topEntries } from '../engine/sampling'
import type { ForwardTrace, ModelManifest } from '../engine/types'
import { BarList } from './BarList'
import { Heatmap, MiniMatrix } from './Heatmap'

type Props = {
  kind: string
  trace: ForwardTrace | null
  manifest: ModelManifest | null
  selectedToken: number
}

function display(token: string) {
  return token === ' ' ? '␣' : token === '\n' ? '↵' : token
}

function rowOf(values: Float32Array, row: number, dim: number) {
  return Array.from(values.subarray(row * dim, (row + 1) * dim))
}

function normOf(values: Float32Array, row: number, dim: number) {
  let sum = 0
  for (let i = 0; i < dim; i += 1) sum += values[row * dim + i] ** 2
  return Math.sqrt(sum)
}

function cosine(a: number[], b: number[]) {
  let dot = 0
  let aa = 0
  let bb = 0
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i]
    aa += a[i] * a[i]
    bb += b[i] * b[i]
  }
  return dot / Math.sqrt(aa * bb || 1)
}

function PanelLoading() {
  return (
    <div className="panel-loading" role="status">
      <span className="spinner" /> 正在用真实权重计算这一章…
    </div>
  )
}

export function ChapterPanel({ kind, trace, manifest, selectedToken }: Props) {
  if (!trace || !manifest) return <PanelLoading />
  const tokenIndex = Math.max(0, Math.min(selectedToken, trace.seq - 1))
  const dim = trace.dim

  if (kind === 'tokens') {
    return (
      <div className="panel-stack">
        <div className="token-table" role="table" aria-label="字符与词表 ID">
          <div className="token-table__head" role="row">
            <span>位置</span>
            <span>字符</span>
            <span>ID</span>
          </div>
          {trace.tokens.map((token, index) => (
            <div className={`token-table__row ${index === tokenIndex ? 'is-selected' : ''}`} role="row" key={index}>
              <span>{index}</span>
              <span className="mono">{display(token)}</span>
              <span className="mono">{trace.tokenIds[index]}</span>
            </div>
          ))}
        </div>
        <p className="panel-note">词表共 {manifest.config.vocabSize} 个字符；空格也是一个独立 token。</p>
      </div>
    )
  }

  if (kind === 'embedding') {
    const rows = Math.min(7, trace.seq)
    const matrix = Array.from({ length: rows }, (_, row) => rowOf(trace.embeddings, row, dim).slice(0, 16))
    return (
      <div className="panel-stack">
        <MiniMatrix
          values={matrix}
          rowLabels={trace.tokens.slice(0, rows).map((token, index) => `${index} ${display(token)}`)}
          label="前 7 个 token 的嵌入向量前 16 维"
        />
        <p className="panel-note">每一行是一个 token 的 128 维向量；这里只展示前 16 维。颜色越深，绝对值越大。</p>
        <BarList
          entries={trace.tokens.slice(0, rows).map((token, index) => ({
            label: `${index} ${display(token)}`,
            value: normOf(trace.embeddings, index, dim),
            color: 'var(--blue)',
          }))}
          format={(value) => value.toFixed(2)}
        />
      </div>
    )
  }

  if (kind === 'qkv') {
    const qkv = trace.layers[0].qkv
    const q = Array.from(qkv.subarray(tokenIndex * dim * 3, tokenIndex * dim * 3 + dim))
    const k = Array.from(qkv.subarray(tokenIndex * dim * 3 + dim, tokenIndex * dim * 3 + dim * 2))
    const v = Array.from(qkv.subarray(tokenIndex * dim * 3 + dim * 2, tokenIndex * dim * 3 + dim * 3))
    const norm = (values: number[]) => Math.sqrt(values.reduce((acc, value) => acc + value * value, 0))
    return (
      <div className="panel-stack">
        <p className="panel-note">当前查看第 {tokenIndex} 个 token（{display(trace.tokens[tokenIndex])}）的第 1 层投影。</p>
        <BarList
          entries={[
            { label: 'Query', value: norm(q), color: 'var(--red)' },
            { label: 'Key', value: norm(k), color: 'var(--green)' },
            { label: 'Value', value: norm(v), color: 'var(--amber)' },
          ]}
          format={(value) => value.toFixed(2)}
        />
        <div className="vector-strips">
          {[
            ['Q', q, 'var(--red)'],
            ['K', k, 'var(--green)'],
            ['V', v, 'var(--amber)'],
          ].map(([label, values, color]) => (
            <div key={label as string} className="vector-strip">
              <span>{label as string}</span>
              <div>
                {(values as number[]).slice(0, 18).map((value, index) => (
                  <i
                    key={index}
                    style={{
                      height: `${Math.min(100, Math.abs(value) * 18 + 4)}%`,
                      background: color as string,
                      opacity: value < 0 ? 0.35 : 0.9,
                    }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
        <p className="panel-note">Q/K/V 都来自同一个归一化向量，只是使用了三组不同的投影权重。</p>
      </div>
    )
  }

  if (kind === 'attention') {
    return <AttentionPanel trace={trace} />
  }

  if (kind === 'residual') {
    return <ResidualPanel trace={trace} />
  }

  if (kind === 'mlp') {
    return <MlpPanel trace={trace} />
  }

  if (kind === 'stack') {
    return <StackPanel trace={trace} />
  }

  if (kind === 'logits') {
    const probabilities = temperatureProbabilities(trace.logits, 1, manifest.config.vocabSize)
    const top = topEntries(probabilities, 9)
    return (
      <div className="panel-stack">
        <BarList
          entries={top.map((entry) => ({
            label: display(manifest.vocab[entry.index] ?? ''),
            value: entry.probability,
            color: 'var(--violet)',
            note: `logit ${trace.logits[entry.index].toFixed(2)}`,
          }))}
          format={(value) => `${(value * 100).toFixed(1)}%`}
        />
        <p className="panel-note">
          最后一个位置（{display(trace.tokens[trace.seq - 1])}）的 65 个 logits 经 Softmax 后得到这些概率，总和为 1。
        </p>
      </div>
    )
  }

  if (kind === 'sampling') {
    return <SamplingPanel trace={trace} manifest={manifest} />
  }

  return null
}

function AttentionPanel({ trace }: { trace: ForwardTrace }) {
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const seq = trace.seq
  const values = useMemo(() => {
    const attention = trace.layers[layer].attention
    return Array.from({ length: seq }, (_, query) =>
      Array.from({ length: seq }, (_, key) => attention[head * seq * seq + query * seq + key]),
    )
  }, [trace, layer, head, seq])
  const lastRow = values[seq - 1]
  const top = topEntries(Array.from(lastRow), 5)
  return (
    <div className="panel-stack">
      <div className="inline-controls">
        <label>
          层
          <select value={layer} onChange={(event) => setLayer(Number(event.target.value))}>
            {[0, 1, 2, 3].map((value) => (
              <option key={value} value={value}>
                L{value + 1}
              </option>
            ))}
          </select>
        </label>
        <label>
          头
          <select value={head} onChange={(event) => setHead(Number(event.target.value))}>
            {[0, 1, 2, 3].map((value) => (
              <option key={value} value={value}>
                H{value + 1}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Heatmap values={values} label={`第 ${layer + 1} 层第 ${head + 1} 个头的注意力权重`} />
      <div className="legend-row">
        <span>查询行 →</span>
        <span>键列 ↑</span>
        <span className="legend-swatch" /> 颜色越深，权重越大
      </div>
      <BarList
        entries={top.map((entry) => ({
          label: `${entry.index} ${display(trace.tokens[entry.index])}`,
          value: entry.probability,
          color: 'var(--blue)',
        }))}
        format={(value) => `${(value * 100).toFixed(1)}%`}
      />
      <p className="panel-note">右上角是空白，因为因果掩码禁止当前位置看到未来 token。</p>
    </div>
  )
}

function ResidualPanel({ trace }: { trace: ForwardTrace }) {
  const [layer, setLayer] = useState(0)
  const dim = trace.dim
  const row = trace.seq - 1
  const block = trace.layers[layer]
  const before = normOf(block.residualIn, row, dim)
  const middle = normOf(block.residualMid, row, dim)
  const after = normOf(block.residualOut, row, dim)
  const lnValues = rowOf(block.ln2, row, dim)
  const mean = lnValues.reduce((acc, value) => acc + value, 0) / dim
  const variance = lnValues.reduce((acc, value) => acc + (value - mean) ** 2, 0) / dim
  return (
    <div className="panel-stack">
      <div className="inline-controls">
        <label>
          层
          <select value={layer} onChange={(event) => setLayer(Number(event.target.value))}>
            {[0, 1, 2, 3].map((value) => (
              <option key={value} value={value}>
                L{value + 1}
              </option>
            ))}
          </select>
        </label>
      </div>
      <BarList
        entries={[
          { label: '进入层', value: before, color: 'var(--slate)' },
          { label: '注意力后', value: middle, color: 'var(--blue)' },
          { label: 'MLP 后', value: after, color: 'var(--violet)' },
        ]}
        format={(value) => value.toFixed(2)}
      />
      <p className="panel-note">
        LayerNorm 输出的均值 {mean.toFixed(3)}，方差 {variance.toFixed(3)}；归一化后再乘可学习的 γ、加 β。
      </p>
      <MiniMatrix
        values={[lnValues.slice(0, 32), rowOf(block.residualOut, row, dim).slice(0, 32)]}
        rowLabels={['LN 输出', '残差输出']}
        label="LayerNorm 输出与残差输出的前 32 维"
      />
      <p className="panel-note">残差把原向量原样加回，因此即使子层输出很小，信息也不会被完全覆盖。</p>
    </div>
  )
}

function MlpPanel({ trace }: { trace: ForwardTrace }) {
  const [layer, setLayer] = useState(0)
  const block = trace.layers[layer]
  const row = trace.seq - 1
  const dFF = block.mlpHidden.length / trace.seq
  const values = Array.from(block.mlpHidden.subarray(row * dFF, (row + 1) * dFF))
  const max = Math.max(...values.map(Math.abs), 1e-6)
  const bins = Array.from({ length: 9 }, (_, index) => {
    const low = -max + (index / 8) * max * 2
    const high = -max + ((index + 1) / 8) * max * 2
    return values.filter((value) => value >= low && (index === 8 ? value <= high : value < high)).length
  })
  const top = Array.from(values.keys())
    .sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]))
    .slice(0, 8)
  return (
    <div className="panel-stack">
      <div className="inline-controls">
        <label>
          层
          <select value={layer} onChange={(event) => setLayer(Number(event.target.value))}>
            {[0, 1, 2, 3].map((value) => (
              <option key={value} value={value}>
                L{value + 1}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="flow-chips">
        <span>128 维</span>
        <b>→</b>
        <span>512 维</span>
        <b>→</b>
        <span>128 维</span>
      </div>
      <div className="histogram" aria-label="MLP 激活分布">
        {bins.map((count, index) => (
          <span key={index} style={{ height: `${Math.max(4, (count / Math.max(...bins)) * 100)}%` }} title={`${count} 个通道`} />
        ))}
      </div>
      <BarList
        entries={top.map((index) => ({
          label: `#${index}`,
          value: Math.abs(values[index]),
          color: values[index] >= 0 ? 'var(--amber)' : 'var(--red)',
        }))}
        format={(value) => value.toFixed(2)}
      />
      <p className="panel-note">GELU 让负值大幅衰减、正值近似保留；每个位置独立计算，不跨 token 混合。</p>
    </div>
  )
}

function StackPanel({ trace }: { trace: ForwardTrace }) {
  const row = trace.seq - 1
  const embedding = rowOf(trace.embeddings, row, trace.dim)
  const layers = trace.layers.map((layer, index) => ({
    label: `L${index + 1}`,
    norm: normOf(layer.residualOut, row, trace.dim),
    similarity: cosine(embedding, rowOf(layer.residualOut, row, trace.dim)),
  }))
  const points = layers
    .map((layer, index) => `${18 + index * 74},${86 - ((layer.similarity + 1) / 2) * 64}`)
    .join(' ')
  return (
    <div className="panel-stack">
      <BarList
        entries={layers.map((layer) => ({
          label: layer.label,
          value: layer.norm,
          color: 'var(--teal)',
          note: `cos ${layer.similarity.toFixed(3)}`,
        }))}
        format={(value) => value.toFixed(2)}
      />
      <svg viewBox="0 0 280 100" className="mini-chart" role="img" aria-label="各层输出与初始嵌入的余弦相似度">
        <line x1="18" y1="54" x2="278" y2="54" className="mini-chart__axis" />
        <polyline points={points} className="mini-chart__line" />
        {layers.map((layer, index) => (
          <circle
            key={layer.label}
            cx={18 + index * 74}
            cy={86 - ((layer.similarity + 1) / 2) * 64}
            r="4"
            className="mini-chart__dot"
          />
        ))}
      </svg>
      <p className="panel-note">横轴是层，纵轴是最后一个 token 的表示与初始嵌入的余弦相似度。偏离越远，说明这一层改写得越多。</p>
    </div>
  )
}

function SamplingPanel({ trace, manifest }: { trace: ForwardTrace; manifest: ModelManifest }) {
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(20)
  const [seed, setSeed] = useState(7)
  const result = useMemo(
    () => sampleFromLogits(trace.logits, { temperature, topK, seed }),
    [trace, temperature, topK, seed],
  )
  const top = topEntries(result.probabilities, 8)
  const chosen = manifest.vocab[result.chosenIndex] ?? ''
  const greedy = manifest.vocab[result.greedyIndex] ?? ''
  return (
    <div className="panel-stack">
      <div className="range-grid">
        <label>
          温度 <b>{temperature.toFixed(2)}</b>
          <input type="range" min="0.1" max="2" step="0.05" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
        </label>
        <label>
          top-k <b>{topK}</b>
          <input type="range" min="1" max={manifest.config.vocabSize} value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
        </label>
        <label>
          seed <b>{seed}</b>
          <input type="range" min="1" max="99" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
        </label>
      </div>
      <div className="sample-readout">
        <div>
          <span>这次抽到</span>
          <strong>{display(chosen)}</strong>
          <small>{(result.probabilities[result.chosenIndex] * 100).toFixed(1)}%</small>
        </div>
        <div>
          <span>greedy 选择</span>
          <strong>{display(greedy)}</strong>
          <small>最高概率</small>
        </div>
      </div>
      <BarList
        entries={top.map((entry) => ({
          label: display(manifest.vocab[entry.index] ?? ''),
          value: entry.probability,
          color: 'var(--violet)',
        }))}
        format={(value) => `${(value * 100).toFixed(1)}%`}
      />
      <button type="button" className="button button--quiet" onClick={() => setSeed((value) => (value % 99) + 1)}>
        换一个随机种子
      </button>
      <p className="panel-note">温度缩放改变分布的陡峭程度；top-k 只保留概率最高的 k 个候选，再重新归一化。</p>
    </div>
  )
}
