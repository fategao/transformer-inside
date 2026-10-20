import { useMemo, useState } from 'react'
import { demoPrompt } from '../../content/chapters'
import { sampleFromLogits, topEntries } from '../../engine/sampling'
import { useModel } from '../../lib/useModel'
import { BarList } from '../BarList'
import { LabLayout } from '../LabLayout'

function display(token: string) {
  return token === ' ' ? '␣' : token === '\n' ? '↵' : token
}

export function SamplingLab() {
  const { trace, manifest, generate, status } = useModel(demoPrompt)
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(20)
  const [seed, setSeed] = useState(7)
  const [generated, setGenerated] = useState('')
  const [busy, setBusy] = useState(false)
  const result = useMemo(
    () => (trace ? sampleFromLogits(trace.logits, { temperature, topK, seed }) : null),
    [trace, temperature, topK, seed],
  )
  const top = result ? topEntries(result.probabilities, 10) : []

  async function runGeneration() {
    if (!trace) return
    setBusy(true)
    try {
      const response = await generate(trace.tokenIds, 24, { temperature, topK, seed })
      setGenerated(response.result.text)
    } catch {
      setGenerated('生成失败：请确认推理线程可用。')
    } finally {
      setBusy(false)
    }
  }

  return (
    <LabLayout
      eyebrow="Lab 03 · Sampling"
      title="温度采样器"
      intro="模型输出的是一整个概率分布，不是一个确定的答案。温度改变分布的陡峭程度，top-k 改变候选集合，seed 决定这一次抽到谁。"
      asideTitle="结论"
      aside={
        <>
          <p>greedy 永远选概率最高的 token；采样则按概率抽签。</p>
          <ul>
            <li>温度越低，分布越尖，结果越保守。</li>
            <li>top-k 越小，低概率候选被直接排除。</li>
            <li>固定 seed 后，同一个分布可以得到可复现的抽样结果。</li>
          </ul>
        </>
      }
    >
      <div className="lab-toolbar">
        <label>
          温度 <b>{temperature.toFixed(2)}</b>
          <input type="range" min="0.1" max="2" step="0.05" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
        </label>
        <label>
          top-k <b>{topK}</b>
          <input type="range" min="1" max={manifest?.config.vocabSize ?? 65} value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
        </label>
        <label>
          seed <b>{seed}</b>
          <input type="range" min="1" max="99" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
        </label>
        <button
          type="button"
          className="button button--small"
          onClick={() => setSeed((value) => (value % 99) + 1)}
        >
          重新抽样
        </button>
      </div>

      {status === 'loading' && <p className="panel-note">正在加载模型…</p>}
      {status === 'error' && <p className="panel-note">推理引擎不可用。请通过 http 打开页面。</p>}
      {trace && result && manifest && (
        <>
          <div className="sample-readout">
            <div>
              <span>这一次抽到</span>
              <strong>{display(manifest.vocab[result.chosenIndex] ?? '')}</strong>
              <small>{(result.probabilities[result.chosenIndex] * 100).toFixed(1)}%</small>
            </div>
            <div>
              <span>greedy</span>
              <strong>{display(manifest.vocab[result.greedyIndex] ?? '')}</strong>
              <small>最高概率</small>
            </div>
          </div>
          <h3 style={{ fontSize: 14, marginTop: 20 }}>当前温度下的候选分布</h3>
          <BarList
            entries={top.map((entry) => ({
              label: `${display(manifest.vocab[entry.index] ?? '')} #${entry.index}`,
              value: entry.probability,
              color: 'var(--violet)',
            }))}
            format={(value) => `${(value * 100).toFixed(1)}%`}
          />
          <div className="generated-text">
            {generated || `ROMEO: what light…\n\n点击“生成 24 个字符”，用当前采样设置继续写下去。`}
          </div>
          <div style={{ display: 'flex', gap: 10, marginTop: 12, flexWrap: 'wrap' }}>
            <button type="button" className="button" onClick={runGeneration} disabled={busy}>
              {busy ? '生成中…' : '生成 24 个字符'}
            </button>
            <button type="button" className="button button--ghost" onClick={() => setGenerated('')}>
              清空
            </button>
          </div>
          <p className="panel-note" style={{ marginTop: 12 }}>
            生成使用字符级自回归：每次把新 token 追加到上下文，再重新计算下一步。因为模型很小，
            这些计算都在浏览器的 Web Worker 里完成。
          </p>
        </>
      )}
    </LabLayout>
  )
}
