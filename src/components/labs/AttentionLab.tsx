import { useMemo, useState } from 'react'
import { demoPrompt } from '../../content/chapters'
import { useModel } from '../../lib/useModel'
import { topEntries } from '../../engine/sampling'
import { BarList } from '../BarList'
import { Heatmap } from '../Heatmap'
import { LabLayout } from '../LabLayout'

function display(token: string) {
  return token === ' ' ? '␣' : token === '\n' ? '↵' : token
}

export function AttentionLab() {
  const [draft, setDraft] = useState(demoPrompt)
  const [prompt, setPrompt] = useState(demoPrompt)
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [query, setQuery] = useState(16)
  const { trace, status } = useModel(prompt)
  const seq = trace?.seq ?? 0
  const actualQuery = Math.min(query, Math.max(0, seq - 1))

  const values = useMemo(() => {
    if (!trace) return []
    const attention = trace.layers[layer].attention
    return Array.from({ length: seq }, (_, row) =>
      Array.from({ length: seq }, (_, column) => {
        return attention[head * seq * seq + row * seq + column]
      }),
    )
  }, [trace, layer, head, seq])

  const selectedRow = values[actualQuery] ?? []
  const top = topEntries(selectedRow, 6)

  return (
    <LabLayout
      eyebrow="Lab 01 · Attention"
      title="注意力显微镜"
      intro="同一句话，换一个层、换一个头，模型关注的位置会完全不同。这里显示的是一个真实训练过的 4 层 4 头模型在字符级输入上的注意力权重。"
      asideTitle="怎么看"
      aside={
        <>
          <p>每一行是一个查询位置，每一列是它能读取的键位置。每一行的权重总和为 1。</p>
          <ul>
            <li>蓝色越深，权重越大。</li>
            <li>右上角空白由因果掩码造成：不能看未来。</li>
            <li>
              第 1 层的头通常更接近局部模式，深层会更分散；这不是规律，只是这个模型在这个输入上的表现。
            </li>
          </ul>
        </>
      }
    >
      <form
        className="lab-toolbar"
        onSubmit={(event) => {
          event.preventDefault()
          setPrompt(draft.slice(0, 32))
        }}
      >
        <label>
          输入一句英文（最多 32 字符，只保留字符级词表内的字符）
          <input type="text" value={draft} maxLength={32} onChange={(event) => setDraft(event.target.value)} />
        </label>
        <button className="button button--small" type="submit">
          重新计算
        </button>
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
        <label>
          查询位置 <b>{actualQuery}</b>
          <input
            type="range"
            min={0}
            max={Math.max(0, seq - 1)}
            value={actualQuery}
            onChange={(event) => setQuery(Number(event.target.value))}
          />
        </label>
      </form>

      {status === 'loading' && <p className="panel-note">正在加载模型并计算…</p>}
      {status === 'error' && <p className="panel-note">推理引擎不可用。请通过 http 打开页面，并确认 model 资源可访问。</p>}
      {trace && (
        <>
          <p className="panel-note">
            当前输入：<span className="mono">{trace.tokens.map(display).join('')}</span> · {seq} 个 token
          </p>
          <Heatmap values={values} label={`第 ${layer + 1} 层第 ${head + 1} 个头的注意力`} />
          <div className="legend-row" style={{ marginTop: 8 }}>
            <span>查询行 →</span>
            <span>键列 ↑</span>
            <span className="legend-swatch" /> 权重
          </div>
          <h3 style={{ fontSize: 14, marginBottom: 4 }}>查询位置 {actualQuery} 在关注谁</h3>
          <BarList
            entries={top.map((entry) => ({
              label: `${entry.index} ${display(trace.tokens[entry.index])}`,
              value: entry.probability,
              color: 'var(--blue)',
            }))}
            format={(value) => `${(value * 100).toFixed(1)}%`}
          />
        </>
      )}
    </LabLayout>
  )
}
