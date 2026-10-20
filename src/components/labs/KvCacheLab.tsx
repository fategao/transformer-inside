import { useState } from 'react'
import { estimateKvCache, formatBytes, formatFlops } from '../../engine/kv-cache'
import { BarList } from '../BarList'
import { LabLayout } from '../LabLayout'

export function KvCacheLab() {
  const [context, setContext] = useState(32)
  const estimate = estimateKvCache(context, 4, 4, 32)
  const savings = (1 - estimate.cachedFlopsPerStep / estimate.fullFlopsPerStep) * 100

  return (
    <LabLayout
      eyebrow="Lab 02 · Inference"
      title="KV cache 账本"
      intro="生成第 N 个 token 时，如果每一步都重新计算整个上下文，计算量会随上下文长度快速增长。KV cache 把历史 K/V 留下来，只算新 token。"
      asideTitle="结论"
      aside={
        <>
          <p>KV cache 不改变模型输出，只改变计算方式：用显存换计算量。</p>
          <ul>
            <li>缓存大小与层数、上下文长度、d_model 成正比。</li>
            <li>上下文越长，节省的比例越明显。</li>
            <li>本项目 4 层 128 维、64 上下文，缓存只有约 256 KB。</li>
          </ul>
        </>
      }
    >
      <div className="lab-toolbar">
        <label>
          上下文长度 <b>{context}</b>
          <input type="range" min="8" max="64" step="1" value={context} onChange={(event) => setContext(Number(event.target.value))} />
        </label>
        <div className="metric">
          <span>每步节省</span>
          <strong>{savings.toFixed(1)}%</strong>
        </div>
        <div className="metric">
          <span>缓存占用</span>
          <strong>{formatBytes(estimate.cacheBytes)}</strong>
        </div>
      </div>

      <div className="compare-grid">
        <div className="compare-card">
          <h4>每步重算整个上下文</h4>
          <p>
            对 {context} 个位置重新做 QKV 投影、QKᵀ、Softmax 与 AV，再算 MLP。
          </p>
          <p style={{ marginTop: 8, color: 'var(--ink)' }}>
            <strong>{formatFlops(estimate.fullFlopsPerStep)}</strong> / step
          </p>
        </div>
        <div className="compare-card">
          <h4>使用 KV cache</h4>
          <p>
            只为新 token 算 Q，与缓存中的 K/V 做注意力；历史 token 的 K/V 直接复用。
          </p>
          <p style={{ marginTop: 8, color: 'var(--ink)' }}>
            <strong>{formatFlops(estimate.cachedFlopsPerStep)}</strong> / step
          </p>
        </div>
      </div>

      <h3 style={{ fontSize: 14, marginTop: 24 }}>计算量对比</h3>
      <BarList
        entries={[
          { label: '全量重算', value: estimate.fullFlopsPerStep, color: 'var(--red)' },
          { label: 'KV cache', value: estimate.cachedFlopsPerStep, color: 'var(--green)' },
        ]}
        format={(value) => formatFlops(value)}
      />

      <h3 style={{ fontSize: 14, marginTop: 24 }}>缓存里到底存了什么</h3>
      <p className="panel-note">
        每层、每个头都保存历史 token 的 K 与 V。总字节数 ≈ 2 × 层数 × 上下文长度 × d_model × 4 bytes。
        本项目：2 × 4 × {context} × 128 × 4 = {estimate.cacheBytes.toLocaleString('en-US')} bytes。
      </p>
      <div className="compare-grid" style={{ marginTop: 12 }}>
        <div className="compare-card">
          <h4>为什么能省</h4>
          <p>历史 token 的 K/V 不依赖未来，生成过程中不会改变，可以一次算好反复使用。</p>
        </div>
        <div className="compare-card">
          <h4>代价是什么</h4>
          <p>每个请求都要额外占用显存；上下文越长、并发越高，cache 越容易成为瓶颈。</p>
        </div>
      </div>
    </LabLayout>
  )
}
