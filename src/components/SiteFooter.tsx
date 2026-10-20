import { Link } from 'react-router-dom'
import { references } from '../content/references'

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="container">
        <div className="site-footer__grid">
          <div>
            <strong>Transformer 内部漫游</strong>
            <p style={{ marginTop: 10 }}>
              一个用真实权重、真实前向传播讲清楚 decoder-only Transformer 的中文交互站。
              纯静态构建，模型在浏览器本地推理。
            </p>
          </div>
          <div>
            <h4>学习</h4>
            <Link to="/walkthrough">9 章主线</Link>
            <Link to="/lab/attention">注意力显微镜</Link>
            <Link to="/lab/kv-cache">KV cache 账本</Link>
            <Link to="/lab/sampling">温度采样器</Link>
          </div>
          <div>
            <h4>模型</h4>
            <p>4 层 · 4 头 · 128 维</p>
            <p>字符级 Tiny Shakespeare</p>
            <p>809,856 参数</p>
            <Link to="/about">模型卡与复现</Link>
          </div>
          <div>
            <h4>参考</h4>
            {references.slice(0, 3).map((reference) => (
              <a key={reference.url} href={reference.url} target="_blank" rel="noreferrer">
                {reference.title}
              </a>
            ))}
          </div>
        </div>
        <div className="site-footer__bottom">
          <span>© 2026 Inside the Transformer · 代码 MIT</span>
          <span>
            体验灵感来自{' '}
            <a href="https://bbycroft.net/llm" target="_blank" rel="noreferrer">
              bbycroft.net/llm
            </a>{' '}
            与{' '}
            <a href="https://www.llm-visualized.com" target="_blank" rel="noreferrer">
              llm-visualized.com
            </a>
            ，未复制其代码或素材。
          </span>
        </div>
      </div>
    </footer>
  )
}
