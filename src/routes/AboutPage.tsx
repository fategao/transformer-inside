import { demoPrompt } from '../content/chapters'
import { references } from '../content/references'
import { useModel } from '../lib/useModel'

export function AboutPage() {
  const { manifest, status } = useModel(demoPrompt)
  return (
    <>
      <section className="page-hero">
        <div className="container">
          <span className="eyebrow">Model card · Reproducibility</span>
          <h1>这个模型是怎么来的</h1>
          <p>
            站点里的每一个注意力权重、范数和概率，都来自下面这个小模型的真实前向传播。
            它不是 GPT-2 的替代品，而是一个结构完整、足够小、能在浏览器里逐层看清的教学模型。
          </p>
        </div>
      </section>

      <div className="container about-grid">
        <article className="about-card">
          <span className="eyebrow">Model card</span>
          <h2>Tiny Shakespeare 4L</h2>
          <p>
            4 个 decoder block、4 个注意力头、d_model 128、FFN 512、上下文 64、字符级词表 65。
            采用 pre-LayerNorm、GELU、因果掩码与权重共享的输出头，结构对应 GPT-2 的 decoder 部分。
            位置信息使用可学习的 position embedding（GPT-2 风格），不是原论文的固定正弦编码。
          </p>
          <h3>训练</h3>
          <p>
            {manifest
              ? `在 ${manifest.training.device} 上训练 ${manifest.training.steps.toLocaleString()} 步，batch ${manifest.training.batchSize}，block ${manifest.training.blockSize}，seed ${manifest.training.seed}。训练 loss ${manifest.training.trainLoss.toFixed(3)}，验证 loss ${manifest.training.valLoss.toFixed(3)}。`
              : '正在读取训练指标…'}
          </p>
          <h3>已知局限</h3>
          <ul>
            <li>字符级、英文、上下文只有 64；不能当作通用聊天模型。</li>
            <li>训练语料是莎士比亚文本，生成风格与语料强相关。</li>
            <li>为了让矩阵能逐个看清，层数与宽度都远小于生产模型。</li>
            <li>它用于解释结构，不用于评估模型能力。</li>
          </ul>
          <h3>数值一致性</h3>
          <p>
            浏览器 TypeScript 推理与 PyTorch 参考实现使用同一套 float32 权重；固定 prompt 的 logits
            与注意力最大绝对误差经测试小于 2e-4。CI 会在不训练的情况下重新跑这项检查。
          </p>
        </article>

        <article className="about-card">
          <span className="eyebrow">Training sample</span>
          <h2>它写出来是什么样</h2>
          <p>下面是用 temperature 0.8、从 “ROMEO: ” 继续生成 240 个字符的原始输出，没有做人工润色。</p>
          <pre className="generated-text" style={{ maxHeight: 280, overflow: 'auto' }}>
            {manifest?.sample ?? (status === 'loading' ? '正在读取样本…' : '样本不可用')}
          </pre>
          <h3>复现步骤</h3>
          <ol>
            <li>
              安装 Python 3.9+、PyTorch 2.x 与 numpy，然后执行 <code>python training/train.py</code>。
            </li>
            <li>脚本会下载 Tiny Shakespeare、训练 5,000 步，并导出 weights.bin、model.json 与 reference.json。</li>
            <li>执行 <code>npm install &amp;&amp; npm test</code>，确认浏览器推理与参考结果一致。</li>
            <li>执行 <code>npm run dev</code> 或 <code>npm run build</code> 查看页面。</li>
          </ol>
          <h3>数据与许可</h3>
          <p>
            代码以 MIT 许可发布。Tiny Shakespeare 是公共领域文本；页面未包含任何第三方模型权重或
            参考站的代码、图片与模型资产。
          </p>
        </article>

        <article className="about-card" style={{ gridColumn: '1 / -1' }}>
          <span className="eyebrow">References</span>
          <h2>统一参考资料</h2>
          <p>正文不插入脚注，所有结构与训练相关的来源集中在这里。</p>
          <div className="reference-list" style={{ marginTop: 16 }}>
            {references.map((reference) => (
              <a key={reference.url} href={reference.url} target="_blank" rel="noreferrer">
                <strong>{reference.title}</strong>
                <span>
                  {reference.authors} · {reference.note}
                </span>
              </a>
            ))}
          </div>
        </article>
      </div>
    </>
  )
}
