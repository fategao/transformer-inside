import { useMemo, type CSSProperties } from 'react'
import { Link } from 'react-router-dom'
import { chapters, demoPrompt } from '../content/chapters'
import { useModel } from '../lib/useModel'
import { useReducedMotion } from '../lib/useReducedMotion'
import { buildSceneData } from '../lib/scene-data'
import { SceneCanvas } from '../scene/SceneCanvas'

const labs = [
  {
    id: 'attention',
    title: '注意力显微镜',
    tone: 'var(--blue)',
    text: '选层、选头、选查询位置，看 4×4 个头如何分配注意力，以及因果掩码切掉了什么。',
  },
  {
    id: 'kv-cache',
    title: 'KV cache 账本',
    tone: 'var(--amber)',
    text: '把上下文从 8 拉到 64，算清楚每步重算与缓存复用的计算量、显存和节省比例。',
  },
  {
    id: 'sampling',
    title: '温度采样器',
    tone: 'var(--violet)',
    text: '固定同一组 logits，调温度、top-k 和 seed，观察概率分布与生成结果如何改变。',
  },
]

export function HomePage() {
  const reduced = useReducedMotion()
  const { trace, manifest, status } = useModel(demoPrompt)
  const sceneData = useMemo(() => buildSceneData(trace, manifest), [trace, manifest])
  const parameterCount = manifest?.training.parameterCount ?? 809856

  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">交互式架构教材 · 01</span>
          <h1>
            Transformer
            <br />
            内部漫游
            <span>Inside the Transformer</span>
          </h1>
          <p className="lede">
            不是把架构图再画一遍，而是跟着一句话真正走完一次前向传播：
            从字符 ID、嵌入、QKV、多头注意力，到最后一个字的概率分布。
            所有数字都来自一个在浏览器里实时运行的小模型。
          </p>
          <div className="hero-actions">
            <Link className="button" to="/walkthrough">
              开始漫游 <span aria-hidden="true">→</span>
            </Link>
            <Link className="button button--ghost" to="/lab/attention">
              直接进实验室
            </Link>
          </div>
          <div className="stat-strip">
            <div>
              <strong>4 层</strong>
              <span>decoder blocks</span>
            </div>
            <div>
              <strong>4 头</strong>
              <span>multi-head attention</span>
            </div>
            <div>
              <strong>128 维</strong>
              <span>d_model</span>
            </div>
            <div>
              <strong>{parameterCount.toLocaleString('en-US')}</strong>
              <span>真实训练参数</span>
            </div>
          </div>
        </div>
        <div className="hero-stage">
          <SceneCanvas
            data={sceneData}
            chapter={0}
            autoCycle={!reduced}
            reducedMotion={reduced}
            className="hero-canvas"
            fallbackLabel="Transformer 前向传播 3D 场景"
          />
          <div className="hero-stage__badge">
            <span className="live-dot" />
            {status === 'ready' ? '浏览器本地实时推理' : status === 'error' ? '3D 预览 · 数值引擎待连接' : '正在加载 3.2 MB 权重'}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="section-head">
            <div>
              <span className="eyebrow">这不是视频</span>
              <h2>每一个数字，都能追到权重</h2>
            </div>
            <p>
              3D 场景只是入口。真正的教学内容在右侧面板里：矩阵、范数、注意力权重与概率，
              都由同一套权重在本地计算出来。
            </p>
          </div>
          <div className="card-grid">
            <article className="card">
              <span className="card__index">01 / 真实</span>
              <h3>真实权重，不是示意</h3>
              <p>
                页面加载 809,856 个训练好的参数。PyTorch 训练脚本、Tiny Shakespeare 语料与
                参考 logits 全部可复现，浏览器推理与 PyTorch 的误差小于 2e-4。
              </p>
            </article>
            <article className="card">
              <span className="card__index">02 / 动态</span>
              <h3>实时渲染，不是逐帧图片</h3>
              <p>
                token 沿轨道流动、注意力连线按权重脉动、层塔逐层点亮。滚动、点击和播放按钮
                都会实时改变相机与数据流。
              </p>
            </article>
            <article className="card">
              <span className="card__index">03 / 可操控</span>
              <h3>改一个参数，看结果怎么变</h3>
              <p>
                切换层与头、拉高温度、缩小 top-k、改变上下文长度，所有实验都直接作用于
                当前这一次真实前向传播。
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="section section--tight">
        <div className="container">
          <div className="section-head">
            <div>
              <span className="eyebrow">9 章主线</span>
              <h2>跟着数据走一遍</h2>
            </div>
            <p>每章只回答一个问题，配一个 3D 动作和一块真实数值面板。可以从任意一章直接进入。</p>
          </div>
          <div className="chapter-list">
            {chapters.map((chapter) => (
              <Link key={chapter.id} to={`/walkthrough#${chapter.id}`}>
                <b>{String(chapter.index + 1).padStart(2, '0')}</b>
                <span>
                  {chapter.title}
                  <small>{chapter.english}</small>
                </span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="section-head">
            <div>
              <span className="eyebrow">概念实验室</span>
              <h2>把“大概懂了”变成可检验</h2>
            </div>
            <p>三个实验分别回答注意力、推理成本和随机性三个最容易含糊过去的问题。</p>
          </div>
          <div className="lab-grid">
            {labs.map((lab) => (
              <Link
                key={lab.id}
                to={`/lab/${lab.id}`}
                className="lab-card"
                style={{ '--tone': lab.tone } as CSSProperties}
              >
                <span className="eyebrow">Experiment</span>
                <h3>{lab.title}</h3>
                <p>{lab.text}</p>
                <span className="lab-card__link">打开实验 →</span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="section section--tight">
        <div className="container">
          <div className="card-grid">
            <article className="card">
              <span className="card__index">模型卡</span>
              <h3>Tiny Shakespeare 4L</h3>
              <p>
                字符级、英文、上下文 64。训练 5,000 步，验证 loss{' '}
                {manifest ? manifest.training.valLoss.toFixed(3) : '约 1.61'}。它不是聊天模型，
                但结构完整、数值可核对。
              </p>
            </article>
            <article className="card">
              <span className="card__index">适合谁</span>
              <h3>学过 Transformer 基础的人</h3>
              <p>
                你知道 QKV 和 softmax 的公式，但想看清它们在真实张量里长什么样。
                不需要 GPU，也不需要 API Key。
              </p>
            </article>
            <article className="card">
              <span className="card__index">边界</span>
              <h3>只讲一次前向传播</h3>
              <p>
                第一版不覆盖训练、位置编码变体、MoE 与长上下文技巧。先把 decoder block
                这一条主线走扎实。
              </p>
            </article>
          </div>
        </div>
      </section>
    </>
  )
}
