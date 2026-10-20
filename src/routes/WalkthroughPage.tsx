import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { chapters, demoPrompt } from '../content/chapters'
import { useModel } from '../lib/useModel'
import { useReducedMotion } from '../lib/useReducedMotion'
import { buildSceneData } from '../lib/scene-data'
import { SceneCanvas } from '../scene/SceneCanvas'
import { ChapterPanel } from '../components/ChapterPanel'

type Mode = 'auto' | 'user' | 'paused'
type Selection = { kind: 'token'; index: number } | { kind: 'layer'; index: number } | null

function clamp(value: number, min = 0, max = 8) {
  return Math.max(min, Math.min(max, value))
}

export function WalkthroughPage() {
  const reduced = useReducedMotion()
  const { trace, manifest, status } = useModel(demoPrompt)
  const sceneData = useMemo(() => buildSceneData(trace, manifest), [trace, manifest])
  const initialHash = typeof window === 'undefined' ? '' : decodeURIComponent(window.location.hash.slice(1))
  const initialIndex = Math.max(0, chapters.findIndex((chapter) => chapter.id === initialHash))
  const [mode, setMode] = useState<Mode>(initialHash ? 'paused' : 'auto')
  const [chapterFloat, setChapterFloat] = useState(initialIndex)
  const [selection, setSelection] = useState<Selection>(null)
  const [viewMode, setViewMode] = useState<'follow' | 'free'>('follow')
  const [followToken, setFollowToken] = useState(0)
  const [sceneOnly, setSceneOnly] = useState(false)
  const modeRef = useRef(mode)
  const chapterFloatRef = useRef(chapterFloat)
  const autoScrollTopRef = useRef(-1)

  useEffect(() => {
    modeRef.current = mode
  }, [mode])

  useEffect(() => {
    chapterFloatRef.current = chapterFloat
  }, [chapterFloat])

  useEffect(() => {
    if (!initialHash) return
    const target = chapters.findIndex((chapter) => chapter.id === initialHash)
    if (target < 0) return
    requestAnimationFrame(() => window.scrollTo({ top: target * window.innerHeight, behavior: 'auto' }))
  }, [initialHash])

  const goTo = useCallback(
    (index: number) => {
      const target = Math.round(clamp(index))
      setMode('paused')
      setChapterFloat(target)
      setViewMode('follow')
      setFollowToken((value) => value + 1)
      window.scrollTo({ top: target * window.innerHeight, behavior: reduced ? 'auto' : 'smooth' })
    },
    [reduced],
  )

  useEffect(() => {
    if (reduced || mode !== 'auto') return
    const from = window.scrollY / window.innerHeight
    const to = Math.min(8, Math.ceil(from + 0.01))
    if (to <= from + 0.01) {
      setMode('paused')
      return
    }
    const start = performance.now()
    const duration = 9000 * Math.max(0.35, to - from)
    let frame = 0
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / duration)
      const eased = progress < 0.5 ? 2 * progress * progress : 1 - (-2 * progress + 2) ** 2 / 2
      const value = from + (to - from) * eased
      setChapterFloat(value)
      autoScrollTopRef.current = value * window.innerHeight
      window.scrollTo({ top: value * window.innerHeight, behavior: 'auto' })
      if (progress < 1) frame = requestAnimationFrame(tick)
      else setMode('paused')
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [mode, reduced])

  useEffect(() => {
    const onScroll = () => {
      if (modeRef.current === 'auto' && Math.abs(window.scrollY - autoScrollTopRef.current) < 4) return
      if (modeRef.current === 'auto') setMode('user')
      const value = clamp(window.scrollY / window.innerHeight)
      setChapterFloat(value)
      if (modeRef.current === 'paused') setMode('user')
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    const takeOver = (event: Event) => {
      const target = event.target as HTMLElement | null
      if (target?.closest('button, a, input, select, .chapter-card, .playback-bar, .chapter-rail')) return
      if (modeRef.current === 'auto') setMode('user')
    }
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target?.matches('input, select, textarea')) return
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        goTo(Math.round(chapterFloatRef.current) + 1)
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        goTo(Math.round(chapterFloatRef.current) - 1)
      } else if (event.code === 'Space') {
        event.preventDefault()
        setMode((value) => (value === 'auto' ? 'paused' : 'auto'))
      } else if (event.key === 'Escape') {
        setSelection(null)
      }
    }
    window.addEventListener('wheel', takeOver, { passive: true })
    window.addEventListener('touchstart', takeOver, { passive: true })
    window.addEventListener('pointerdown', takeOver)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('wheel', takeOver)
      window.removeEventListener('touchstart', takeOver)
      window.removeEventListener('pointerdown', takeOver)
      window.removeEventListener('keydown', onKey)
    }
  }, [goTo])

  const activeIndex = Math.round(chapterFloat)
  const selectedToken = selection?.kind === 'token' ? selection.index : trace ? trace.seq - 1 : 0

  return (
    <div className={`walkthrough ${sceneOnly ? 'is-scene-only' : ''}`}>
      <SceneCanvas
        data={sceneData}
        chapter={chapterFloat}
        reducedMotion={reduced}
        className="walkthrough-canvas"
        onSelect={setSelection}
        onUserInteract={() => setMode('user')}
        onViewModeChange={setViewMode}
        followToken={followToken}
        fallbackLabel="Transformer 前向传播 3D 场景"
      />
      <div className="scene-toolbar">
        <span className="scene-toolbar__hint">拖动旋转 · Ctrl + 滚轮缩放 · 竖柱 = 同一 token 跨层</span>
        <button type="button" onClick={() => goTo(6)}>
          总览 4 层
        </button>
        {viewMode === 'free' ? (
          <button type="button" onClick={() => setFollowToken((value) => value + 1)}>
            回到跟随镜头
          </button>
        ) : (
          <button type="button" onClick={() => goTo(activeIndex)}>
            跟随当前章
          </button>
        )}
      </div>
      <div className="mobile-scene-hint">拖动旋转 · 双指缩放 · 竖柱 = 跨层 token</div>
      <button
        type="button"
        className="scene-only-toggle"
        onClick={() => {
          if (!sceneOnly) goTo(6)
          setSceneOnly((value) => !value)
        }}
      >
        {sceneOnly ? '显示讲解' : '全屏 3D'}
      </button>
      {viewMode === 'free' && (
        <button
          type="button"
          className="mobile-follow-button"
          onClick={() => setFollowToken((value) => value + 1)}
        >
          回到跟随镜头
        </button>
      )}

      {chapters.map((chapter) => (
        <section className="walkthrough-section" id={chapter.id} key={chapter.id} aria-labelledby={`${chapter.id}-title`}>
          <article className="chapter-card">
            <div className="chapter-card__top">
              <span>
                CH {String(chapter.index + 1).padStart(2, '0')} / 09
              </span>
              <span>{chapter.short}</span>
            </div>
            <h2 id={`${chapter.id}-title`}>
              {chapter.title}
              <small>{chapter.english}</small>
            </h2>
            <p className="chapter-question">{chapter.question}</p>
            <p className="chapter-conclusion">{chapter.conclusion}</p>
            <div className="chapter-panel">
              {status === 'error' ? (
                <p className="panel-note">推理引擎加载失败：{status === 'error' ? '请检查 model 资源或改用支持 Web Worker 的浏览器。' : ''}</p>
              ) : (
                <ChapterPanel kind={chapter.panel} trace={trace} manifest={manifest} selectedToken={selectedToken} />
              )}
            </div>
            <div className="chapter-hint">
              <b>动手</b>
              <span>{chapter.hint}</span>
            </div>
          </article>
        </section>
      ))}

      <nav className="chapter-rail" aria-label="章节导航">
        {chapters.map((chapter, index) => (
          <button
            key={chapter.id}
            type="button"
            className={index === activeIndex ? 'is-active' : ''}
            onClick={() => goTo(index)}
            aria-label={`跳到第 ${index + 1} 章：${chapter.title}`}
          >
            <small>{chapter.short}</small>
            <span />
          </button>
        ))}
      </nav>

      <div className="playback-bar" role="group" aria-label="播放与章节控制">
        <div className="playback-controls">
          <button type="button" className="icon-button" onClick={() => goTo(activeIndex - 1)} aria-label="上一章">
            ←
          </button>
          <button
            type="button"
            className="icon-button icon-button--primary"
            onClick={() => setMode((value) => (value === 'auto' ? 'paused' : 'auto'))}
            aria-label={mode === 'auto' ? '暂停自动播放' : '继续自动播放'}
          >
            {mode === 'auto' ? '❚❚' : '▶'}
          </button>
          <button type="button" className="icon-button" onClick={() => goTo(activeIndex + 1)} aria-label="下一章">
            →
          </button>
        </div>
        <div className="playback-progress" aria-hidden="true">
          <span style={{ width: `${(chapterFloat / 8) * 100}%` }} />
        </div>
        <span className="playback-label">
          {mode === 'auto' ? '自动播放' : `${String(activeIndex + 1).padStart(2, '0')} / 09`}
        </span>
      </div>

      {selection?.kind === 'token' && trace && (
        <aside className="inspector">
          <strong>Token #{selection.index}</strong>
          <span>
            {trace.tokens[selection.index] === ' ' ? '空格' : trace.tokens[selection.index]} · ID{' '}
            {trace.tokenIds[selection.index]} · 嵌入范数{' '}
            {Math.sqrt(
              Array.from(
                trace.embeddings.subarray(selection.index * trace.dim, (selection.index + 1) * trace.dim),
              ).reduce((acc, value) => acc + value * value, 0),
            ).toFixed(3)}
          </span>
        </aside>
      )}
      {selection?.kind === 'layer' && trace && (
        <aside className="inspector">
          <strong>第 {selection.index + 1} 层</strong>
          <span>
            输出范数{' '}
            {Math.sqrt(
              Array.from(
                trace.layers[selection.index].residualOut.subarray(
                  (trace.seq - 1) * trace.dim,
                  trace.seq * trace.dim,
                ),
              ).reduce((acc, value) => acc + value * value, 0),
            ).toFixed(3)}
            {' · '}
            4 个头 × 128 维
          </span>
        </aside>
      )}
    </div>
  )
}
