import { useEffect, useRef, useState } from 'react'
import { SceneController, type SceneData } from './SceneController'

type Props = {
  data: SceneData
  chapter: number
  className?: string
  onSelect?: (selection: { kind: 'token'; index: number } | { kind: 'layer'; index: number } | null) => void
  fallbackLabel?: string
  autoCycle?: boolean
  reducedMotion?: boolean
}

export function SceneCanvas({
  data,
  chapter,
  className,
  onSelect,
  fallbackLabel,
  autoCycle = false,
  reducedMotion = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const controllerRef = useRef<SceneController | null>(null)
  const [unsupported, setUnsupported] = useState(false)

  useEffect(() => {
    if (!canvasRef.current) return
    try {
      const controller = new SceneController(canvasRef.current, { onSelect, reducedMotion })
      controllerRef.current = controller
      controller.setData(data)
      controller.setChapter(chapter)
      return () => {
        controller.dispose()
        controllerRef.current = null
      }
    } catch {
      setUnsupported(true)
    }
    return undefined
    // The controller owns the canvas for its whole lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    controllerRef.current?.setData(data)
  }, [data])

  useEffect(() => {
    controllerRef.current?.setChapter(chapter)
  }, [chapter])

  useEffect(() => {
    controllerRef.current?.setReducedMotion(reducedMotion)
  }, [reducedMotion])

  useEffect(() => {
    if (!autoCycle) return
    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      const phase = ((now - start) / 26000) % 1
      controllerRef.current?.setChapter(phase * 8)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [autoCycle])

  if (unsupported) {
    return (
      <div className={`scene-fallback ${className ?? ''}`} role="img" aria-label={fallbackLabel ?? 'Transformer 结构示意图'}>
        <div className="scene-fallback__rail">
          {data.tokens.map((token, index) => (
            <span key={`${token}-${index}`}>{token === ' ' ? '␣' : token}</span>
          ))}
        </div>
        <div className="scene-fallback__towers">
          {[1, 2, 3, 4].map((layer) => (
            <div key={layer} className={layer <= Math.round(chapter) + 1 ? 'is-live' : ''}>
              第 {layer} 层
            </div>
          ))}
        </div>
        <p>当前浏览器未启用 WebGL，已切换为 2D 结构视图；全部数值与实验仍然可用。</p>
      </div>
    )
  }

  return (
    <canvas
      ref={canvasRef}
      className={className}
      role="img"
      aria-label="可交互的 Transformer 3D 前向传播场景"
    />
  )
}
