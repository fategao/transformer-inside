import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  ForwardTrace,
  GenerateResult,
  ModelManifest,
  SampleResult,
  SamplingOptions,
  WorkerRequest,
  WorkerResponse,
} from '../engine/types'

type Pending = {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

type WorkerPayload =
  | { type: 'init'; baseUrl: string }
  | { type: 'forward'; tokenIds: number[] }
  | { type: 'sample'; tokenIds: number[]; options: SamplingOptions }
  | { type: 'generate'; tokenIds: number[]; maxNewTokens: number; options: SamplingOptions }

type ModelState = {
  status: 'loading' | 'ready' | 'error'
  manifest: ModelManifest | null
  trace: ForwardTrace | null
  error: string | null
}

export function useModel(prompt: string, options: { enabled?: boolean } = {}) {
  const enabled = options.enabled ?? true
  const [state, setState] = useState<ModelState>({ status: 'loading', manifest: null, trace: null, error: null })
  const workerRef = useRef<Worker | null>(null)
  const pendingRef = useRef(new Map<number, Pending>())
  const requestId = useRef(0)
  const request = useCallback(<T,>(message: WorkerPayload): Promise<T> => {
    const worker = workerRef.current
    if (!worker) return Promise.reject(new Error('推理线程尚未就绪'))
    const id = ++requestId.current
    return new Promise<T>((resolve, reject) => {
      pendingRef.current.set(id, { resolve: resolve as (value: unknown) => void, reject })
      worker.postMessage({ ...message, id } as WorkerRequest)
    })
  }, [])

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    let worker: Worker
    try {
      worker = new Worker(new URL('../engine/worker.ts', import.meta.url), { type: 'module' })
    } catch (error) {
      setState({ status: 'error', manifest: null, trace: null, error: String(error) })
      return
    }
    workerRef.current = worker
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data
      const pending = pendingRef.current.get(response.id)
      pendingRef.current.delete(response.id)
      if (response.ok === false) {
        pending?.reject(new Error(response.error))
        if (response.id === 1) {
          setState({ status: 'error', manifest: null, trace: null, error: response.error })
        }
        return
      }
      if (response.type === 'init') {
        if (cancelled) return
        setState({ status: 'ready', manifest: response.manifest, trace: null, error: null })
        pending?.resolve(response.manifest)
        return
      }
      pending?.resolve(response)
    }
    worker.onerror = (event) => {
      setState({ status: 'error', manifest: null, trace: null, error: event.message || '推理线程加载失败' })
    }
    const pendingRequests = pendingRef.current
    const baseUrl = `${import.meta.env.BASE_URL.replace(/\/?$/, '/')}model/`
    worker.postMessage({ id: 1, type: 'init', baseUrl } satisfies WorkerRequest)
    return () => {
      cancelled = true
      worker.terminate()
      workerRef.current = null
      pendingRequests.forEach((pending) => pending.reject(new Error('推理线程已关闭')))
      pendingRequests.clear()
    }
  }, [enabled])

  useEffect(() => {
    if (!enabled || state.status !== 'ready' || !state.manifest) return
    let cancelled = false
    const tokenizer = state.manifest.vocab
    const stoi = new Map(tokenizer.map((char, index) => [char, index]))
    const tokenIds = Array.from(prompt)
      .filter((char) => stoi.has(char))
      .slice(-state.manifest!.config.contextLength)
      .map((char) => stoi.get(char)!)
    if (!tokenIds.length) return
    request<{ trace: ForwardTrace }>({ type: 'forward', tokenIds })
      .then((response) => {
        if (!cancelled) setState((previous) => ({ ...previous, trace: response.trace, error: null }))
      })
      .catch((error: Error) => {
        if (!cancelled) setState((previous) => ({ ...previous, status: 'error', error: error.message }))
      })
    return () => {
      cancelled = true
    }
  }, [enabled, prompt, request, state.manifest, state.status])

  const sample = useCallback(
    (tokenIds: number[], options: SamplingOptions) => request<{ result: SampleResult }>({ type: 'sample', tokenIds, options }),
    [request],
  )

  const generate = useCallback(
    (tokenIds: number[], maxNewTokens: number, options: SamplingOptions) =>
      request<{ result: GenerateResult }>({ type: 'generate', tokenIds, maxNewTokens, options }),
    [request],
  )

  return { ...state, sample, generate }
}
