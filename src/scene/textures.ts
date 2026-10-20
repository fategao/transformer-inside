import * as THREE from 'three'

export const palette = {
  paper: '#f5f2ea',
  ink: '#1f2a37',
  muted: '#6b7280',
  blue: '#2f6fed',
  red: '#e05263',
  green: '#2f9e7a',
  amber: '#d98a2b',
  violet: '#7c5ce5',
  cyan: '#1c9bb5',
}

export const headColors = [palette.blue, palette.red, palette.green, palette.violet]

export function tokenColor(char: string): string {
  if (char === ' ') return '#c8c2b4'
  if ('.,:;!?\'"'.includes(char)) return palette.amber
  if ('aeiouAEIOU'.includes(char)) return palette.blue
  return palette.cyan
}

export function createLabelTexture(text: string, options: { color?: string; background?: string; fontSize?: number } = {}) {
  const size = 256
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const context = canvas.getContext('2d')
  if (!context) throw new Error('2D canvas unavailable')
  if (options.background) {
    context.fillStyle = options.background
    context.roundRect(8, 8, size - 16, size - 16, 28)
    context.fill()
  }
  context.fillStyle = options.color ?? palette.ink
  context.font = `600 ${options.fontSize ?? 92}px "Georgia", "Songti SC", serif`
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText(text, size / 2, size / 2 + 4)
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 4
  return texture
}

export function createBarLabel(text: string, color = palette.ink) {
  const canvas = document.createElement('canvas')
  canvas.width = 256
  canvas.height = 128
  const context = canvas.getContext('2d')
  if (!context) throw new Error('2D canvas unavailable')
  context.clearRect(0, 0, canvas.width, canvas.height)
  context.fillStyle = color
  context.font = '600 54px "Inter", "PingFang SC", sans-serif'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText(text, canvas.width / 2, canvas.height / 2)
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}
