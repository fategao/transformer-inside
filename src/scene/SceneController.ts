import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { createBarLabel, createLabelTexture, headColors, palette, tokenColor } from './textures'

export type SceneData = {
  tokens: string[]
  attention: number[][][][]
  probabilities: number[]
  vocab: string[]
  embedding?: number[][]
}

type Selection = { kind: 'token'; index: number } | { kind: 'layer'; index: number } | null

const cameraKeyframes: Array<{ position: [number, number, number]; target: [number, number, number] }> = [
  { position: [15, 11, 30], target: [3, 4.6, 0] },
  { position: [-11, 11, 32], target: [-3, 5.2, 0] },
  { position: [13, 11, 31], target: [4, 5, 0] },
  { position: [-12, 13, 32], target: [-3, 5.8, 0] },
  { position: [15, 11, 30], target: [5, 5.4, 0] },
  { position: [-15, 12, 30], target: [-4, 6, 0] },
  { position: [4, 20, 44], target: [2, 6, 0] },
  { position: [8, 11, 25], target: [0, 7.2, 0] },
  { position: [6, 10, 23], target: [0, 6.8, 0] },
]

function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value))
}

function near(value: number, target: number, range = 1.2) {
  return clamp(1 - Math.abs(value - target) / range)
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh
    if (mesh.geometry) mesh.geometry.dispose()
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined
    if (Array.isArray(material)) material.forEach((item) => item.dispose())
    else material?.dispose()
  })
}

export class SceneController {
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(40, 1, 0.1, 160)
  private renderer: THREE.WebGLRenderer
  private raycaster = new THREE.Raycaster()
  private pointer = new THREE.Vector2()
  private startedAt = performance.now()
  private tokenGroup = new THREE.Group()
  private towerGroup = new THREE.Group()
  private qkvGroup = new THREE.Group()
  private attentionGroup = new THREE.Group()
  private embeddingGroup = new THREE.Group()
  private residualGroup = new THREE.Group()
  private logitsGroup = new THREE.Group()
  private sampleGroup = new THREE.Group()
  private particleGroup: THREE.Points | null = null
  private tokenMeshes: Array<THREE.Mesh> = []
  private towerMeshes: Array<THREE.Mesh> = []
  private mlpPlates: Array<THREE.Mesh[]> = []
  private data: SceneData = { tokens: [], attention: [], probabilities: [], vocab: [] }
  private chapterTarget = 0
  private chapter = 0
  private attentionLayer = 0
  private cameraPosition = new THREE.Vector3(...cameraKeyframes[0].position)
  private cameraTarget = new THREE.Vector3(...cameraKeyframes[0].target)
  private frame = 0
  private selected: Selection = null
  private selectionCallback?: (selection: Selection) => void
  private resizeObserver: ResizeObserver
  private disposed = false
  private reducedMotion = false
  private tokenX: number[] = []
  private worldWidth = 20
  private residualPulse: THREE.Mesh | null = null
  private residualCurve: THREE.CatmullRomCurve3 | null = null
  private flowPulses: THREE.Mesh[] = []
  private sampleOrb: THREE.Mesh | null = null
  private samplePath: THREE.QuadraticBezierCurve3 | null = null
  private canvas: HTMLCanvasElement

  constructor(
    canvas: HTMLCanvasElement,
    options: {
      onSelect?: (selection: Selection) => void
      onTokenHover?: (index: number | null) => void
      reducedMotion?: boolean
    } = {},
  ) {
    this.canvas = canvas
    this.selectionCallback = options.onSelect
    this.reducedMotion = options.reducedMotion ?? false
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
    const mobile = window.matchMedia('(max-width: 720px)').matches
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 2))
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFShadowMap
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.02

    this.scene.background = new THREE.Color(palette.paper)
    this.scene.fog = new THREE.Fog(palette.paper, 44, 92)
    this.scene.add(
      this.tokenGroup,
      this.towerGroup,
      this.qkvGroup,
      this.attentionGroup,
      this.embeddingGroup,
      this.residualGroup,
      this.logitsGroup,
      this.sampleGroup,
    )

    this.buildLights()
    this.buildGround()
    this.buildTowers()
    this.buildParticles()
    this.setChapter(0)

    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.resizeObserver.observe(canvas)
    this.resize()
    canvas.addEventListener('pointerdown', this.handlePointer)
    canvas.addEventListener('pointermove', this.handleHover)
    this.setChapter(0)
    this.renderer.setAnimationLoop(this.tick)
  }

  private buildLights() {
    const hemisphere = new THREE.HemisphereLight(0xfffdf8, 0xb9b2a2, 1.55)
    this.scene.add(hemisphere)
    const key = new THREE.DirectionalLight(0xffffff, 2.4)
    key.position.set(14, 24, 18)
    key.castShadow = true
    key.shadow.mapSize.set(2048, 2048)
    key.shadow.camera.left = -32
    key.shadow.camera.right = 32
    key.shadow.camera.top = 32
    key.shadow.camera.bottom = -32
    key.shadow.camera.near = 1
    key.shadow.camera.far = 70
    key.shadow.bias = -0.0004
    this.scene.add(key)
    const fill = new THREE.DirectionalLight(0xbcd4ff, 0.55)
    fill.position.set(-18, 8, -14)
    this.scene.add(fill)
  }

  private buildGround() {
    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(64, 96),
      new THREE.MeshStandardMaterial({ color: 0xebe5d8, roughness: 0.96, metalness: 0 }),
    )
    ground.rotation.x = -Math.PI / 2
    ground.position.y = -0.62
    ground.receiveShadow = true
    this.scene.add(ground)
    const grid = new THREE.GridHelper(72, 36, 0xd9d2c2, 0xe4ded0)
    grid.position.y = -0.6
    const material = grid.material as THREE.LineBasicMaterial
    material.transparent = true
    material.opacity = 0.38
    this.scene.add(grid)
  }

  private buildTowers() {
    for (let layer = 0; layer < 4; layer += 1) {
      const group = new THREE.Group()
      const y = 2.9 + layer * 3
      const geometry = new RoundedBoxGeometry(1, 1.38, 2.35, 5, 0.1)
      const material = new THREE.MeshPhysicalMaterial({
        color: [0x91b5d8, 0x9db9e8, 0xa9c4d9, 0xb8cfe0][layer],
        transparent: true,
        opacity: 0.2,
        roughness: 0.14,
        metalness: 0.04,
        clearcoat: 1,
        clearcoatRoughness: 0.16,
        depthWrite: false,
        side: THREE.DoubleSide,
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set(0, y, 0)
      mesh.scale.x = 20
      mesh.userData = { kind: 'layer', index: layer }
      mesh.castShadow = false
      mesh.receiveShadow = false
      group.add(mesh)
      this.towerMeshes.push(mesh)

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(new RoundedBoxGeometry(1, 1.38, 2.35, 3, 0.08)),
        new THREE.LineBasicMaterial({ color: 0x6e8ca9, transparent: true, opacity: 0.34 }),
      )
      edges.position.copy(mesh.position)
      edges.scale.x = 20
      group.add(edges)

      const plates: THREE.Mesh[] = []
      for (let plate = 0; plate < 3; plate += 1) {
        const plateMesh = new THREE.Mesh(
          new RoundedBoxGeometry(17.4, 0.08, 1.7, 3, 0.04),
          new THREE.MeshStandardMaterial({
            color: 0xf0a35e,
            transparent: true,
            opacity: 0.26,
            roughness: 0.5,
          }),
        )
        plateMesh.position.set(0, y - 0.36 + plate * 0.36, 0)
        plateMesh.scale.set(0.82, 1, 0.82)
        plates.push(plateMesh)
        group.add(plateMesh)
      }
      this.mlpPlates.push(plates)

      const label = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: createBarLabel(`L${layer + 1}`, '#41536a'),
          transparent: true,
          depthWrite: false,
        }),
      )
      label.position.set(-11.6, y, 1.35)
      label.scale.set(1.5, 0.75, 1)
      group.add(label)
      this.towerGroup.add(group)
    }
    const spine = new THREE.Mesh(
      new THREE.CylinderGeometry(0.06, 0.06, 11.6, 12),
      new THREE.MeshStandardMaterial({ color: 0x7890a7, transparent: true, opacity: 0.32, roughness: 0.5 }),
    )
    spine.position.set(0, 7.4, -1.35)
    this.scene.add(spine)
    const pulseColors = [palette.blue, palette.amber, palette.violet]
    for (let index = 0; index < 3; index += 1) {
      const pulse = new THREE.Mesh(
        new THREE.SphereGeometry(0.11, 18, 18),
        new THREE.MeshStandardMaterial({
          color: pulseColors[index],
          emissive: pulseColors[index],
          emissiveIntensity: 0.6,
          transparent: true,
          opacity: 0.85,
        }),
      )
      pulse.userData.offset = index / 3
      this.scene.add(pulse)
      this.flowPulses.push(pulse)
    }
  }

  private buildParticles() {
    const count = 110
    const positions = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    const color = new THREE.Color(palette.blue)
    for (let i = 0; i < count; i += 1) {
      positions[i * 3] = (Math.random() - 0.5) * 24
      positions[i * 3 + 1] = Math.random() * 13
      positions[i * 3 + 2] = (Math.random() - 0.5) * 6
      colors[i * 3] = color.r
      colors[i * 3 + 1] = color.g
      colors[i * 3 + 2] = color.b
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    const material = new THREE.PointsMaterial({
      size: 0.075,
      vertexColors: true,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
    })
    this.particleGroup = new THREE.Points(geometry, material)
    this.scene.add(this.particleGroup)
  }

  setData(data: SceneData) {
    this.data = data
    this.rebuildTokens()
    this.rebuildQkv()
    this.rebuildEmbeddings()
    this.rebuildAttention()
    this.rebuildLogits()
    this.rebuildResidual()
  }

  private rebuildTokens() {
    disposeObject(this.tokenGroup)
    this.tokenGroup.clear()
    this.tokenMeshes = []
    this.tokenX = []
    const tokens = this.data.tokens.length ? this.data.tokens : [' ', ' ', ' ', ' ', ' ', ' ', ' ']
    this.worldWidth = Math.max(20, (tokens.length - 1) * 1.55 + 3)
    const offset = (tokens.length - 1) / 2
    tokens.forEach((token, index) => {
      const x = (index - offset) * 1.55
      this.tokenX.push(x)
      const color = new THREE.Color(tokenColor(token))
      const mesh = new THREE.Mesh(
        new RoundedBoxGeometry(1.12, 0.94, 0.58, 5, 0.08),
        new THREE.MeshStandardMaterial({
          color,
          roughness: 0.34,
          metalness: 0.02,
          emissive: color,
          emissiveIntensity: 0.04,
          transparent: true,
          opacity: 0.96,
        }),
      )
      mesh.position.set(x, 0.56, 0)
      mesh.castShadow = true
      mesh.receiveShadow = true
      mesh.userData = { kind: 'token', index }
      this.tokenMeshes.push(mesh)
      this.tokenGroup.add(mesh)

      const label = new THREE.Mesh(
        new THREE.PlaneGeometry(0.9, 0.9),
        new THREE.MeshBasicMaterial({
          map: createLabelTexture(token === ' ' ? '␣' : token, { color: '#ffffff', fontSize: 104 }),
          transparent: true,
          depthWrite: false,
        }),
      )
      label.position.set(x, 0.57, 0.305)
      this.tokenGroup.add(label)
    })
    this.towerMeshes.forEach((tower) => {
      tower.scale.x = this.worldWidth + 4
    })
    this.towerGroup.children.forEach((group, index) => {
      const edges = group.children.find((child) => (child as THREE.LineSegments).isLineSegments) as THREE.LineSegments | undefined
      if (edges) edges.scale.x = this.worldWidth + 4
      const label = group.children.find((child) => (child as THREE.Sprite).isSprite) as THREE.Sprite | undefined
      if (label) label.position.set(-((this.worldWidth + 4) / 2) + 1.6, 2.9 + index * 3, 1.35)
    })
  }

  private rebuildQkv() {
    disposeObject(this.qkvGroup)
    this.qkvGroup.clear()
    const colors = [palette.red, palette.green, palette.amber]
    this.tokenX.forEach((x) => {
      colors.forEach((color, role) => {
        const node = new THREE.Mesh(
          new THREE.SphereGeometry(0.085, 16, 16),
          new THREE.MeshStandardMaterial({ color, roughness: 0.3, emissive: color, emissiveIntensity: 0.12 }),
        )
        node.position.set(x, 1.9, (role - 1) * 0.36)
        this.qkvGroup.add(node)
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x, 1.05, 0),
          new THREE.Vector3(x, 1.82, (role - 1) * 0.36),
        ])
        const line = new THREE.Line(
          geometry,
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.32 }),
        )
        this.qkvGroup.add(line)
      })
    })
  }

  private rebuildEmbeddings() {
    disposeObject(this.embeddingGroup)
    this.embeddingGroup.clear()
    const vectors = this.data.embedding
    if (!vectors?.length) return
    vectors.forEach((vector, tokenIndex) => {
      const x = this.tokenX[tokenIndex]
      vector.slice(0, 7).forEach((rawValue, dim) => {
        const value = Math.max(-1, Math.min(1, rawValue))
        const height = 0.08 + Math.abs(value) * 0.9
        const bar = new THREE.Mesh(
          new THREE.BoxGeometry(0.065, 1, 0.065),
          new THREE.MeshStandardMaterial({
            color: value >= 0 ? palette.blue : palette.red,
            emissive: value >= 0 ? palette.blue : palette.red,
            emissiveIntensity: 0.18,
            transparent: true,
            opacity: 0.8,
            roughness: 0.3,
          }),
        )
        bar.position.set(x + (dim - 3) * 0.11, 1.12 + height / 2, 0.34)
        bar.scale.y = height
        this.embeddingGroup.add(bar)
      })
    })
  }

  private rebuildAttention() {
    disposeObject(this.attentionGroup)
    this.attentionGroup.clear()
    const layer = this.data.attention[this.attentionLayer]
    if (!layer || !this.tokenX.length) return
    const positions: number[] = []
    const colors: number[] = []
    layer.forEach((head, headIndex) => {
      const base = new THREE.Color(headColors[headIndex % headColors.length])
      const y = 2.9 + this.attentionLayer * 3 + 0.9
      head.forEach((row, query) => {
        row.forEach((weight, key) => {
          if (key > query || weight < 0.006) return
          const start = new THREE.Vector3(this.tokenX[query], y - 1.1, 0)
          const end = new THREE.Vector3(this.tokenX[key], y - 1.1, 0)
          const mid = start.clone().lerp(end, 0.5)
          mid.y += 1.7 + weight * 2.6
          mid.z += 1.6 + headIndex * 0.45
          const curve = new THREE.QuadraticBezierCurve3(start, mid, end)
          const points = curve.getPoints(10)
          for (let i = 0; i < points.length - 1; i += 1) {
            positions.push(points[i].x, points[i].y, points[i].z, points[i + 1].x, points[i + 1].y, points[i + 1].z)
            const shade = base.clone().lerp(new THREE.Color(palette.paper), 1 - clamp(weight * 2.4, 0.08, 1))
            for (let k = 0; k < 2; k += 1) colors.push(shade.r, shade.g, shade.b)
          }
        })
      })
    })
    if (!positions.length) return
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
    })
    this.attentionGroup.add(new THREE.LineSegments(geometry, material))
  }

  private rebuildResidual() {
    disposeObject(this.residualGroup)
    this.residualGroup.clear()
    const half = (this.worldWidth + 4) / 2
    const points = [
      new THREE.Vector3(-half - 1.2, 0.9, -1.7),
      new THREE.Vector3(-half - 2.1, 3.2, -2.2),
      new THREE.Vector3(-half - 2.2, 9.4, -2.2),
      new THREE.Vector3(-half - 1.2, 12.6, -1.7),
    ]
    const curve = new THREE.CatmullRomCurve3(points)
    this.residualCurve = curve
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 64, 0.075, 8, false),
      new THREE.MeshStandardMaterial({
        color: palette.amber,
        transparent: true,
        opacity: 0.34,
        roughness: 0.35,
        emissive: palette.amber,
        emissiveIntensity: 0.15,
      }),
    )
    this.residualGroup.add(tube)
    const pulse = new THREE.Mesh(
      new THREE.SphereGeometry(0.17, 18, 18),
      new THREE.MeshStandardMaterial({ color: palette.amber, emissive: palette.amber, emissiveIntensity: 0.6 }),
    )
    this.residualGroup.add(pulse)
    this.residualPulse = pulse
  }

  private rebuildLogits() {
    disposeObject(this.logitsGroup)
    disposeObject(this.sampleGroup)
    this.logitsGroup.clear()
    this.sampleGroup.clear()
    this.sampleOrb = null
    this.samplePath = null
    const probabilities = this.data.probabilities
    if (!probabilities.length) return
    const count = probabilities.length
    const geometry = new THREE.BoxGeometry(0.09, 1, 0.5)
    const material = new THREE.MeshStandardMaterial({ roughness: 0.38, metalness: 0.02 })
    const mesh = new THREE.InstancedMesh(geometry, material, count)
    const matrix = new THREE.Matrix4()
    const color = new THREE.Color()
    const spacing = 0.14
    const xBase = -((count - 1) * spacing) / 2
    probabilities.forEach((probability, index) => {
      const height = Math.max(0.22, probability * 4.2)
      matrix.makeScale(1, height, 1)
      matrix.setPosition(xBase + index * spacing, 12.2 + height / 2, 1.6)
      mesh.setMatrixAt(index, matrix)
      color.set(probability > 0.12 ? palette.violet : probability > 0.03 ? palette.blue : '#d8d2c4')
      mesh.setColorAt(index, color)
    })
    mesh.instanceMatrix.needsUpdate = true
    mesh.instanceColor!.needsUpdate = true
    mesh.castShadow = true
    this.logitsGroup.add(mesh)

    const top = Array.from(probabilities.keys())
      .sort((a, b) => probabilities[b] - probabilities[a])
      .slice(0, 5)
    top.forEach((index, rank) => {
      const token = this.data.vocab[index] ?? ''
      const label = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: createBarLabel(token === ' ' ? '␣' : token, '#4b3f78'),
          transparent: true,
          depthWrite: false,
        }),
      )
      label.position.set(xBase + index * spacing, 12.2 + probabilities[index] * 4.2 + 0.35, 1.6)
      label.scale.set(0.9, 0.45, 1)
      this.logitsGroup.add(label)
      if (rank === 0) {
        const halo = new THREE.Mesh(
          new THREE.RingGeometry(0.3, 0.36, 32),
          new THREE.MeshBasicMaterial({ color: palette.violet, transparent: true, opacity: 0.65, side: THREE.DoubleSide }),
        )
        halo.rotation.x = -Math.PI / 2
        halo.position.set(xBase + index * spacing, 12.25, 1.6)
        this.logitsGroup.add(halo)
      }
    })
    if (top.length && this.tokenX.length) {
      const topIndex = top[0]
      const start = new THREE.Vector3(xBase + topIndex * spacing, 12.2 + probabilities[topIndex] * 4.2, 1.6)
      const end = new THREE.Vector3(this.tokenX[this.tokenX.length - 1], 0.95, 0.55)
      const control = new THREE.Vector3((start.x + end.x) / 2, 7.5, 4)
      this.samplePath = new THREE.QuadraticBezierCurve3(start, control, end)
      const orb = new THREE.Mesh(
        new THREE.SphereGeometry(0.2, 20, 20),
        new THREE.MeshStandardMaterial({
          color: palette.violet,
          emissive: palette.violet,
          emissiveIntensity: 0.7,
          transparent: true,
          opacity: 0.9,
        }),
      )
      this.sampleGroup.add(orb)
      const halo = new THREE.Mesh(
        new THREE.RingGeometry(0.28, 0.34, 32),
        new THREE.MeshBasicMaterial({ color: palette.violet, transparent: true, opacity: 0.55, side: THREE.DoubleSide }),
      )
      halo.rotation.x = -Math.PI / 2
      halo.position.copy(end)
      this.sampleGroup.add(halo)
      this.sampleOrb = orb
    }
  }

  setChapter(value: number) {
    this.chapterTarget = clamp(value, 0, cameraKeyframes.length - 1)
    const layer = this.chapterTarget >= 2.5 ? Math.min(3, Math.max(0, Math.round(this.chapterTarget) - 3)) : 0
    if (layer !== this.attentionLayer) {
      this.attentionLayer = layer
      this.rebuildAttention()
    }
  }

  setSelection(selection: Selection) {
    this.selected = selection
  }

  setReducedMotion(reduced: boolean) {
    this.reducedMotion = reduced
  }

  private resize() {
    const rect = this.canvas.getBoundingClientRect()
    const width = Math.max(1, rect.width)
    const height = Math.max(1, rect.height)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height, false)
  }

  private handlePointer = (event: PointerEvent) => {
    const rect = this.canvas.getBoundingClientRect()
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
    this.raycaster.setFromCamera(this.pointer, this.camera)
    const tokenHit = this.raycaster.intersectObjects(this.tokenMeshes, false)[0]
    if (tokenHit) {
      const index = tokenHit.object.userData.index as number
      this.selected = { kind: 'token', index }
      this.selectionCallback?.(this.selected)
      return
    }
    const towerHit = this.raycaster.intersectObjects(this.towerMeshes, false)[0]
    if (towerHit) {
      const index = towerHit.object.userData.index as number
      this.selected = { kind: 'layer', index }
      this.selectionCallback?.(this.selected)
    }
  }

  private handleHover = (event: PointerEvent) => {
    const rect = this.canvas.getBoundingClientRect()
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
    this.raycaster.setFromCamera(this.pointer, this.camera)
    const hit = this.raycaster.intersectObjects([...this.tokenMeshes, ...this.towerMeshes], false)[0]
    this.canvas.style.cursor = hit ? 'pointer' : 'grab'
  }

  private tick = () => {
    if (this.disposed) return
    this.frame += 1
    const time = (performance.now() - this.startedAt) / 1000
    this.chapter += (this.chapterTarget - this.chapter) * (this.reducedMotion ? 1 : 0.085)
    const chapter = this.chapter
    const lower = Math.floor(chapter)
    const upper = Math.min(cameraKeyframes.length - 1, lower + 1)
    const blend = chapter - lower
    const start = cameraKeyframes[lower]
    const end = cameraKeyframes[upper]
    const targetPosition = new THREE.Vector3(
      start.position[0] + (end.position[0] - start.position[0]) * blend,
      start.position[1] + (end.position[1] - start.position[1]) * blend,
      start.position[2] + (end.position[2] - start.position[2]) * blend,
    )
    const fit = Math.max(1, 1.5 / Math.max(0.45, this.camera.aspect))
    targetPosition.z += Math.max(0, fit - 1) * 13
    const targetLook = new THREE.Vector3(
      start.target[0] + (end.target[0] - start.target[0]) * blend,
      start.target[1] + (end.target[1] - start.target[1]) * blend,
      start.target[2] + (end.target[2] - start.target[2]) * blend,
    )
    const cameraLerp = this.reducedMotion ? 1 : 0.08
    this.cameraPosition.lerp(targetPosition, cameraLerp)
    this.cameraTarget.lerp(targetLook, cameraLerp)
    this.camera.position.copy(this.cameraPosition)
    this.camera.lookAt(this.cameraTarget)

    const tokenWeight = 0.55 + near(chapter, 0, 1.4) * 0.45
    this.tokenGroup.children.forEach((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const material = (child as THREE.Mesh).material as THREE.MeshStandardMaterial
        material.emissiveIntensity = 0.03 + tokenWeight * 0.16
      }
    })
    this.tokenGroup.position.y = Math.sin(time * 0.8) * 0.025
    this.qkvGroup.visible = near(chapter, 2, 1.5) > 0.05 || chapter > 2
    this.qkvGroup.children.forEach((child) => {
      const material = (child as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined
      if (material?.opacity !== undefined) material.opacity = 0.12 + near(chapter, 2, 1.6) * 0.8
    })
    this.attentionGroup.visible = near(chapter, 3, 2.1) > 0.04 || (chapter > 3 && chapter < 6.4)
    this.attentionGroup.children.forEach((child) => {
      const material = (child as THREE.LineSegments).material as THREE.LineBasicMaterial
      material.opacity = 0.18 + near(chapter, 3, 1.8) * 0.62
    })
    this.residualGroup.visible = near(chapter, 4, 1.6) > 0.04
    this.residualGroup.children.forEach((child) => {
      const material = (child as THREE.Mesh).material as THREE.MeshStandardMaterial
      material.opacity = 0.12 + near(chapter, 4, 1.5) * 0.7
    })
    if (this.residualPulse && this.residualCurve) {
      const pulse = (time * 0.14) % 1
      this.residualPulse.position.copy(this.residualCurve.getPoint(pulse))
    }
    const mlpWeight = near(chapter, 5, 1.5)
    this.mlpPlates.forEach((plates) => {
      plates.forEach((plate, index) => {
        const material = plate.material as THREE.MeshStandardMaterial
        material.opacity = 0.04 + mlpWeight * (0.4 - index * 0.07)
        plate.scale.y = 0.4 + mlpWeight * (1.8 + index * 0.35)
      })
    })
    this.logitsGroup.visible = chapter > 6.1
    this.logitsGroup.children.forEach((child) => {
      const material = (child as THREE.Mesh | THREE.Sprite).material as THREE.MeshStandardMaterial | THREE.SpriteMaterial
      if (material && 'opacity' in material) material.opacity = 0.08 + near(chapter, 7, 1.8) * 0.92
    })
    if (this.particleGroup) {
      const positions = this.particleGroup.geometry.getAttribute('position') as THREE.BufferAttribute
      for (let i = 0; i < positions.count; i += 1) {
        const y = positions.getY(i) + 0.006
        positions.setY(i, y > 13 ? 0.4 : y)
      }
      positions.needsUpdate = true
    }
    this.flowPulses.forEach((pulse, index) => {
      const phase = (time * 0.18 + index / this.flowPulses.length) % 1
      pulse.position.set(Math.sin(phase * Math.PI * 2) * 0.65, 1.2 + phase * 11.2, 1.15)
    })
    const embeddingWeight = near(chapter, 1, 1.45)
    this.embeddingGroup.visible = embeddingWeight > 0.03
    this.embeddingGroup.children.forEach((bar) => {
      const material = (bar as THREE.Mesh).material as THREE.MeshStandardMaterial
      material.opacity = 0.08 + embeddingWeight * 0.78
    })
    const sampleWeight = near(chapter, 8, 1.5)
    this.sampleGroup.visible = sampleWeight > 0.03
    if (this.sampleOrb && this.samplePath) {
      const pathProgress = (time * 0.28) % 1
      this.sampleOrb.position.copy(this.samplePath.getPoint(pathProgress))
      this.sampleOrb.scale.setScalar(0.75 + Math.sin(time * 3) * 0.12)
    }
    this.sampleGroup.children.forEach((child) => {
      const material = (child as THREE.Mesh).material as THREE.MeshStandardMaterial | THREE.MeshBasicMaterial
      if ('opacity' in material) material.opacity = 0.08 + sampleWeight * 0.8
    })

    this.renderer.render(this.scene, this.camera)
  }

  dispose() {
    this.disposed = true
    this.renderer.setAnimationLoop(null)
    this.resizeObserver.disconnect()
    this.canvas.removeEventListener('pointerdown', this.handlePointer)
    this.canvas.removeEventListener('pointermove', this.handleHover)
    this.scene.traverse((object) => {
      const mesh = object as THREE.Mesh
      mesh.geometry?.dispose()
      const material = mesh.material as THREE.Material | THREE.Material[] | undefined
      if (Array.isArray(material)) material.forEach((item) => item.dispose())
      else material?.dispose()
    })
    this.renderer.dispose()
  }
}
