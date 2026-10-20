# Transformer 内部漫游 / Inside the Transformer

一个用**真实权重、真实前向传播**讲清楚 decoder-only Transformer 的中文交互站。
访客跟着一句话从字符 ID、嵌入、QKV、多头注意力、残差、MLP 一路走到下一个字的概率分布；
3D 场景是动态的 WebGL 渲染，模型在浏览器 Web Worker 里实时推理，不调用任何 API。

## 页面结构

- `/`：首页，自动循环的 3D 微演示与两个入口。
- `/walkthrough`：9 章滚动叙事主线；进入后自动播放第一段，滚动或点击即接管。
  3D 画面可直接拖动旋转，`Ctrl + 滚轮` 缩放，拖动后出现“回到跟随镜头”；
  手机端提供“全屏 3D”开关。
- `/lab/attention`：注意力显微镜，选层、选头、选查询位置看真实权重。
- `/lab/kv-cache`：KV cache 账本，对比全量重算与缓存复用的计算量和显存。
- `/lab/sampling`：温度采样器，调 temperature / top-k / seed 并生成字符。
- `/about`：模型卡、训练指标、复现步骤与统一参考资料。

## 模型

Tiny Shakespeare 4L：

| 项 | 值 |
|---|---|
| 层数 / 头数 | 4 / 4 |
| d_model / FFN | 128 / 512 |
| 上下文 | 64 |
| 词表 | 65 个字符（字符级） |
| 参数量 | 809,856 |
| 训练 | 5,000 步，batch 64，block 64，seed 1337 |
| 验证 loss | 1.606 |

训练脚本在 `training/train.py`，使用 PyTorch。产物提交到 `public/model/`：

- `weights.bin`：float32 权重（约 3.24 MB）。
- `model.json`：配置、词表、张量偏移、训练指标、权重 SHA-256。
- `reference.json`：固定 prompt 的 PyTorch 参考 logits 与注意力。
- `training-log.json` / `sample.txt`：训练指标与原始生成样例。

浏览器端 TypeScript 推理与 PyTorch 参考结果的 logits / attention 最大绝对误差小于 `2e-4`，
由 `tests/unit/engine.test.ts` 在 CI 中校验。

## 技术栈

React 19 + TypeScript + Vite + Three.js + Tailwind CSS v4 + React Router。
3D 场景封装在 `SceneController` 中，与 React 状态解耦；模型推理放在
`src/engine/worker.ts`，通过 `Float32Array` 在前向过程中暴露每一层中间张量。

场景里每个 token 有一根贯穿四层的竖柱，柱上的节点表示该 token 在对应层的表示；
拖动可以绕模型自由观察，滚动仍负责推进章节。

## 本地运行

```bash
npm install
npm run dev
```

构建与预览（默认部署在 `/transformer-inside/` 子路径）：

```bash
npm run build
npm run preview
# http://127.0.0.1:4174/transformer-inside/
```

本地开发时 `base` 为 `/`；生产构建自动读取 `site.config.json` 的 `basePath`。

## 训练与复现

```bash
python training/train.py
npm test
```

训练脚本会从 GitHub 下载 Tiny Shakespeare（API 不通时会回退到 `curl` 的 git blob 接口）。
如果网络不可用，可手动把语料放到 `training/data/tinyshakespeare.txt`。
脚本默认使用 CUDA；没有 GPU 时加 `--device cpu`（会明显更慢）。

## 验证

```bash
npm run typecheck
npm run lint
npm run test
npm run build
npm run e2e
```

E2E 覆盖桌面 1440×900 与 Pixel 7：自动播放接管、键盘章节导航、三个实验、
深链、`prefers-reduced-motion` 与移动端无横向溢出。

## 部署

GitHub Pages + GitHub Actions。`scripts/postbuild.mjs` 会生成：

- `404.html` 与每个路由的静态入口（深链返回 200）。
- `sitemap.xml`、`robots.txt` 与 `.nojekyll`。

部署后检查：

```bash
npm run deploy:check -- --url https://<user>.github.io/transformer-inside/
```

修改 `site.config.json` 可切换自定义域名与部署子路径。绑定自定义域名时把
`customDomain` 设为 `true`、`basePath` 设为 `""`，并在 GitHub Pages 中填写域名。

## 内容来源

- Attention Is All You Need, Vaswani et al., 2017.
- Language Models are Unsupervised Multitask Learners, Radford et al., OpenAI, 2019.
- nanoGPT, Andrej Karpathy（训练实现参考）。
- Tiny Shakespeare, Andrej Karpathy / char-rnn（公共领域语料）。

体验灵感来自 [bbycroft.net/llm](https://bbycroft.net/llm) 与
[llm-visualized.com](https://www.llm-visualized.com)，未复制其代码、模型或素材。
代码以 MIT 许可发布。
