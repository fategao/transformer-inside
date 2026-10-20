export type Chapter = {
  id: string
  index: number
  title: string
  english: string
  question: string
  conclusion: string
  short: string
  hint: string
  panel: 'tokens' | 'embedding' | 'qkv' | 'attention' | 'residual' | 'mlp' | 'stack' | 'logits' | 'sampling'
}

export const chapters: Chapter[] = [
  {
    id: 'tokenization',
    index: 0,
    title: 'Tokenization',
    english: '文字先变成整数',
    question: '一句话怎么变成模型真正处理的符号？',
    conclusion: '模型看不到字符，只看到词表里的整数 ID。',
    short: '切词',
    hint: '把 prompt 换成一个含空格的短句，观察 ID 与字符是一一对应的。',
    panel: 'tokens',
  },
  {
    id: 'embedding',
    index: 1,
    title: 'Embedding + 位置',
    english: '每个符号获得坐标',
    question: '整数 ID 如何变成有意义的向量？顺序信息又从哪里来？',
    conclusion: '词向量给出“是谁”，位置向量给出“在哪里”，两者相加进入网络。',
    short: '嵌入',
    hint: '同一个字出现在不同位置时，词向量相同，但位置向量不同。',
    panel: 'embedding',
  },
  {
    id: 'qkv',
    index: 2,
    title: 'Q / K / V',
    english: '一个向量，三种角色',
    question: '为什么同一个输入要投影成查询、键和值？',
    conclusion: 'Q 决定“我在找什么”，K 决定“我能被怎样找到”，V 是被读取的内容。',
    short: 'QKV',
    hint: 'Q 和 K 只负责算相关性；真正被加权汇总的是 V。',
    panel: 'qkv',
  },
  {
    id: 'attention',
    index: 3,
    title: '多头注意力',
    english: '按相关性读取上下文',
    question: '模型如何同时从不同角度“看”前面的 token？',
    conclusion: '每个头独立计算注意力，因果掩码保证只能看见当前位置及之前的内容。',
    short: '注意力',
    hint: '把层和头切到不同组合，注意每个查询行的权重加起来都是 1。',
    panel: 'attention',
  },
  {
    id: 'residual',
    index: 4,
    title: '残差 + LayerNorm',
    english: '让深层网络走得稳',
    question: '为什么每层都要把原向量加回来，还要做归一化？',
    conclusion: '残差保留原始信息与梯度通路，LayerNorm 把每个 token 的尺度拉回稳定范围。',
    short: '残差',
    hint: '比较 residualIn、residualMid、residualOut 的范数，看看每层改动了多少。',
    panel: 'residual',
  },
  {
    id: 'mlp',
    index: 5,
    title: 'MLP',
    english: '逐位置的非线性计算',
    question: '注意力之后，模型在哪里做“思考”？',
    conclusion: 'MLP 对每个位置独立地先扩张到 512 维，再压回 128 维。',
    short: 'MLP',
    hint: 'GELU 会把一部分通道压到接近 0，只留下被激活的特征。',
    panel: 'mlp',
  },
  {
    id: 'stack',
    index: 6,
    title: '堆叠 4 层',
    english: '表示被逐层改写',
    question: '四层结构相同，为什么不能只留一层？',
    conclusion: '每层都在上一层的表示上继续改写；层数带来的是组合深度，不是简单重复。',
    short: '堆叠',
    hint: '看每一层输出与初始嵌入的余弦相似度：越往后，表示偏离得越明显。',
    panel: 'stack',
  },
  {
    id: 'logits',
    index: 7,
    title: 'Logits + Softmax',
    english: '分数变成概率',
    question: '最后一个位置的向量，如何变成下一个字的概率？',
    conclusion: '最终向量与词表每一行做点积得到 logits，Softmax 再把它们变成总和为 1 的概率。',
    short: '概率',
    hint: 'logits 可以是任意实数；概率一定在 0 到 1 之间，并且总和为 1。',
    panel: 'logits',
  },
  {
    id: 'sampling',
    index: 8,
    title: '采样',
    english: '概率不是唯一答案',
    question: '为什么同一个 prompt，每次生成的结果可能不同？',
    conclusion: '模型给出分布，采样策略决定从分布里怎么挑；温度越低越保守，top-k 越小候选越少。',
    short: '采样',
    hint: '固定 seed 后结果可复现；把温度调到 0.1 再试一次。',
    panel: 'sampling',
  },
]

export const demoPrompt = 'ROMEO: what light'
