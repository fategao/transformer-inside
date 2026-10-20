export type Tokenizer = {
  stoi: Map<string, number>
  itos: string[]
  encode: (text: string, maxLength?: number) => number[]
  decode: (ids: number[]) => string
  sanitize: (text: string) => string
}

export function createTokenizer(vocab: string[]): Tokenizer {
  const stoi = new Map(vocab.map((char, index) => [char, index]))
  const fallback = stoi.get(' ') ?? 0
  const sanitize = (text: string) =>
    Array.from(text)
      .filter((char) => stoi.has(char))
      .join('')
  const encode = (text: string, maxLength = 64) =>
    Array.from(sanitize(text))
      .slice(-maxLength)
      .map((char) => stoi.get(char) ?? fallback)
  const decode = (ids: number[]) => ids.map((id) => vocab[id] ?? '').join('')
  return { stoi, itos: vocab, encode, decode, sanitize }
}
