const urlArgIndex = process.argv.indexOf('--url')
const baseUrl = urlArgIndex >= 0 ? process.argv[urlArgIndex + 1] : 'https://fategao.github.io/transformer-inside/'
const root = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`

const checks = ['', 'walkthrough/', 'about/', 'lab/attention/', 'lab/kv-cache/', 'lab/sampling/', 'model/model.json', 'model/weights.bin']
let failed = 0
for (const path of checks) {
  const target = `${root}${path}`
  try {
    const response = await fetch(target, { redirect: 'follow' })
    const ok = response.status >= 200 && response.status < 400
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${response.status} ${target}`)
    if (!ok) failed += 1
  } catch (error) {
    failed += 1
    console.log(`FAIL      ${target} (${error instanceof Error ? error.message : error})`)
  }
}
if (failed) {
  console.error(`${failed} check(s) failed`)
  process.exit(1)
}
console.log('deploy check passed')
