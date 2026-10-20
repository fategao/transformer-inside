import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const dist = join(root, 'dist')
const config = JSON.parse(await readFile(join(root, 'site.config.json'), 'utf8'))
const basePath = process.env.BASE_PATH ?? config.basePath ?? ''
const origin = config.customDomain
  ? `https://${config.domain}`
  : `https://${config.repo.split('/')[0]}.github.io`
const prefix = config.customDomain ? '' : basePath
const siteUrl = `${origin}${prefix}`

const html = await readFile(join(dist, 'index.html'), 'utf8')
await copyFile(join(dist, 'index.html'), join(dist, '404.html'))
await writeFile(join(dist, '.nojekyll'), '')

const routes = ['walkthrough', 'about', 'lab/attention', 'lab/kv-cache', 'lab/sampling']
for (const route of routes) {
  const directory = join(dist, route)
  await mkdir(directory, { recursive: true })
  await writeFile(join(directory, 'index.html'), html)
}

const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${['', '/walkthrough', '/about', '/lab/attention', '/lab/kv-cache', '/lab/sampling']
  .map((route) => `  <url><loc>${siteUrl}${route}/</loc></url>`)
  .join('\n')}
</urlset>
`
await writeFile(join(dist, 'sitemap.xml'), sitemap)
await writeFile(join(dist, 'robots.txt'), `User-agent: *\nAllow: /\nSitemap: ${siteUrl}/sitemap.xml\n`)

if (config.customDomain) {
  await writeFile(join(dist, 'CNAME'), `${config.domain}\n`)
}

console.log(`postbuild: ${routes.length + 1} static entries, base ${basePath || '/'}`)
