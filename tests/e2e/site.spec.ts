import { expect, test } from '@playwright/test'

test.describe('Transformer 内部漫游', () => {
  test('home introduces the model and links to the walkthrough', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { level: 1, name: /Transformer/ })).toBeVisible()
    await expect(page.getByText('809,856', { exact: true })).toBeVisible()
    await page.getByRole('link', { name: /开始漫游/ }).click()
    await expect(page).toHaveURL(/\/walkthrough/)
    await expect(page.getByRole('heading', { name: /Tokenization/ })).toBeVisible()
  })

  test('walkthrough advances chapters and supports keyboard navigation', async ({ page }) => {
    await page.goto('/walkthrough')
    await expect(page.getByRole('heading', { name: /Tokenization/ })).toBeVisible()
    await page.getByRole('button', { name: '下一章' }).click()
    await expect(page.getByRole('heading', { name: /Embedding/ })).toBeVisible()
    await page.keyboard.press('ArrowRight')
    await expect(page.getByRole('heading', { name: /Q \/ K \/ V/ })).toBeVisible()
  })

  test('autoplay moves the scene and hands control to scrolling', async ({ page }) => {
    await page.goto('/walkthrough')
    await expect
      .poll(async () => page.evaluate(() => window.scrollY), { timeout: 6000, intervals: [200, 400, 800] })
      .toBeGreaterThan(0)
    await page.mouse.wheel(0, 2400)
    await page.waitForTimeout(700)
    await expect(page.getByRole('heading', { name: /Q \/ K \/ V/ })).toBeVisible()
  })

  test('dragging the scene enters free orbit and can return to follow mode', async ({ page }) => {
    await page.goto('/walkthrough')
    await page.waitForTimeout(1200)
    const box = await page.locator('canvas').first().boundingBox()
    expect(box).not.toBeNull()
    await page.mouse.move(box!.x + box!.width * 0.5, box!.y + box!.height * 0.15)
    await page.mouse.down()
    await page.mouse.move(box!.x + box!.width * 0.62, box!.y + box!.height * 0.26, { steps: 8 })
    await page.mouse.up()
    await expect(page.getByRole('button', { name: '回到跟随镜头' })).toBeVisible()
    await page.getByRole('button', { name: '回到跟随镜头' }).click()
    await expect(page.getByRole('button', { name: '回到跟随镜头' })).toHaveCount(0)
  })

  test('attention lab recalculates a real 4x4-head trace', async ({ page }) => {
    await page.goto('/lab/attention')
    await expect(page.getByText(/当前输入/)).toBeVisible({ timeout: 20_000 })
    const cells = await page.locator('.heatmap__cell').count()
    expect(cells).toBeGreaterThan(100)
    await page.locator('.lab-toolbar select').nth(0).selectOption('1')
    await page.locator('.lab-toolbar select').nth(1).selectOption('2')
    await expect(page.locator('.heatmap__cell').first()).toBeVisible()
  })

  test('KV cache ledger reacts to the context slider', async ({ page }) => {
    await page.goto('/lab/kv-cache')
    await expect(page.getByRole('heading', { name: 'KV cache 账本' })).toBeVisible()
    const slider = page.locator('input[type="range"]').first()
    const before = await page.locator('.metric strong').first().innerText()
    await slider.fill('64')
    await expect(page.locator('.metric strong').first()).not.toHaveText(before)
    await expect(page.getByText(/每步节省/)).toBeVisible()
    await expect(page.getByText(/256\.0 KB/)).toBeVisible()
  })

  test('sampling lab generates characters in the browser', async ({ page }) => {
    await page.goto('/lab/sampling')
    const button = page.getByRole('button', { name: '生成 24 个字符' })
    await expect(button).toBeVisible({ timeout: 20_000 })
    await button.click()
    await expect(page.locator('.generated-text')).not.toContainText('点击“生成', { timeout: 25_000 })
    await expect(page.locator('.generated-text')).toContainText('ROMEO')
  })

  test('deep links render directly', async ({ page }) => {
    await page.goto('/lab/kv-cache')
    await expect(page.getByRole('heading', { name: 'KV cache 账本' })).toBeVisible()
    await page.goto('/about')
    await expect(page.getByRole('heading', { name: '这个模型是怎么来的' })).toBeVisible()
  })

  test('reduced motion disables autoplay and preserves the static first chapter', async ({ browser }) => {
    const context = await browser.newContext({ reducedMotion: 'reduce' })
    const page = await context.newPage()
    await page.goto('/walkthrough')
    await page.waitForTimeout(1200)
    const scrollY = await page.evaluate(() => window.scrollY)
    expect(scrollY).toBeLessThan(10)
    await context.close()
  })

  test('mobile layout has no horizontal overflow', async ({ page }) => {
    await page.goto('/walkthrough')
    await page.waitForTimeout(800)
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    expect(overflow).toBeLessThanOrEqual(1)
  })
})
