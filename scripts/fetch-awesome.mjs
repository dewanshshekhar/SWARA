#!/usr/bin/env node
/**
 * Fetches the awesome-empath README from GitHub at build time
 * and writes it to docs/en/awesome.md with VitePress frontmatter.
 */
import { writeFileSync, mkdirSync } from 'fs'
import { dirname } from 'path'

const RAW_URL = 'https://raw.githubusercontent.com/empath/awesome-empath/main/README.md'
const OUTPUT = 'docs/en/awesome.md'

const FRONTMATTER = `---
# This file is auto-generated at build time from https://github.com/empath/awesome-empath
# Do not edit manually — changes will be overwritten on next build.
editLink: false
lastUpdated: false
---

<script setup>
const updated = new Date().toISOString().slice(0, 10)
</script>

::: tip Auto-synced from GitHub
This page is automatically pulled from [empath/awesome-empath](https://github.com/empath/awesome-empath) at build time.
Last synced: {{ updated }}
:::

`

async function main() {
  try {
    const res = await fetch(RAW_URL)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    let content = await res.text()

    // Remove the awesome badge line from title (already shown in sidebar)
    content = content.replace(/^#\s+Awesome Empath.*\n/m, '# Awesome Empath\n')

    mkdirSync(dirname(OUTPUT), { recursive: true })
    writeFileSync(OUTPUT, FRONTMATTER + content, 'utf-8')
    console.log(`✓ Fetched awesome-empath README → ${OUTPUT}`)
  } catch (err) {
    console.error(`✗ Failed to fetch awesome-empath: ${err.message}`)
    // Write a fallback so build doesn't break
    const fallback = FRONTMATTER + `# Awesome Empath\n\nFailed to fetch content. Visit [awesome-empath on GitHub](https://github.com/empath/awesome-empath) directly.\n`
    mkdirSync(dirname(OUTPUT), { recursive: true })
    writeFileSync(OUTPUT, fallback, 'utf-8')
    console.log(`✓ Wrote fallback → ${OUTPUT}`)
  }
}

main()
