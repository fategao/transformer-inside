import type { ReactNode } from 'react'

export function LabLayout({
  eyebrow,
  title,
  intro,
  children,
  asideTitle,
  aside,
}: {
  eyebrow: string
  title: string
  intro: string
  children: ReactNode
  asideTitle: string
  aside: ReactNode
}) {
  return (
    <>
      <section className="page-hero">
        <div className="container">
          <span className="eyebrow">{eyebrow}</span>
          <h1>{title}</h1>
          <p>{intro}</p>
        </div>
      </section>
      <div className="container lab-shell">
        <div className="lab-main">{children}</div>
        <aside className="lab-aside">
          <h3>{asideTitle}</h3>
          {aside}
        </aside>
      </div>
    </>
  )
}
