import { useEffect } from 'react'
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom'
import { SiteHeader } from './components/SiteHeader'
import { SiteFooter } from './components/SiteFooter'
import { HomePage } from './routes/HomePage'
import { WalkthroughPage } from './routes/WalkthroughPage'
import { LabPage } from './routes/LabPage'
import { AboutPage } from './routes/AboutPage'
import { NotFoundPage } from './routes/NotFoundPage'

function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => {
    if (!window.location.hash) window.scrollTo({ top: 0, behavior: 'auto' })
  }, [pathname])
  return null
}

export default function App() {
  const basename = import.meta.env.BASE_URL.replace(/\/$/, '')
  return (
    <BrowserRouter basename={basename}>
      <ScrollToTop />
      <div className="site-shell">
        <a className="skip-link" href="#main-content">
          跳到主要内容
        </a>
        <SiteHeader />
        <main id="main-content" className="site-main">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/walkthrough" element={<WalkthroughPage />} />
            <Route path="/lab/:labId" element={<LabPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </main>
        <SiteFooter />
      </div>
    </BrowserRouter>
  )
}
