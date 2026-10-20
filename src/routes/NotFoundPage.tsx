import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <div className="not-found">
      <div>
        <h1>404</h1>
        <p>这个位置不在词表里。</p>
        <Link className="button" to="/">
          回到首页
        </Link>
      </div>
    </div>
  )
}
