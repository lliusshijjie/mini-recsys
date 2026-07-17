import { AlertCircle } from 'lucide-react'

function Notice({ message }) {
  if (!message) return null

  return (
    <div className="notice" role="alert">
      <AlertCircle aria-hidden="true" />
      <div>
        <strong>Could not complete the request</strong>
        <p>{message}</p>
      </div>
    </div>
  )
}

export default Notice
