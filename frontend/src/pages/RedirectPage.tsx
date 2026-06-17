import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSessions } from '../context/SessionContext'

export function RedirectPage() {
  const navigate = useNavigate()
  const { sessions, createSession } = useSessions()

  useEffect(() => {
    if (sessions.length > 0) {
      navigate(`/chat/${sessions[0].id}`, { replace: true })
    } else {
      const session = createSession()
      navigate(`/chat/${session.id}`, { replace: true })
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return null
}
