import React, { useState, useEffect, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || window.location.origin

export default function App() {
  const [url, setUrl] = useState('')
  const [jobId, setJobId] = useState(null)
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const polling = useRef(null)

  useEffect(() => {
    if (jobId) {
      polling.current = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/jobs/${jobId}`)
          if (!res.ok) throw new Error('Failed to fetch job')
          const data = await res.json()
          setJob(data)
          if (data.status === 'done' || data.status === 'failed') {
            clearInterval(polling.current)
          }
        } catch (err) {
          setError(err.message)
          clearInterval(polling.current)
        }
      }, 3000)
      return () => clearInterval(polling.current)
    }
  }, [jobId])

  async function startJob() {
    setError(null)
    setJob(null)
    setJobId(null)
    try {
      const res = await fetch(`${API_BASE}/recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: url })
      })
      if (!res.ok) throw new Error('Failed to start job')
      const data = await res.json()
      setJobId(data.job_id)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="container">
      <h1>FaceRecognition — Submit Video URL</h1>
      <div className="form">
        <input
          placeholder="https://.../video.mp4"
          value={url}
          onChange={e => setUrl(e.target.value)}
        />
        <button onClick={startJob} disabled={!url}>Start Job</button>
      </div>

      {error && <div className="error">Error: {error}</div>}

      {jobId && <div className="info">Started job: <strong>{jobId}</strong></div>}

      {job && (
        <div className="job">
          <h3>Job Status: {job.status}</h3>
          <pre>{JSON.stringify(job, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}
