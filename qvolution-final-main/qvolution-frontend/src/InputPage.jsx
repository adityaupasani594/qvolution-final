import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'

const REQUIRED_COUNT = 224

function parseValues(raw) {
  return raw
    .trim()
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
}

function validateValues(parts) {
  if (parts.length !== REQUIRED_COUNT) {
    return `Expected exactly ${REQUIRED_COUNT} values, got ${parts.length}.`
  }
  for (let i = 0; i < parts.length; i++) {
    if (isNaN(Number(parts[i])) || parts[i] === '') {
      return `Value at position ${i + 1} is not a valid number: "${parts[i]}".`
    }
  }
  return null
}

export default function InputPage() {
  const navigate = useNavigate()
  const fileRef = useRef(null)
  const [textValue, setTextValue] = useState('')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  function handleFileChange(e) {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (evt) => {
      const content = evt.target.result
      // Flatten all cells from CSV into a single comma-separated string
      const flat = content
        .split(/[\r\n]+/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0)
        .join(',')
      setTextValue(flat)
      setError('')
    }
    reader.readAsText(file)
  }

  async function handleSubmit() {
    const parts = parseValues(textValue)
    const validationError = validateValues(parts)
    if (validationError) {
      setError(validationError)
      return
    }

    const values = parts.map(Number)
    setIsSubmitting(true)
    setError('')

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ values }),
      })

      if (!response.ok) {
        throw new Error(`Prediction request failed (${response.status})`)
      }

      const prediction = await response.json()
      navigate('/dashboard', { state: { values, prediction } })
    } catch (e) {
      setError(e.message || 'Failed to connect to backend. Please start the API server and retry.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-black flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="w-full max-w-2xl bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl p-10"
      >
        {/* Title */}
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-slate-100 tracking-tight leading-tight">
            Quantum Reservoir
          </h1>
          <h1 className="text-3xl font-bold text-cyan-400 tracking-tight leading-tight">
            Volatility Predictor
          </h1>
          <p className="mt-3 text-slate-400 text-sm">
            Upload a CSV file or paste {REQUIRED_COUNT} comma-separated numeric values below.
          </p>
        </div>

        {/* File Upload */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Upload CSV File
          </label>
          <label className="flex items-center justify-center gap-3 w-full cursor-pointer border-2 border-dashed border-slate-600 hover:border-cyan-500 bg-slate-900 rounded-xl p-5 transition-colors duration-200">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-6 h-6 text-slate-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1M12 12V4m0 0L8 8m4-4l4 4"
              />
            </svg>
            <span className="text-slate-400 text-sm">
              Click to choose a <span className="text-cyan-400 font-medium">.csv</span> file
            </span>
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleFileChange}
            />
          </label>
        </div>

        {/* Divider */}
        <div className="flex items-center gap-4 mb-6">
          <hr className="flex-1 border-slate-700" />
          <span className="text-slate-500 text-xs uppercase tracking-widest">or paste values</span>
          <hr className="flex-1 border-slate-700" />
        </div>

        {/* Textarea */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Paste {REQUIRED_COUNT} Comma-Separated Values
          </label>
          <textarea
            className="w-full h-36 bg-slate-900 border border-slate-600 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 rounded-xl p-4 text-slate-200 text-sm font-mono resize-none outline-none placeholder-slate-600 transition-colors duration-200"
            placeholder="0.12, 0.34, 0.56, ... (224 values)"
            value={textValue}
            onChange={(e) => {
              setTextValue(e.target.value)
              setError('')
            }}
          />
          <p className="mt-1 text-xs text-slate-500 text-right">
            {parseValues(textValue).length} / {REQUIRED_COUNT} values detected
          </p>
        </div>

        {/* Inline Error */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-5 flex items-start gap-2 bg-red-950 border border-red-700 rounded-xl px-4 py-3"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-4 h-4 text-red-400 mt-0.5 shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
              />
            </svg>
            <p className="text-red-300 text-sm">{error}</p>
          </motion.div>
        )}

        {/* Submit Button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={handleSubmit}
          disabled={isSubmitting}
          className="w-full py-3.5 bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-bold text-base rounded-xl transition-colors duration-200 shadow-lg shadow-cyan-900/40"
        >
          {isSubmitting ? 'Running Prediction...' : 'Perform Prediction'}
        </motion.button>
      </motion.div>
    </div>
  )
}
