import { useLocation, useNavigate } from 'react-router-dom'
import { useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Label,
} from 'recharts'

function inferSurfaceShape(size) {
  let bestRows = 1
  let bestCols = size
  let bestDiff = Number.POSITIVE_INFINITY

  for (let rows = 1; rows <= Math.sqrt(size); rows += 1) {
    if (size % rows !== 0) continue
    const cols = size / rows
    const diff = Math.abs(cols - rows)
    if (diff < bestDiff) {
      bestDiff = diff
      bestRows = rows
      bestCols = cols
    }
  }

  return { rows: bestRows, cols: bestCols }
}

function toMatrix(values, rows, cols) {
  const matrix = []
  for (let row = 0; row < rows; row += 1) {
    matrix.push(values.slice(row * cols, (row + 1) * cols))
  }
  return matrix
}

function colorScale(value, min, max, hue) {
  const range = Math.max(max - min, 1e-9)
  const normalized = (value - min) / range
  const lightness = 18 + normalized * 50
  const saturation = 65 + normalized * 15
  return `hsl(${hue} ${saturation}% ${lightness}%)`
}

function SurfaceHeatmap({
  title,
  matrix,
  min,
  max,
  hue,
  valueLabel,
}) {
  const rows = matrix.length
  const cols = rows > 0 ? matrix[0].length : 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: 'easeOut' }}
      className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl p-5"
    >
      <h2 className="text-lg font-semibold text-slate-200 mb-3">{title}</h2>

      <div
        className="w-full rounded-xl border border-slate-700 overflow-hidden"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
          aspectRatio: cols / rows,
        }}
      >
        {matrix.flatMap((row, rowIndex) =>
          row.map((value, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              style={{ backgroundColor: colorScale(value, min, max, hue) }}
              title={`Row ${rowIndex + 1}, Col ${colIndex + 1} • ${valueLabel}: ${value.toFixed(6)}`}
            />
          ))
        )}
      </div>

      <div className="flex items-center justify-between mt-3 text-xs text-slate-400">
        <span>{valueLabel} min: {min.toFixed(6)}</span>
        <span>{rows}×{cols}</span>
        <span>{valueLabel} max: {max.toFixed(6)}</span>
      </div>
    </motion.div>
  )
}

export default function Dashboard() {
  const location = useLocation()
  const navigate = useNavigate()
  const values = location.state?.values
  const prediction = location.state?.prediction

  // Redirect back if no data
  useEffect(() => {
    if (!values || values.length === 0 || !prediction) {
      navigate('/')
    }
  }, [values, prediction, navigate])

  if (!values || !prediction) return null

  const predictedCurve = prediction?.predicted_curve || []
  const priceForecastData = prediction?.price_forecast || []
  const putCallData = prediction?.put_call_prices || []

  const shape = inferSurfaceShape(values.length)
  const inputMatrix = toMatrix(values, shape.rows, shape.cols)
  const predictedMatrix = toMatrix(predictedCurve, shape.rows, shape.cols)
  const errorValues = values.map((value, index) =>
    Math.abs((predictedCurve[index] ?? 0) - value),
  )
  const errorMatrix = toMatrix(errorValues, shape.rows, shape.cols)

  const volMin = Math.min(...values, ...predictedCurve)
  const volMax = Math.max(...values, ...predictedCurve)
  const errorMin = 0
  const errorMax = Math.max(...errorValues)

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.7, ease: 'easeOut' }}
      className="min-h-screen bg-gradient-to-br from-slate-900 to-black p-6 flex flex-col"
    >
      {/* Top bar */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">
            Quantum Reservoir{' '}
            <span className="text-cyan-400">Volatility Predictor</span>
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">Dashboard — Analysis Results</p>
        </div>
        <motion.button
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.96 }}
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm rounded-xl border border-slate-600 transition-colors"
        >
          ← New Input
        </motion.button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <SurfaceHeatmap
          title="1️⃣ Input Volatility Surface"
          matrix={inputMatrix}
          min={volMin}
          max={volMax}
          hue={190}
          valueLabel="Vol"
        />

        <SurfaceHeatmap
          title="2️⃣ Predicted Volatility Surface"
          matrix={predictedMatrix}
          min={volMin}
          max={volMax}
          hue={265}
          valueLabel="Vol"
        />

        <SurfaceHeatmap
          title="3️⃣ Error Surface"
          matrix={errorMatrix}
          min={errorMin}
          max={errorMax}
          hue={345}
          valueLabel="Abs Error"
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl p-6"
        >
          <h2 className="text-xl font-semibold text-slate-200 mb-4">Time Series Forecast of Prices</h2>
          <div className="w-full" style={{ height: '290px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={priceForecastData} margin={{ top: 10, right: 25, left: 25, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="day" stroke="#475569">
                  <Label
                    value="Forecast Day"
                    position="insideBottom"
                    offset={-5}
                    fill="#94a3b8"
                    fontSize={12}
                  />
                </XAxis>
                <YAxis stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 11 }}>
                  <Label
                    value="Option Price"
                    angle={-90}
                    position="insideLeft"
                    offset={-8}
                    fill="#94a3b8"
                    fontSize={12}
                  />
                </YAxis>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '10px',
                    color: '#e2e8f0',
                    fontSize: '12px',
                  }}
                />
                <Line type="monotone" dataKey="call" name="Call" stroke="#22d3ee" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="put" name="Put" stroke="#f472b6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl p-6"
        >
          <h2 className="text-xl font-semibold text-slate-200 mb-4">Put vs Call Prices</h2>
          <div className="w-full" style={{ height: '290px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={putCallData} margin={{ top: 10, right: 15, left: 5, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 11 }}>
                  <Label
                    value="Average Forecast Price"
                    angle={-90}
                    position="insideLeft"
                    offset={-2}
                    fill="#94a3b8"
                    fontSize={12}
                  />
                </YAxis>
                <Tooltip
                  formatter={(val) => [Number(val).toFixed(4), 'Price']}
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '10px',
                    color: '#e2e8f0',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="price" fill="#22d3ee" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>
    </motion.div>
  )
}
