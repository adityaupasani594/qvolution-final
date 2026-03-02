import { useLocation, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import katex from 'katex'
import 'katex/dist/katex.min.css'
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

// ── KaTeX helper ──────────────────────────────────────────────────────────────

function Eq({ math, display = false }) {
  const html = katex.renderToString(math, { throwOnError: false, displayMode: display })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

// ── Surface helpers ───────────────────────────────────────────────────────────

function inferSurfaceShape(size) {
  let bestRows = 1
  let bestCols = size
  let bestDiff = Number.POSITIVE_INFINITY
  for (let rows = 1; rows <= Math.sqrt(size); rows += 1) {
    if (size % rows !== 0) continue
    const cols = size / rows
    const diff = Math.abs(cols - rows)
    if (diff < bestDiff) { bestDiff = diff; bestRows = rows; bestCols = cols }
  }
  return { rows: bestRows, cols: bestCols }
}

function toMatrix(values, rows, cols) {
  const matrix = []
  for (let row = 0; row < rows; row += 1)
    matrix.push(values.slice(row * cols, (row + 1) * cols))
  return matrix
}

function colorScale(value, min, max, hue) {
  const range = Math.max(max - min, 1e-9)
  const normalized = (value - min) / range
  const lightness = 18 + normalized * 50
  const saturation = 65 + normalized * 15
  return `hsl(${hue} ${saturation}% ${lightness}%)`
}

// ── Stats helpers ─────────────────────────────────────────────────────────────

function computeStats(vals) {
  const n = vals.length
  const mean = vals.reduce((a, b) => a + b, 0) / n
  const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n
  return { mean, std: Math.sqrt(variance) }
}

// ── Heatmap explanation builders ──────────────────────────────────────────────

function buildInputExplanation(values, shape, min, max) {
  const { mean, std } = computeStats(values)
  const range = max - min
  const cv = (std / mean) * 100
  const patternNote = cv > 20
    ? 'The high relative spread indicates a pronounced volatility smile or skew — implied vol varies considerably across your strike-maturity grid.'
    : 'The surface shows moderate variation, suggesting a relatively flat implied-volatility regime with mild smile/skew.'
  const levelNote = mean > 0.15
    ? 'The overall volatility level is elevated, consistent with stressed or uncertain market conditions.'
    : 'The overall volatility level is moderate, consistent with a calm market regime.'
  return (
    `Your input volatility surface spans ${shape.rows} strikes x ${shape.cols} maturities (${values.length} total data points).\n\n` +
    `Range:         ${min.toFixed(4)}  ->  ${max.toFixed(4)}  (spread: ${range.toFixed(4)})\n` +
    `Mean vol:      ${mean.toFixed(4)}\n` +
    `Std deviation: ${std.toFixed(4)}  (${cv.toFixed(1)}% relative spread)\n\n` +
    `${patternNote}\n\n${levelNote}\n\n` +
    `Lighter cells represent higher volatility; darker cells represent lower volatility.`
  )
}

function buildPredictedExplanation(values, predictedCurve, shape, volMin, volMax) {
  const errorValues = values.map((v, i) => Math.abs((predictedCurve[i] ?? 0) - v))
  const mae = errorValues.reduce((a, b) => a + b, 0) / errorValues.length
  const { mean: predMean } = computeStats(predictedCurve)
  const { mean: inputMean } = computeStats(values)
  const range = Math.max(volMax - volMin, 1e-9)
  const accuracy = (1 - mae / range) * 100
  const meanDrift = Math.abs(predMean - inputMean)
  const driftNote = meanDrift < 0.005
    ? 'The model closely preserves the mean volatility level of your input - no significant bias shift.'
    : `There is a slight mean shift of ${meanDrift.toFixed(4)}, which typically reflects the quantum reservoir smoothing out local noise.`
  const qualityNote = accuracy > 95
    ? 'Excellent fit - the quantum reservoir has closely reconstructed your volatility surface.'
    : accuracy > 88
    ? 'Good fit - the model captures the broad structure while smoothing fine-grained irregularities.'
    : 'Moderate fit - the model captures the general trend but deviates meaningfully in some regions.'
  return (
    `The quantum reservoir computing model predicted a ${shape.rows}x${shape.cols} volatility surface from your ${values.length}-point input.\n\n` +
    `Prediction accuracy:   ~${accuracy.toFixed(1)}%  (relative to input range)\n` +
    `Mean Absolute Error:   ${mae.toFixed(6)}\n` +
    `Predicted mean vol:    ${predMean.toFixed(4)}\n` +
    `Input mean vol:        ${inputMean.toFixed(4)}\n\n` +
    `${driftNote}\n\n${qualityNote}\n\n` +
    `The predicted surface tends to appear smoother than the input - this is expected behaviour from a learned reservoir model.`
  )
}

function buildErrorExplanation(errorValues, values, shape, errorMax) {
  const mae = errorValues.reduce((a, b) => a + b, 0) / errorValues.length
  const { mean: inputMean } = computeStats(values)
  const mre = (mae / inputMean) * 100
  const highErrorCount = errorValues.filter((e) => e > errorMax * 0.7).length
  const maxIdx = errorValues.indexOf(Math.max(...errorValues))
  const maxRow = Math.floor(maxIdx / shape.cols) + 1
  const maxCol = (maxIdx % shape.cols) + 1
  const severityNote = mre < 5
    ? 'Mean relative error < 5% - the model performs with high accuracy across your entire input surface.'
    : mre < 10
    ? 'Mean relative error is 5-10% - good overall fit with localised deviations.'
    : 'Mean relative error > 10% - the model struggles with certain regions of this particular surface.'
  return (
    `The error surface shows the absolute difference |predicted - actual| at each of the ${errorValues.length} strike-maturity points.\n\n` +
    `Mean Absolute Error:   ${mae.toFixed(6)}\n` +
    `Max Absolute Error:    ${errorMax.toFixed(6)}  (at row ${maxRow}, col ${maxCol})\n` +
    `Mean Relative Error:   ${mre.toFixed(2)}%\n` +
    `High-error cells:      ${highErrorCount} / ${errorValues.length}  (error > 70% of max)\n\n` +
    `${severityNote}\n\n` +
    `Bright pink/red cells mark the strike-maturity combinations where the model deviates most from your input data. The scattered pattern suggests the errors are not systematically biased toward any particular region of the surface.`
  )
}

// ── Price forecast explanation builder (data-driven) ──────────────────────────

function buildForecastStats(predictedCurve, priceForecastData) {
  // Mirror the backend logic to derive anchor / trend from predicted curve
  const center = Math.floor(predictedCurve.length / 2)
  const anchorSigma = predictedCurve[center] ?? 0
  const nearSigma   = predictedCurve[Math.max(center - 8, 0)] ?? 0
  const farSigma    = predictedCurve[Math.min(center + 8, predictedCurve.length - 1)] ?? 0
  const slope = (farSigma - nearSigma) / 16.0
  const trend = slope * 0.12

  const calls = priceForecastData.map((p) => p.call)
  const puts  = priceForecastData.map((p) => p.put)
  const callMin = Math.min(...calls)
  const callMax = Math.max(...calls)
  const putMin  = Math.min(...puts)
  const putMax  = Math.max(...puts)
  const { mean: callMean } = computeStats(calls)
  const { mean: putMean }  = computeStats(puts)
  const trendDir = trend > 0.0001 ? 'upward' : trend < -0.0001 ? 'downward' : 'flat'

  return { anchorSigma, trend, trendDir, callMin, callMax, callMean, putMin, putMax, putMean }
}

// ── Generic text + KaTeX explanation modal ────────────────────────────────────

function ExplanationModal({ title, explanation, chartData, chartLines, onClose }) {
  return (
    <AnimatePresence>
      <motion.div
        key="overlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          key="panel"
          initial={{ opacity: 0, scale: 0.92, y: 24 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.92, y: 24 }}
          transition={{ duration: 0.28, ease: 'easeOut' }}
          className="bg-slate-800 border border-slate-600 rounded-2xl shadow-2xl p-6 max-w-xl w-full mx-4"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-start justify-between mb-4">
            <h3 className="text-base font-semibold text-cyan-400 leading-snug pr-4">{title}</h3>
            <button onClick={onClose} className="text-slate-400 hover:text-white text-2xl leading-none shrink-0 transition-colors" aria-label="Close">&times;</button>
          </div>
          <div className="text-slate-300 text-sm leading-relaxed whitespace-pre-line font-mono bg-slate-900/60 rounded-xl p-4 border border-slate-700 max-h-52 overflow-y-auto">
            {explanation}
          </div>
          {chartData && chartLines && (
            <div className="mt-4">
              <p className="text-xs text-slate-400 mb-2 uppercase tracking-widest">Volatility Curve</p>
              <div style={{ height: '180px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 4, right: 12, left: 0, bottom: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="index" stroke="#475569" tick={{ fill: '#64748b', fontSize: 10 }}
                      label={{ value: 'Data Point Index', position: 'insideBottom', offset: -4, fill: '#64748b', fontSize: 10 }} />
                    <YAxis stroke="#475569" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)} width={42} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0', fontSize: '11px' }}
                      formatter={(val, name) => [Number(val).toFixed(5), name]}
                      labelFormatter={(l) => `Point ${l}`} />
                    {chartLines.map((line) => (
                      <Line key={line.key} type="monotone" dataKey={line.key} name={line.name}
                        stroke={line.color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="flex gap-4 mt-1 justify-center">
                {chartLines.map((line) => (
                  <div key={line.key} className="flex items-center gap-1.5">
                    <span className="inline-block w-5 h-0.5 rounded" style={{ backgroundColor: line.color }} />
                    <span className="text-xs text-slate-400">{line.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="mt-4 flex justify-end">
            <button onClick={onClose} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm rounded-xl border border-slate-600 transition-colors">Close</button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

// ── Price Forecast explanation modal (with KaTeX) ─────────────────────────────

function PriceForecastModal({ predictedCurve, priceForecastData, onClose }) {
  const { anchorSigma, trend, trendDir, callMin, callMax, callMean, putMin, putMax, putMean } =
    buildForecastStats(predictedCurve, priceForecastData)

  const trendNote = trendDir === 'flat'
    ? 'The predicted volatility curve is nearly flat around the anchor, so call and put prices remain relatively stable across the 30-day forecast horizon.'
    : `The predicted volatility curve has a ${trendDir} trend (trend coefficient ≈ ${trend.toExponential(3)}), which causes option prices to drift ${trendDir === 'upward' ? 'higher' : 'lower'} across the forecast horizon. Higher volatility always increases both call and put values under Black-Scholes.`

  const moneyNote = callMean > putMean
    ? 'Calls are priced above puts on average, reflecting the risk-neutral drift embedded in the risk-free rate.'
    : 'Puts are priced above calls on average, which is unusual and may indicate very low input volatilities compressing call value.'

  return (
    <AnimatePresence>
      <motion.div
        key="overlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          key="panel"
          initial={{ opacity: 0, scale: 0.92, y: 24 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.92, y: 24 }}
          transition={{ duration: 0.28, ease: 'easeOut' }}
          className="bg-slate-800 border border-slate-600 rounded-2xl shadow-2xl p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <h3 className="text-base font-semibold text-cyan-400 leading-snug pr-4">
              Time Series Forecast of Prices — Explanation
            </h3>
            <button onClick={onClose} className="text-slate-400 hover:text-white text-2xl leading-none shrink-0 transition-colors" aria-label="Close">&times;</button>
          </div>

          {/* Overview */}
          <div className="text-slate-300 text-sm leading-relaxed bg-slate-900/60 rounded-xl p-4 border border-slate-700 mb-4">
            <p>
              This graph shows the 30-day forecast of European call and put option prices,
              derived from your input volatility surface using the <span className="text-cyan-400 font-medium">Black-Scholes model</span>.
              The anchor volatility was extracted from the centre of the quantum-predicted curve
              (<Eq math={`\\sigma_{\\text{anchor}} = ${anchorSigma.toFixed(5)}`} />),
              then evolved day-by-day with a learned trend and a small seasonal oscillation.
            </p>
          </div>

          {/* Black-Scholes equations */}
          <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700 mb-4">
            <p className="text-xs text-slate-400 uppercase tracking-widest mb-3">Black-Scholes Pricing Formulas</p>

            <p className="text-slate-400 text-xs mb-1">European call price:</p>
            <div className="flex justify-center my-2">
              <Eq math="C = S\,N(d_1) - K e^{-rT} N(d_2)" display={true} />
            </div>

            <p className="text-slate-400 text-xs mb-1 mt-3">European put price:</p>
            <div className="flex justify-center my-2">
              <Eq math="P = K e^{-rT} N(-d_2) - S\,N(-d_1)" display={true} />
            </div>

            <p className="text-slate-400 text-xs mb-1 mt-3">where:</p>
            <div className="flex justify-center my-2">
              <Eq math="d_1 = \frac{\ln(S/K) + \bigl(r + \tfrac{\sigma^2}{2}\bigr)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}" display={true} />
            </div>

            <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-slate-400 text-xs">
              <span><Eq math="S = 100.00" /> &mdash; current underlying price</span>
              <span><Eq math="K = 100.00" /> &mdash; strike price</span>
              <span><Eq math="r = 0.05" /> &mdash; risk-free interest rate</span>
              <span><Eq math="T = 1.0" /> &mdash; time to maturity (years)</span>
              <span className="col-span-2"><Eq math={`\\sigma_t \\approx ${anchorSigma.toFixed(4)} + \\text{trend} \\cdot t + \\text{seasonal}(t)`} /> &mdash; volatility at day <Eq math="t" /></span>
            </div>
            <p className="text-slate-500 text-xs mt-2">
              <Eq math="N(\cdot)" /> denotes the CDF of the standard normal distribution.
            </p>
          </div>

          {/* Data-driven stats */}
          <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700 mb-4">
            <p className="text-xs text-slate-400 uppercase tracking-widest mb-3">Forecast Statistics (your data)</p>
            <div className="grid grid-cols-2 gap-3 text-xs text-slate-300">
              <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
                <p className="text-cyan-400 font-medium mb-1">Call Option</p>
                <p>Range: {callMin.toFixed(4)} — {callMax.toFixed(4)}</p>
                <p>Mean:  {callMean.toFixed(4)}</p>
              </div>
              <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
                <p className="text-pink-400 font-medium mb-1">Put Option</p>
                <p>Range: {putMin.toFixed(4)} — {putMax.toFixed(4)}</p>
                <p>Mean:  {putMean.toFixed(4)}</p>
              </div>
            </div>
          </div>

          {/* Interpretation */}
          <div className="text-slate-300 text-sm leading-relaxed bg-slate-900/60 rounded-xl p-4 border border-slate-700 mb-4">
            <p className="text-xs text-slate-400 uppercase tracking-widest mb-2">Interpretation</p>
            <p className="mb-2">{trendNote}</p>
            <p>{moneyNote}</p>
          </div>

          <div className="flex justify-end">
            <button onClick={onClose} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm rounded-xl border border-slate-600 transition-colors">Close</button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

// ── Put vs Call explanation modal ───────────────────────────────────────────────

function PutCallModal({ putCallData, priceForecastData, predictedCurve, onClose }) {
  const { mean: callMean } = computeStats(priceForecastData.map((p) => p.call))
  const { mean: putMean }  = computeStats(priceForecastData.map((p) => p.put))
  const diff    = callMean - putMean
  const absDiff = Math.abs(diff)
  const pctDiff = callMean > 0 ? (absDiff / callMean) * 100 : 0
  const higher  = diff > 0 ? 'call' : 'put'

  const center = Math.floor(predictedCurve.length / 2)
  const anchorSigma = predictedCurve[center] ?? 0

  const parityNote = diff > 0
    ? `The call is priced ${absDiff.toFixed(4)} higher than the put (${pctDiff.toFixed(1)}% difference). This is consistent with put-call parity under a positive risk-free rate (r = 5%), where the forward price exceeds the spot, giving calls an edge.`
    : `The put is priced ${absDiff.toFixed(4)} higher than the call (${pctDiff.toFixed(1)}% difference). This can occur when the input volatility is very low, compressing the call's time value below the put's intrinsic discounted value.`

  const volNote = anchorSigma > 0.20
    ? `Your input implied volatility is elevated (anchor σ ≈ ${anchorSigma.toFixed(4)}), which inflates both option prices — higher vol expands the probability of large price moves in either direction.`
    : `Your input implied volatility is moderate (anchor σ ≈ ${anchorSigma.toFixed(4)}), producing tighter option premiums. Small changes in σ have an outsized effect at this range.`

  return (
    <AnimatePresence>
      <motion.div
        key="overlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          key="panel"
          initial={{ opacity: 0, scale: 0.92, y: 24 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.92, y: 24 }}
          transition={{ duration: 0.28, ease: 'easeOut' }}
          className="bg-slate-800 border border-slate-600 rounded-2xl shadow-2xl p-6 max-w-xl w-full mx-4"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <h3 className="text-base font-semibold text-cyan-400 leading-snug pr-4">
              Put vs Call Prices — Explanation
            </h3>
            <button onClick={onClose} className="text-slate-400 hover:text-white text-2xl leading-none shrink-0 transition-colors" aria-label="Close">&times;</button>
          </div>

          {/* Overview */}
          <div className="text-slate-300 text-sm leading-relaxed bg-slate-900/60 rounded-xl p-4 border border-slate-700 mb-4">
            <p>
              This bar chart compares the <span className="text-cyan-400 font-medium">average call price</span> and{' '}
              <span className="text-pink-400 font-medium">average put price</span> across the full 30-day forecast horizon,
              both derived from your input volatility surface via the Black-Scholes model.
              The bars represent time-averaged values — they summarise the forecast in a single comparable figure.
            </p>
          </div>

          {/* Stats */}
          <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700 mb-4">
            <p className="text-xs text-slate-400 uppercase tracking-widest mb-3">Price Summary (your data)</p>
            <div className="grid grid-cols-2 gap-3 text-xs text-slate-300">
              <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
                <p className="text-cyan-400 font-medium mb-1">Call Option</p>
                <p>Avg forecast price: <span className="text-slate-100 font-mono">{callMean.toFixed(4)}</span></p>
              </div>
              <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
                <p className="text-pink-400 font-medium mb-1">Put Option</p>
                <p>Avg forecast price: <span className="text-slate-100 font-mono">{putMean.toFixed(4)}</span></p>
              </div>
            </div>
            <div className="mt-3 text-xs text-slate-400 font-mono bg-slate-800 rounded-lg p-3 border border-slate-700">
              <span>Absolute difference: <span className="text-slate-200">{absDiff.toFixed(4)}</span></span>
              <span className="mx-3">|</span>
              <span>Relative gap: <span className="text-slate-200">{pctDiff.toFixed(1)}%</span></span>
              <span className="mx-3">|</span>
              <span>Higher: <span className={higher === 'call' ? 'text-cyan-400' : 'text-pink-400'}>{higher.toUpperCase()}</span></span>
            </div>
          </div>

          {/* Interpretation */}
          <div className="text-slate-300 text-sm leading-relaxed bg-slate-900/60 rounded-xl p-4 border border-slate-700 mb-4">
            <p className="text-xs text-slate-400 uppercase tracking-widest mb-2">Interpretation</p>
            <p className="mb-2">{parityNote}</p>
            <p>{volNote}</p>
          </div>

          <div className="flex justify-end">
            <button onClick={onClose} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm rounded-xl border border-slate-600 transition-colors">Close</button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

// ── Heatmap card ──────────────────────────────────────────────────────────────

function SurfaceHeatmap({ title, matrix, min, max, hue, valueLabel, onExplain }) {
  const rows = matrix.length
  const cols = rows > 0 ? matrix[0].length : 0
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: 'easeOut' }}
      className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl p-5 flex flex-col"
    >
      <h2 className="text-lg font-semibold text-slate-200 mb-3">{title}</h2>
      <div
        className="w-full rounded-xl border border-slate-700 overflow-hidden"
        style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, aspectRatio: cols / rows }}
      >
        {matrix.flatMap((row, rowIndex) =>
          row.map((value, colIndex) => (
            <div key={`${rowIndex}-${colIndex}`}
              style={{ backgroundColor: colorScale(value, min, max, hue) }}
              title={`Row ${rowIndex + 1}, Col ${colIndex + 1} x ${valueLabel}: ${value.toFixed(6)}`} />
          ))
        )}
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
        <span>{valueLabel} min: {min.toFixed(6)}</span>
        <span>{rows}x{cols}</span>
        <span>{valueLabel} max: {max.toFixed(6)}</span>
      </div>
      <div className="mt-3 flex justify-end">
        <motion.button
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.96 }}
          onClick={onExplain}
          className="px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-600 bg-slate-700 hover:bg-cyan-600/25 hover:border-cyan-500 text-slate-300 hover:text-cyan-300 transition-colors duration-200"
        >
          Explain the Map
        </motion.button>
      </div>
    </motion.div>
  )
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const location = useLocation()
  const navigate = useNavigate()
  const values = location.state?.values
  const prediction = location.state?.prediction

  const [activeExplanation, setActiveExplanation] = useState(null)
  const [showForecastExplain, setShowForecastExplain] = useState(false)
  const [showPutCallExplain,  setShowPutCallExplain]  = useState(false)

  useEffect(() => {
    if (!values || values.length === 0 || !prediction) navigate('/')
  }, [values, prediction, navigate])

  if (!values || !prediction) return null

  const predictedCurve = prediction?.predicted_curve || []
  const priceForecastData = prediction?.price_forecast || []
  const putCallData = prediction?.put_call_prices || []

  const shape = inferSurfaceShape(values.length)
  const inputMatrix = toMatrix(values, shape.rows, shape.cols)
  const predictedMatrix = toMatrix(predictedCurve, shape.rows, shape.cols)
  const errorValues = values.map((value, index) => Math.abs((predictedCurve[index] ?? 0) - value))
  const errorMatrix = toMatrix(errorValues, shape.rows, shape.cols)

  const volMin = Math.min(...values, ...predictedCurve)
  const volMax = Math.max(...values, ...predictedCurve)
  const errorMin = 0
  const errorMax = Math.max(...errorValues)

  const inputChartData = values.map((v, i) => ({ index: i, input: v }))
  const predictedChartData = values.map((v, i) => ({ index: i, input: v, predicted: predictedCurve[i] ?? 0 }))

  const explanations = {
    input: {
      title: 'Input Volatility Surface — Explanation',
      text: buildInputExplanation(values, shape, volMin, volMax),
      chartData: inputChartData,
      chartLines: [{ key: 'input', color: '#22d3ee', name: 'Input Volatility' }],
    },
    predicted: {
      title: 'Predicted Volatility Surface — Explanation',
      text: buildPredictedExplanation(values, predictedCurve, shape, volMin, volMax),
      chartData: predictedChartData,
      chartLines: [
        { key: 'input',     color: '#38bdf8', name: 'Input' },
        { key: 'predicted', color: '#a78bfa', name: 'Predicted' },
      ],
    },
    error: {
      title: 'Error Surface — Explanation',
      text: buildErrorExplanation(errorValues, values, shape, errorMax),
    },
  }

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
            Quantum Reservoir <span className="text-cyan-400">Volatility Predictor</span>
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">Dashboard — Analysis Results</p>
        </div>
        <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm rounded-xl border border-slate-600 transition-colors">
          New Input
        </motion.button>
      </div>

      {/* Heatmaps */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <SurfaceHeatmap title="1️⃣ Input Volatility Surface"   matrix={inputMatrix}     min={volMin}   max={volMax}   hue={190} valueLabel="Vol"       onExplain={() => setActiveExplanation(explanations.input)} />
        <SurfaceHeatmap title="2️⃣ Predicted Volatility Surface" matrix={predictedMatrix} min={volMin}   max={volMax}   hue={265} valueLabel="Vol"       onExplain={() => setActiveExplanation(explanations.predicted)} />
        <SurfaceHeatmap title="3️⃣ Error Surface"              matrix={errorMatrix}     min={errorMin} max={errorMax} hue={345} valueLabel="Abs Error" onExplain={() => setActiveExplanation(explanations.error)} />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">

        {/* Time Series Forecast */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl p-6 flex flex-col"
        >
          <h2 className="text-xl font-semibold text-slate-200 mb-4">4️⃣ Time Series Forecast of Prices</h2>
          <div className="w-full" style={{ height: '290px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={priceForecastData} margin={{ top: 10, right: 25, left: 25, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="day" stroke="#475569">
                  <Label value="Forecast Day" position="insideBottom" offset={-5} fill="#94a3b8" fontSize={12} />
                </XAxis>
                <YAxis stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 11 }}>
                  <Label value="Option Price" angle={-90} position="insideLeft" offset={-8} fill="#94a3b8" fontSize={12} />
                </YAxis>
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '10px', color: '#e2e8f0', fontSize: '12px' }} />
                <Line type="monotone" dataKey="call" name="Call" stroke="#22d3ee" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="put"  name="Put"  stroke="#f472b6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {/* Explain button */}
          <div className="mt-3 flex justify-end">
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              onClick={() => setShowForecastExplain(true)}
              className="px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-600 bg-slate-700 hover:bg-cyan-600/25 hover:border-cyan-500 text-slate-300 hover:text-cyan-300 transition-colors duration-200"
            >
              Explain the Graph
            </motion.button>
          </div>
        </motion.div>

        {/* Put vs Call */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl p-6 flex flex-col"
        >
          <h2 className="text-xl font-semibold text-slate-200 mb-4">5️⃣ Put vs Call Prices</h2>
          <div className="w-full" style={{ height: '290px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={putCallData} margin={{ top: 10, right: 15, left: 5, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 11 }}>
                  <Label value="Average Forecast Price" angle={-90} position="insideLeft" offset={-2} fill="#94a3b8" fontSize={12} />
                </YAxis>
                <Tooltip formatter={(val) => [Number(val).toFixed(4), 'Price']}
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '10px', color: '#e2e8f0', fontSize: '12px' }} />
                <Bar dataKey="price" fill="#22d3ee" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 flex justify-end">
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              onClick={() => setShowPutCallExplain(true)}
              className="px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-600 bg-slate-700 hover:bg-cyan-600/25 hover:border-cyan-500 text-slate-300 hover:text-cyan-300 transition-colors duration-200"
            >
              Explain the Graph
            </motion.button>
          </div>
        </motion.div>
      </div>

      {/* Heatmap explanation modal */}
      {activeExplanation && (
        <ExplanationModal
          title={activeExplanation.title}
          explanation={activeExplanation.text}
          chartData={activeExplanation.chartData}
          chartLines={activeExplanation.chartLines}
          onClose={() => setActiveExplanation(null)}
        />
      )}

      {/* Price forecast explanation modal */}
      {showForecastExplain && (
        <PriceForecastModal
          predictedCurve={predictedCurve}
          priceForecastData={priceForecastData}
          onClose={() => setShowForecastExplain(false)}
        />
      )}

      {/* Put vs Call explanation modal */}
      {showPutCallExplain && (
        <PutCallModal
          putCallData={putCallData}
          priceForecastData={priceForecastData}
          predictedCurve={predictedCurve}
          onClose={() => setShowPutCallExplain(false)}
        />
      )}
    </motion.div>
  )
}
