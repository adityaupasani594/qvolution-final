import { BrowserRouter, Routes, Route } from 'react-router-dom'
import InputPage from './InputPage.jsx'
import Dashboard from './Dashboard.jsx'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<InputPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </BrowserRouter>
  )
}
