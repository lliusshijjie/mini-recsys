import { useState } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:3000'

function App() {
    const [userId, setUserId] = useState('')
    const [recommendations, setRecommendations] = useState([])
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    const fetchRecommendations = async () => {
        if (!userId) return
        setLoading(true)
        setError('')
        try {
            const res = await axios.get(`${API_BASE}/recommend?uid=${userId}`)
            setRecommendations(res.data.recommendations)
        } catch (err) {
            setError(err.response?.data?.error || 'Request failed')
            setRecommendations([])
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white p-8">
            <div className="max-w-4xl mx-auto">
                <h1 className="text-4xl font-bold mb-8 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                    Mini-RecSys Demo
                </h1>

                <div className="flex gap-4 mb-8">
                    <input
                        type="number"
                        placeholder="Enter User ID (1-100)"
                        value={userId}
                        onChange={(e) => setUserId(e.target.value)}
                        className="flex-1 px-4 py-3 rounded-lg bg-slate-700 border border-slate-600 focus:border-blue-500 focus:outline-none"
                    />
                    <button
                        onClick={fetchRecommendations}
                        disabled={loading}
                        className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors disabled:opacity-50"
                    >
                        {loading ? 'Loading...' : 'Get Recommendations'}
                    </button>
                </div>

                {error && (
                    <div className="p-4 mb-6 bg-red-900/50 border border-red-500 rounded-lg text-red-300">
                        {error}
                    </div>
                )}

                {recommendations.length > 0 && (
                    <div className="bg-slate-800/50 rounded-xl p-6 backdrop-blur">
                        <h2 className="text-xl font-semibold mb-4">Top 10 Recommendations</h2>
                        <div className="space-y-3">
                            {recommendations.map((item, idx) => (
                                <div
                                    key={item.item_id}
                                    className="flex items-center justify-between p-4 bg-slate-700/50 rounded-lg hover:bg-slate-700 transition-colors"
                                >
                                    <div className="flex items-center gap-4">
                                        <span className="text-2xl font-bold text-slate-500">#{idx + 1}</span>
                                        <div>
                                            <div className="font-medium">{item.name}</div>
                                            <div className="text-sm text-slate-400">ID: {item.item_id}</div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-lg font-semibold text-green-400">
                                            {(item.final_score * 100).toFixed(1)}%
                                        </div>
                                        <div className="text-xs text-slate-400">
                                            sim: {item.sim_score.toFixed(3)} | pop: {item.popularity.toFixed(3)}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default App
