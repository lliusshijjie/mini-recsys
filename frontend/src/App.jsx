import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:3000'

const USER_PERSONAS = {
    1: { name: 'User 1', desc: '喜欢 C++ 和底层技术，对系统编程感兴趣' },
    2: { name: 'User 2', desc: '前端开发者，偏好 React 和现代 Web 技术' },
    3: { name: 'User 3', desc: '数据科学家，关注机器学习和算法' },
    42: { name: 'User 42', desc: '全栈工程师，涉猎广泛' },
    100: { name: 'User 100', desc: '新用户，兴趣待探索' },
}

function App() {
    const [selectedUserId, setSelectedUserId] = useState(1)
    const [recommendations, setRecommendations] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [lastFetchTime, setLastFetchTime] = useState(null)

    const fetchRecommendations = async () => {
        setLoading(true)
        setError('')
        const startTime = performance.now()

        try {
            const res = await axios.get(`${API_BASE}/recommend?uid=${selectedUserId}`)
            setRecommendations(res.data.recommendations)
            setLastFetchTime((performance.now() - startTime).toFixed(0))
        } catch (err) {
            setError(err.response?.data?.error || err.message || 'Request failed')
            setRecommendations([])
            setLastFetchTime(null)
        } finally {
            setLoading(false)
        }
    }

    const currentPersona = USER_PERSONAS[selectedUserId] || {
        name: `User ${selectedUserId}`,
        desc: '标准用户画像'
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
            {/* Header */}
            <header className="border-b border-slate-700 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-xl">
                            🎯
                        </div>
                        <div>
                            <h1 className="text-xl font-bold text-white">Mini-RecSys Dashboard</h1>
                            <p className="text-xs text-slate-400">Rust + C++ FFI Recommendation Engine</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                        <span className="text-slate-400">Backend: localhost:3000</span>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-6 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left Panel - Controls */}
                    <div className="lg:col-span-1 space-y-6">
                        {/* User Selection */}
                        <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
                            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <span>👤</span> Select User
                            </h2>
                            <select
                                value={selectedUserId}
                                onChange={(e) => setSelectedUserId(Number(e.target.value))}
                                className="w-full px-4 py-3 rounded-lg bg-slate-700 border border-slate-600 text-white focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all"
                            >
                                {Object.entries(USER_PERSONAS).map(([id, persona]) => (
                                    <option key={id} value={id}>{persona.name}</option>
                                ))}
                            </select>
                        </div>

                        {/* User Persona */}
                        <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 rounded-xl p-6 border border-blue-500/30">
                            <h3 className="text-sm font-medium text-blue-400 mb-2">当前用户画像</h3>
                            <p className="text-lg font-semibold text-white mb-2">{currentPersona.name}</p>
                            <p className="text-sm text-slate-300">{currentPersona.desc}</p>
                        </div>

                        {/* Action Button */}
                        <button
                            onClick={fetchRecommendations}
                            disabled={loading}
                            className="w-full py-4 px-6 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:from-slate-600 disabled:to-slate-600 rounded-xl font-semibold text-white transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:transform-none shadow-lg shadow-blue-500/25"
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                    </svg>
                                    Loading...
                                </span>
                            ) : (
                                '🚀 Get Recommendations'
                            )}
                        </button>

                        {/* Stats */}
                        {lastFetchTime && (
                            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700 text-center">
                                <p className="text-sm text-slate-400">Response Time</p>
                                <p className="text-2xl font-bold text-green-400">{lastFetchTime}ms</p>
                            </div>
                        )}
                    </div>

                    {/* Right Panel - Results */}
                    <div className="lg:col-span-2">
                        {error && (
                            <div className="p-4 mb-6 bg-red-900/30 border border-red-500/50 rounded-xl text-red-300 flex items-center gap-3">
                                <span className="text-xl">⚠️</span>
                                <span>{error}</span>
                            </div>
                        )}

                        {recommendations.length > 0 ? (
                            <div className="space-y-4">
                                <div className="flex items-center justify-between mb-4">
                                    <h2 className="text-xl font-bold text-white">Top 10 Recommendations</h2>
                                    <span className="text-sm text-slate-400">{recommendations.length} items</span>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {recommendations.map((item, idx) => (
                                        <div
                                            key={item.item_id}
                                            className="group bg-slate-800/50 hover:bg-slate-800 rounded-xl p-5 border border-slate-700 hover:border-slate-600 transition-all hover:shadow-xl hover:shadow-blue-500/5"
                                        >
                                            <div className="flex items-start justify-between mb-3">
                                                <div className="flex items-center gap-3">
                                                    <span className="w-8 h-8 rounded-lg bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-sm font-bold text-slate-300">
                                                        #{idx + 1}
                                                    </span>
                                                    <div>
                                                        <h3 className="font-semibold text-white group-hover:text-blue-400 transition-colors">
                                                            {item.name}
                                                        </h3>
                                                        <p className="text-xs text-slate-500">ID: {item.item_id}</p>
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="grid grid-cols-3 gap-2 text-center">
                                                <div className="bg-slate-900/50 rounded-lg p-2">
                                                    <p className="text-xs text-slate-500 mb-1">Final Score</p>
                                                    <p className="text-lg font-bold text-green-400">
                                                        {item.final_score.toFixed(4)}
                                                    </p>
                                                </div>
                                                <div className="bg-slate-900/50 rounded-lg p-2">
                                                    <p className="text-xs text-slate-500 mb-1">Similarity</p>
                                                    <p className="text-sm font-medium text-blue-400">
                                                        {item.sim_score.toFixed(4)}
                                                    </p>
                                                </div>
                                                <div className="bg-slate-900/50 rounded-lg p-2">
                                                    <p className="text-xs text-slate-500 mb-1">Popularity</p>
                                                    <p className="text-sm font-medium text-purple-400">
                                                        {item.popularity.toFixed(4)}
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ) : !loading && !error && (
                            <div className="h-full flex items-center justify-center">
                                <div className="text-center py-20">
                                    <div className="text-6xl mb-4">🎯</div>
                                    <h3 className="text-xl font-semibold text-slate-400 mb-2">Ready to Recommend</h3>
                                    <p className="text-slate-500">Select a user and click the button to get personalized recommendations</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-slate-800 mt-12 py-6 text-center text-sm text-slate-500">
                <p>Mini-RecSys • Rust + C++ FFI Demo • Powered by Axum & React</p>
            </footer>
        </div>
    )
}

export default App
