import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:3000'

const CATEGORY_COLORS = {
    Electronics: { bg: 'bg-blue-500/20', border: 'border-blue-500', text: 'text-blue-400' },
    Books: { bg: 'bg-amber-500/20', border: 'border-amber-500', text: 'text-amber-400' },
    Home: { bg: 'bg-emerald-500/20', border: 'border-emerald-500', text: 'text-emerald-400' },
    Clothing: { bg: 'bg-pink-500/20', border: 'border-pink-500', text: 'text-pink-400' },
}

function App() {
    const [users, setUsers] = useState([])
    const [selectedUserId, setSelectedUserId] = useState(1)
    const [currentUser, setCurrentUser] = useState(null)
    const [recommendations, setRecommendations] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [responseTime, setResponseTime] = useState(null)
    const [filteredCount, setFilteredCount] = useState(0)
    const [searchQuery, setSearchQuery] = useState('')
    const [searchResults, setSearchResults] = useState([])
    const [isSearchMode, setIsSearchMode] = useState(false)

    useEffect(() => {
        axios.get(`${API_BASE}/users`)
            .then(res => setUsers(res.data.users))
            .catch(() => { })
    }, [])

    const fetchRecommendations = async () => {
        setLoading(true)
        setError('')
        const start = performance.now()

        try {
            const res = await axios.get(`${API_BASE}/recommend?uid=${selectedUserId}`)
            setCurrentUser(res.data.user)
            setRecommendations(res.data.recommendations)
            setFilteredCount(res.data.filtered_count || 0)
            setResponseTime((performance.now() - start).toFixed(0))

            // 自动标记为已看
            const itemIds = res.data.recommendations.map(r => r.item_id)
            if (itemIds.length > 0) {
                axios.post(`${API_BASE}/mark_seen`, {
                    uid: selectedUserId,
                    item_ids: itemIds
                }).catch(() => { })
            }
        } catch (err) {
            setError(err.response?.data?.error || err.message)
            setRecommendations([])
        } finally {
            setLoading(false)
        }
    }

    const handleSearch = async () => {
        if (!searchQuery.trim()) return
        setLoading(true)
        setError('')
        setIsSearchMode(true)
        const start = performance.now()

        try {
            const res = await axios.get(`${API_BASE}/search?q=${encodeURIComponent(searchQuery)}`)
            setSearchResults(res.data.results)
            setRecommendations([])
            setCurrentUser(null)
            setResponseTime((performance.now() - start).toFixed(0))
        } catch (err) {
            setError(err.response?.data?.error || err.message)
            setSearchResults([])
        } finally {
            setLoading(false)
        }
    }

    const getCategoryStyle = (cat) => CATEGORY_COLORS[cat] || CATEGORY_COLORS.Electronics

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
            {/* Header */}
            <header className="border-b border-slate-700 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-xl">🎯</div>
                        <div>
                            <h1 className="text-xl font-bold text-white">Mini-RecSys Dashboard</h1>
                            <p className="text-xs text-slate-400">Rust + C++ FFI Recommendation Engine</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-6">
                        {filteredCount > 0 && (
                            <div className="text-sm text-slate-400">
                                Filtered: <span className="text-amber-400 font-semibold">{filteredCount}</span> seen
                            </div>
                        )}
                        {responseTime && (
                            <div className="text-sm text-slate-400">
                                Response: <span className="text-green-400 font-semibold">{responseTime}ms</span>
                            </div>
                        )}
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-6 py-8">
                {/* Control Bar */}
                <div className="flex flex-wrap gap-4 mb-8 items-center">
                    {/* Search Box */}
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                            placeholder="🔍 Semantic search..."
                            className="px-4 py-3 rounded-lg bg-slate-800 border border-slate-600 text-white focus:border-purple-500 focus:outline-none w-[280px]"
                        />
                        <button
                            onClick={handleSearch}
                            disabled={loading || !searchQuery.trim()}
                            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 rounded-lg font-semibold text-white transition-all disabled:opacity-50"
                        >
                            🔍 Search
                        </button>
                    </div>

                    <div className="border-l border-slate-600 h-8 mx-2"></div>

                    <select
                        value={selectedUserId}
                        onChange={(e) => setSelectedUserId(Number(e.target.value))}
                        className="px-4 py-3 rounded-lg bg-slate-800 border border-slate-600 text-white focus:border-blue-500 focus:outline-none min-w-[200px]"
                    >
                        {users.map(u => (
                            <option key={u.id} value={u.id}>{u.name}</option>
                        ))}
                    </select>

                    <button
                        onClick={() => { setIsSearchMode(false); setSearchResults([]); fetchRecommendations(); }}
                        disabled={loading}
                        className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 rounded-lg font-semibold text-white transition-all disabled:opacity-50"
                    >
                        {loading ? '⏳ Loading...' : '🚀 Recommend'}
                    </button>

                    {(currentUser || isSearchMode) && (
                        <div className="ml-auto text-sm text-slate-400">
                            {isSearchMode
                                ? <>Searching: <span className="text-purple-400 font-medium">"{searchQuery}"</span></>
                                : <>Showing results for: <span className="text-white font-medium">{currentUser?.name}</span></>
                            }
                        </div>
                    )}
                </div>

                {error && (
                    <div className="p-4 mb-6 bg-red-900/30 border border-red-500/50 rounded-xl text-red-300">
                        ⚠️ {error}
                    </div>
                )}

                {/* Results Grid */}
                {(isSearchMode ? searchResults : recommendations).length > 0 ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                        {(isSearchMode ? searchResults : recommendations).map((item, idx) => {
                            const style = getCategoryStyle(item.category)
                            return (
                                <div
                                    key={item.item_id}
                                    className={`group bg-slate-800/50 rounded-xl overflow-hidden border-2 ${style.border} hover:shadow-xl hover:shadow-blue-500/10 transition-all`}
                                >
                                    {/* Rank Badge */}
                                    <div className="absolute top-3 left-3 w-8 h-8 bg-black/60 backdrop-blur rounded-full flex items-center justify-center text-sm font-bold text-white z-10">
                                        #{idx + 1}
                                    </div>

                                    {/* Image */}
                                    <div className="relative aspect-[4/3] bg-slate-700">
                                        <img
                                            src={item.image_url}
                                            alt={item.name}
                                            className="w-full h-full object-cover"
                                        />
                                        <div className={`absolute top-3 right-3 px-2 py-1 rounded text-xs font-medium ${style.bg} ${style.text} backdrop-blur`}>
                                            {item.category}
                                        </div>
                                    </div>

                                    {/* Content */}
                                    <div className="p-4">
                                        <h3 className="font-semibold text-white mb-1 line-clamp-2 group-hover:text-blue-400 transition-colors">
                                            {item.name}
                                        </h3>
                                        <p className="text-xl font-bold text-green-400 mb-3">${item.price.toFixed(2)}</p>

                                        {/* Scores */}
                                        <div className="grid grid-cols-3 gap-2 text-center bg-slate-900/50 rounded-lg p-2">
                                            <div>
                                                <p className="text-[10px] text-slate-500">Final</p>
                                                <p className="text-sm font-bold text-green-400">{item.final_score.toFixed(4)}</p>
                                            </div>
                                            <div>
                                                <p className="text-[10px] text-slate-500">Sim</p>
                                                <p className="text-sm font-medium text-blue-400">{item.sim_score.toFixed(4)}</p>
                                            </div>
                                            <div>
                                                <p className="text-[10px] text-slate-500">Pop</p>
                                                <p className="text-sm font-medium text-purple-400">{item.popularity.toFixed(4)}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                ) : !loading && !error && (
                    <div className="text-center py-20">
                        <div className="text-6xl mb-4">🎯</div>
                        <h3 className="text-xl font-semibold text-slate-400 mb-2">Ready to Recommend</h3>
                        <p className="text-slate-500">Select a user and click the button</p>
                    </div>
                )}
            </main>
        </div>
    )
}

export default App
