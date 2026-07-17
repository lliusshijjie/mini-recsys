import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChartNoAxesColumn } from 'lucide-react'

import {
  getRecommendations,
  listUsers,
  recordEvent,
  searchItems,
} from './api'
import MobileInspectorSheet from './components/MobileInspectorSheet'
import Notice from './components/Notice'
import ProductGrid from './components/ProductGrid'
import RecommendationInspector from './components/RecommendationInspector'
import ResultsHeader from './components/ResultsHeader'
import ShopHeader from './components/ShopHeader'

function requestErrorMessage(error) {
  return error?.response?.data?.error || error?.message || 'The request could not be completed.'
}

function App() {
  const [users, setUsers] = useState([])
  const [selectedUserId, setSelectedUserId] = useState(1)
  const [currentUser, setCurrentUser] = useState(null)
  const [recommendations, setRecommendations] = useState([])
  const [searchResults, setSearchResults] = useState([])
  const [mode, setMode] = useState('recommendations')
  const [searchQuery, setSearchQuery] = useState('')
  const [activeQuery, setActiveQuery] = useState('')
  const [recommendationUserId, setRecommendationUserId] = useState(null)
  const [selectedItemId, setSelectedItemId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [hasRequested, setHasRequested] = useState(false)
  const [error, setError] = useState('')
  const [responseTime, setResponseTime] = useState(null)
  const [filteredCount, setFilteredCount] = useState(null)
  const [feedbackItemId, setFeedbackItemId] = useState(null)
  const [mobileInspectorOpen, setMobileInspectorOpen] = useState(false)
  const requestGeneration = useRef(0)

  useEffect(() => {
    let active = true

    listUsers()
      .then((loadedUsers) => {
        if (!active) return
        setUsers(loadedUsers)
        if (loadedUsers.length > 0) {
          setSelectedUserId((currentId) => (
            loadedUsers.some((user) => user.id === currentId) ? currentId : loadedUsers[0].id
          ))
        }
      })
      .catch((requestError) => {
        if (active) setError(requestErrorMessage(requestError))
      })

    return () => {
      active = false
    }
  }, [])

  const items = mode === 'search' ? searchResults : recommendations
  const selectedItem = useMemo(
    () => items.find((item) => item.item_id === selectedItemId) || items[0] || null,
    [items, selectedItemId],
  )

  const fetchRecommendations = async (userId = selectedUserId) => {
    const generation = ++requestGeneration.current
    setLoading(true)
    setError('')
    setMode('recommendations')
    const startedAt = performance.now()

    try {
      const data = await getRecommendations(userId)
      if (generation !== requestGeneration.current) return

      const nextItems = data.recommendations || []
      setCurrentUser(data.user || users.find((user) => user.id === userId) || null)
      setRecommendations(nextItems)
      setRecommendationUserId(userId)
      setSearchResults([])
      setActiveQuery('')
      setFilteredCount(data.filtered_count ?? 0)
      setResponseTime(Math.round(performance.now() - startedAt))
      setSelectedItemId(nextItems[0]?.item_id ?? null)
      setHasRequested(true)

      nextItems.forEach((item) => {
        recordEvent({
          uid: userId,
          itemId: item.item_id,
          eventType: 'impression',
        }).catch(() => {})
      })
    } catch (requestError) {
      if (generation !== requestGeneration.current) return
      setError(requestErrorMessage(requestError))
      setRecommendations([])
      setRecommendationUserId(null)
      setSelectedItemId(null)
      setHasRequested(true)
    } finally {
      if (generation === requestGeneration.current) setLoading(false)
    }
  }

  const handleSearch = async () => {
    const query = searchQuery.trim()
    if (!query) return

    const generation = ++requestGeneration.current
    setLoading(true)
    setError('')
    const startedAt = performance.now()

    try {
      const nextItems = await searchItems(query)
      if (generation !== requestGeneration.current) return

      setMode('search')
      setSearchResults(nextItems)
      setRecommendations([])
      setRecommendationUserId(null)
      setCurrentUser(null)
      setActiveQuery(query)
      setFilteredCount(null)
      setResponseTime(Math.round(performance.now() - startedAt))
      setSelectedItemId(nextItems[0]?.item_id ?? null)
      setHasRequested(true)
    } catch (requestError) {
      if (generation !== requestGeneration.current) return
      setMode('search')
      setActiveQuery(query)
      setError(requestErrorMessage(requestError))
      setSearchResults([])
      setSelectedItemId(null)
      setHasRequested(true)
    } finally {
      if (generation === requestGeneration.current) setLoading(false)
    }
  }

  const sendFeedback = async (itemId, eventType) => {
    if (mode !== 'recommendations' || recommendationUserId !== selectedUserId || loading) return

    const userId = selectedUserId
    setFeedbackItemId(itemId)
    setError('')

    try {
      await recordEvent({ uid: userId, itemId, eventType })
      if (selectedUserId === userId && recommendationUserId === userId) {
        await fetchRecommendations(userId)
      }
    } catch (requestError) {
      setError(requestErrorMessage(requestError))
    } finally {
      setFeedbackItemId(null)
    }
  }

  const handleUserChange = (userId) => {
    requestGeneration.current += 1
    setSelectedUserId(userId)
    setCurrentUser(users.find((user) => user.id === userId) || null)
    setRecommendations([])
    setSearchResults([])
    setRecommendationUserId(null)
    setSelectedItemId(null)
    setMode('recommendations')
    setActiveQuery('')
    setFilteredCount(null)
    setResponseTime(null)
    setHasRequested(false)
    setLoading(false)
    setFeedbackItemId(null)
    setMobileInspectorOpen(false)
  }

  const closeMobileInspector = useCallback(() => setMobileInspectorOpen(false), [])
  const interactionsPending = loading || feedbackItemId !== null

  const inspector = (
    <RecommendationInspector
      mode={mode}
      selectedItem={selectedItem}
      responseTime={responseTime}
      filteredCount={filteredCount}
      query={activeQuery}
    />
  )

  return (
    <div className="app-shell">
      <ShopHeader
        users={users}
        selectedUserId={selectedUserId}
        onUserChange={handleUserChange}
        searchQuery={searchQuery}
        onSearchQueryChange={setSearchQuery}
        onSearch={handleSearch}
        loading={interactionsPending}
      />

      <main className="page-shell">
        <ResultsHeader
          mode={mode}
          currentUser={currentUser || users.find((user) => user.id === selectedUserId)}
          query={activeQuery}
          itemCount={items.length}
          responseTime={responseTime}
          loading={loading}
          onRefresh={fetchRecommendations}
        />

        {selectedItem && (
          <button
            type="button"
            className="mobile-inspector-trigger"
            aria-label="Open recommendation diagnostics"
            onClick={() => setMobileInspectorOpen(true)}
          >
            <ChartNoAxesColumn aria-hidden="true" />
            <span>View diagnostics</span>
          </button>
        )}

        <Notice message={error} />

        <div className="content-layout">
          <section className="results-region" aria-label="Products">
            <ProductGrid
              items={items}
              mode={mode}
              selectedItemId={selectedItem?.item_id ?? null}
              feedbackItemId={feedbackItemId}
              interactionsDisabled={
                interactionsPending
                || mode !== 'recommendations'
                || recommendationUserId !== selectedUserId
              }
              onSelect={setSelectedItemId}
              onFeedback={sendFeedback}
              loading={loading}
              hasRequested={hasRequested}
              onReturnToRecommendations={fetchRecommendations}
            />
          </section>

          <aside className="desktop-inspector" aria-label="Recommendation diagnostics">
            {inspector}
          </aside>
        </div>
      </main>

      <MobileInspectorSheet
        open={mobileInspectorOpen}
        onClose={closeMobileInspector}
      >
        {inspector}
      </MobileInspectorSheet>
    </div>
  )
}

export default App
