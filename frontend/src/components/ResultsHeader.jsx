import { ArrowLeft, RefreshCw } from 'lucide-react'

function ResultsHeader({
  mode,
  currentUser,
  query,
  itemCount,
  responseTime,
  loading,
  onRefresh,
}) {
  const isSearch = mode === 'search'

  return (
    <section className="results-header" id="top">
      <div className="results-header__copy">
        <p className="eyebrow">
          {isSearch ? 'Search results for' : `Selected for ${currentUser?.name || 'you'}`}
        </p>
        <h1>{isSearch ? <>&ldquo;{query}&rdquo;</> : 'Considered essentials'}</h1>
        <p className="results-header__summary">
          {itemCount > 0
            ? `${itemCount} ${itemCount === 1 ? 'piece' : 'pieces'} in this edit`
            : 'A personal edit, shaped by what you explore'}
          {responseTime !== null && <span>{responseTime} ms response</span>}
        </p>
      </div>

      <button
        type="button"
        className="refresh-button"
        onClick={() => onRefresh()}
        disabled={loading}
        aria-label={isSearch ? 'Back to recommendations' : 'Refresh recommendations'}
      >
        {isSearch ? <ArrowLeft aria-hidden="true" /> : <RefreshCw aria-hidden="true" />}
        <span>{isSearch ? 'Back to recommendations' : 'Refresh edit'}</span>
      </button>
    </section>
  )
}

export default ResultsHeader
