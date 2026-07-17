import { Activity, Clock3, Filter, Layers3 } from 'lucide-react'

const SCORE_FIELDS = [
  ['Final score', 'final_score'],
  ['Similarity', 'sim_score'],
  ['Category', 'category_score'],
  ['Popularity', 'popularity'],
  ['Price affinity', 'price_affinity'],
  ['Novelty', 'novelty'],
  ['Feedback', 'feedback_score'],
]

function formatLabel(value, fallback) {
  if (!value) return fallback
  return value.replaceAll('_', ' ')
}

function formatScore(value) {
  return typeof value === 'number' ? value.toFixed(4) : 'Not available'
}

function RecommendationInspector({ mode, selectedItem, responseTime, filteredCount, query }) {
  const isSearch = mode === 'search'

  return (
    <div className="inspector">
      <div className="inspector__brand-row">
        <div>
          <p className="inspector__eyebrow">Recommendation notes</p>
          <p className="inspector__powered">Powered by Mini-RecSys</p>
        </div>
        <Activity aria-hidden="true" />
      </div>

      <div className={`inspector__health${isSearch ? ' inspector__health--single' : ''}`}>
        <div>
          <Clock3 aria-hidden="true" />
          <span>Response</span>
          <strong>{responseTime === null ? 'Waiting' : `${responseTime} ms`}</strong>
        </div>
        {!isSearch && (
          <div>
            <Filter aria-hidden="true" />
            <span>Filtered</span>
            <strong>{filteredCount === null ? 'Waiting' : filteredCount}</strong>
          </div>
        )}
      </div>

      {selectedItem ? (
        <>
          <div className="inspector__item">
            <p className="inspector__section-label">
              {isSearch ? `Search context: ${query}` : 'Selected product'}
            </p>
            <h3>{selectedItem.name}</h3>
            <p>{selectedItem.category}</p>
          </div>

          <div className="inspector__signals">
            <div>
              <Layers3 aria-hidden="true" />
              <span>Source</span>
              <strong>{formatLabel(selectedItem.source, isSearch ? 'search' : 'unknown')}</strong>
            </div>
            <div>
              <span>Reason</span>
              <strong>{formatLabel(selectedItem.reason, isSearch ? 'query match' : 'recommended')}</strong>
            </div>
          </div>

          <dl className="score-list">
            {SCORE_FIELDS.map(([label, field]) => (
              <div key={field}>
                <dt>{label}</dt>
                <dd>{isSearch ? 'Not available' : formatScore(selectedItem[field])}</dd>
              </div>
            ))}
          </dl>

          {isSearch && (
            <p className="inspector__note">
              Recommendation-only scores may be unavailable for search results.
            </p>
          )}
        </>
      ) : (
        <div className="inspector__empty">
          <p>No product is selected.</p>
        </div>
      )}
    </div>
  )
}

export default RecommendationInspector
