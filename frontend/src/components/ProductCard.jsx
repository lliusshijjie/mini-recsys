import { Eye, Heart, X } from 'lucide-react'
import { useState } from 'react'

function ProductCard({
  item,
  index,
  mode,
  selected,
  feedbackPending,
  interactionsDisabled,
  onSelect,
  onFeedback,
}) {
  const [imageFailed, setImageFailed] = useState(false)
  const canGiveFeedback = mode === 'recommendations'
  const initial = item.name?.trim()?.charAt(0)?.toUpperCase() || 'M'

  return (
    <article className={`product-card${selected ? ' product-card--selected' : ''}`}>
      <button
        type="button"
        className="product-card__select"
        aria-label={`Inspect ${item.name}`}
        aria-pressed={selected}
        onClick={() => onSelect(item.item_id)}
      >
        <div className="product-card__media">
          {imageFailed || !item.image_url ? (
            <div className="image-fallback" role="img" aria-label={`${item.name} image unavailable`}>
              <span>{initial}</span>
            </div>
          ) : (
            <img src={item.image_url} alt="" onError={() => setImageFailed(true)} />
          )}
          <span className="product-card__rank">{String(index + 1).padStart(2, '0')}</span>
          <span className="product-card__category">{item.category}</span>
          {selected && <span className="product-card__selected-label">Selected for inspection</span>}
        </div>

        <div className="product-card__details">
        <div className="product-card__title-row">
          <h2>{item.name}</h2>
          <span className="icon-button inspect-button" aria-hidden="true">
            <Eye aria-hidden="true" />
          </span>
        </div>
        <p className="product-card__price">${Number(item.price || 0).toFixed(2)}</p>
        </div>
      </button>

      {canGiveFeedback && (
        <div className="product-card__body">
          <div className="feedback-actions">
            <button
              type="button"
              className="feedback-button feedback-button--like"
              aria-label={`Like ${item.name}`}
              title={`Like ${item.name}`}
              disabled={feedbackPending || interactionsDisabled}
              onClick={() => onFeedback(item.item_id, 'like')}
            >
              <Heart aria-hidden="true" />
              <span>Like</span>
            </button>
            <button
              type="button"
              className="feedback-button"
              aria-label={`Dismiss ${item.name}`}
              title={`Dismiss ${item.name}`}
              disabled={feedbackPending || interactionsDisabled}
              onClick={() => onFeedback(item.item_id, 'dismiss')}
            >
              <X aria-hidden="true" />
              <span>Dismiss</span>
            </button>
          </div>
        </div>
      )}
    </article>
  )
}

export default ProductCard
