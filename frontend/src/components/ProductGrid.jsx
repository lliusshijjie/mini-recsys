import { PackageSearch } from 'lucide-react'

import ProductCard from './ProductCard'

function LoadingGrid() {
  return (
    <div className="product-grid" aria-label="Loading products" aria-busy="true">
      {Array.from({ length: 4 }, (_, index) => (
        <div className="product-skeleton" key={index}>
          <div className="product-skeleton__media" />
          <div className="product-skeleton__line" />
          <div className="product-skeleton__line product-skeleton__line--short" />
        </div>
      ))}
    </div>
  )
}

function ProductGrid({
  items,
  mode,
  selectedItemId,
  feedbackItemId,
  interactionsDisabled,
  onSelect,
  onFeedback,
  loading,
  hasRequested,
  onReturnToRecommendations,
}) {
  if (loading && items.length === 0) return <LoadingGrid />

  if (items.length === 0) {
    return (
      <div className="empty-state">
        <PackageSearch aria-hidden="true" />
        <p className="eyebrow">{hasRequested ? 'Nothing matched' : 'Ready when you are'}</p>
        <h2>{hasRequested ? 'No products found' : 'Your selection is ready'}</h2>
        <p>{hasRequested ? 'The current selection returned no matching products.' : 'A personal edit has not been loaded yet.'}</p>
        <button type="button" onClick={() => onReturnToRecommendations()}>
          {hasRequested ? 'Return to recommendations' : 'Show recommendations'}
        </button>
      </div>
    )
  }

  return (
    <div className="product-grid" id="products">
      {items.map((item, index) => (
        <ProductCard
          key={item.item_id}
          item={item}
          index={index}
          mode={mode}
          selected={item.item_id === selectedItemId}
          feedbackPending={feedbackItemId === item.item_id}
          interactionsDisabled={interactionsDisabled}
          onSelect={onSelect}
          onFeedback={onFeedback}
        />
      ))}
    </div>
  )
}

export default ProductGrid
