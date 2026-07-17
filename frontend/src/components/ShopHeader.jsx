import { Search, UserRound } from 'lucide-react'

function shopperLabel(user) {
  return user?.name?.trim() || `Shopper ${user?.id ?? ''}`.trim()
}

function ShopHeader({
  users,
  selectedUserId,
  onUserChange,
  searchQuery,
  onSearchQueryChange,
  onSearch,
  loading,
}) {
  const submitSearch = (event) => {
    event.preventDefault()
    onSearch()
  }

  return (
    <header className="shop-header">
      <div className="shop-header__inner">
        <a className="brand" href="#top" aria-label="Morrow home">
          Morrow
        </a>

        <nav className="category-nav" aria-label="Shop categories">
          <a href="#products">Discover</a>
        </nav>

        <form className="shop-search" role="search" onSubmit={submitSearch}>
          <Search aria-hidden="true" />
          <input
            type="search"
            aria-label="Search products"
            placeholder="Search products"
            value={searchQuery}
            onChange={(event) => onSearchQueryChange(event.target.value)}
          />
          <button type="submit" disabled={loading || !searchQuery.trim()}>
            Search
          </button>
        </form>

        <label className="user-picker">
          <UserRound aria-hidden="true" />
          <span className="sr-only">Current shopper</span>
          <select
            aria-label="Current shopper"
            value={selectedUserId}
            disabled={loading}
            onChange={(event) => onUserChange(Number(event.target.value))}
          >
            {users.length === 0 && (
              <option value={selectedUserId}>Loading shoppers</option>
            )}
            {users.map((user) => (
              <option key={user.id} value={user.id}>
                {shopperLabel(user)}
              </option>
            ))}
          </select>
        </label>
      </div>
    </header>
  )
}

export default ShopHeader
