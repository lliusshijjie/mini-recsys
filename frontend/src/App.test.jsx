import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import App from './App'
import { getRecommendations, listUsers, recordEvent, searchItems } from './api'

vi.mock('./api', () => ({
  listUsers: vi.fn(),
  getRecommendations: vi.fn(),
  searchItems: vi.fn(),
  recordEvent: vi.fn(),
}))

const users = [
  { id: 1, name: 'Alex' },
  { id: 2, name: 'Jordan' },
]

const headphones = {
  item_id: 11,
  name: 'Wireless headphones',
  price: 129,
  category: 'Electronics',
  image_url: 'https://example.com/headphones.jpg',
  source: 'semantic',
  reason: 'user_profile_similarity',
  final_score: 0.8412,
  sim_score: 0.7921,
  category_score: 0.64,
  popularity: 0.2841,
  price_affinity: 0.44,
  novelty: 0.31,
  feedback_score: 0.2,
}

const lamp = {
  ...headphones,
  item_id: 12,
  name: 'Focused desk lamp',
  price: 48,
  category: 'Home',
  image_url: 'https://example.com/lamp.jpg',
  source: 'category',
  reason: 'preferred_category',
  final_score: 0.734,
}

const searchLamp = {
  item_id: lamp.item_id,
  name: lamp.name,
  price: lamp.price,
  category: lamp.category,
  image_url: lamp.image_url,
  source: 'tantivy',
  reason: 'query_match',
  final_score: 0,
  sim_score: 0,
  category_score: 0,
  popularity: 0,
  price_affinity: 0,
  novelty: 0,
  feedback_score: 0,
}

const recommendationPayload = {
  user: users[0],
  recommendations: [headphones, lamp],
  filtered_count: 2,
}

describe('Morrow storefront', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    listUsers.mockResolvedValue(users)
    getRecommendations.mockResolvedValue(recommendationPayload)
    searchItems.mockResolvedValue([searchLamp])
    recordEvent.mockResolvedValue({ ok: true })
  })

  it('loads recommendations, records impressions, and inspects another product', async () => {
    const user = userEvent.setup()
    render(<App />)

    await screen.findByRole('option', { name: 'Alex' })
    await user.click(screen.getByRole('button', { name: /refresh recommendations/i }))

    expect(getRecommendations).toHaveBeenCalledWith(1)
    expect(
      await screen.findByRole('heading', { name: 'Wireless headphones', level: 2 }),
    ).toBeInTheDocument()
    await waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith({
        uid: 1,
        itemId: 11,
        eventType: 'impression',
      })
      expect(recordEvent).toHaveBeenCalledWith({
        uid: 1,
        itemId: 12,
        eventType: 'impression',
      })
    })

    await user.click(screen.getByRole('button', { name: /inspect focused desk lamp/i }))
    expect(
      screen.getByRole('heading', { name: 'Focused desk lamp', level: 3 }),
    ).toBeInTheDocument()
    expect(screen.getByText('0.7340')).toBeInTheDocument()
  })

  it('searches products and hides recommendation feedback actions', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByRole('searchbox'), 'desk lamp')
    await user.click(screen.getByRole('button', { name: /^search$/i }))

    expect(searchItems).toHaveBeenCalledWith('desk lamp')
    expect(
      await screen.findByRole('heading', { name: 'Focused desk lamp', level: 2 }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /like focused desk lamp/i })).not.toBeInTheDocument()
    expect(screen.getByText(/search results for/i)).toBeInTheDocument()
    expect(screen.queryByText('Filtered')).not.toBeInTheDocument()
    expect(screen.getAllByText('Not available')).toHaveLength(7)

    await user.clear(screen.getByRole('searchbox'))
    await user.type(screen.getByRole('searchbox'), 'another query')
    expect(screen.getByRole('heading', { name: '“desk lamp”', level: 1 })).toBeInTheDocument()
    expect(screen.getByText('Search context: desk lamp')).toBeInTheDocument()
  })

  it('changes shopper without reloading the user directory', async () => {
    const user = userEvent.setup()
    render(<App />)

    await screen.findByRole('option', { name: 'Jordan' })
    await user.selectOptions(screen.getByRole('combobox', { name: 'Current shopper' }), '2')

    expect(screen.getByRole('combobox', { name: 'Current shopper' })).toHaveValue('2')
    expect(listUsers).toHaveBeenCalledTimes(1)
  })

  it('clears recommendation results before changing the feedback owner', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /refresh recommendations/i }))
    await screen.findByRole('heading', { name: 'Wireless headphones', level: 2 })
    await user.selectOptions(screen.getByRole('combobox', { name: 'Current shopper' }), '2')

    expect(screen.queryByRole('heading', { name: 'Wireless headphones', level: 2 })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /like wireless headphones/i })).not.toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Current shopper' })).toHaveValue('2')
  })

  it('writes feedback and refreshes recommendations', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /refresh recommendations/i }))
    await screen.findByRole('heading', { name: 'Wireless headphones', level: 2 })
    await user.click(screen.getByRole('button', { name: /like wireless headphones/i }))

    expect(recordEvent).toHaveBeenCalledWith({ uid: 1, itemId: 11, eventType: 'like' })
    expect(getRecommendations).toHaveBeenCalledTimes(2)
  })

  it('shows actionable request errors', async () => {
    const user = userEvent.setup()
    getRecommendations.mockRejectedValue({
      response: { data: { error: 'Recommendation request timed out' } },
    })
    render(<App />)

    await user.click(screen.getByRole('button', { name: /refresh recommendations/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Recommendation request timed out')
  })
})
