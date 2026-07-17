import { beforeEach, describe, expect, it, vi } from 'vitest'

const axiosMocks = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}))

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => axiosMocks),
  },
}))

import { getRecommendations, listUsers, recordEvent, searchItems } from './api'

describe('API client', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('loads users from the users endpoint', async () => {
    const users = [{ id: 1, name: 'Alex' }]
    axiosMocks.get.mockResolvedValue({ data: { users } })

    await expect(listUsers()).resolves.toEqual(users)
    expect(axiosMocks.get).toHaveBeenCalledWith('/users')
  })

  it('loads recommendations for a selected user', async () => {
    const payload = { user: { id: 2 }, recommendations: [] }
    axiosMocks.get.mockResolvedValue({ data: payload })

    await expect(getRecommendations(2)).resolves.toEqual(payload)
    expect(axiosMocks.get).toHaveBeenCalledWith('/recommend', { params: { uid: 2 } })
  })

  it('searches with the provided query', async () => {
    const results = [{ item_id: 9, name: 'Desk lamp' }]
    axiosMocks.get.mockResolvedValue({ data: { results } })

    await expect(searchItems('desk lamp')).resolves.toEqual(results)
    expect(axiosMocks.get).toHaveBeenCalledWith('/search', { params: { q: 'desk lamp' } })
  })

  it('maps event fields to the backend contract', async () => {
    axiosMocks.post.mockResolvedValue({ data: { ok: true } })

    await expect(
      recordEvent({ uid: 2, itemId: 9, eventType: 'like' }),
    ).resolves.toEqual({ ok: true })
    expect(axiosMocks.post).toHaveBeenCalledWith('/events', {
      uid: 2,
      item_id: 9,
      event_type: 'like',
    })
  })
})
