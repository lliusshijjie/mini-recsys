import axios from 'axios'

const apiClient = axios.create({
  baseURL: 'http://localhost:3000',
})

export async function listUsers() {
  const { data } = await apiClient.get('/users')
  return data.users
}

export async function getRecommendations(uid) {
  const { data } = await apiClient.get('/recommend', { params: { uid } })
  return data
}

export async function searchItems(query) {
  const { data } = await apiClient.get('/search', { params: { q: query } })
  return data.results
}

export async function recordEvent({ uid, itemId, eventType }) {
  const { data } = await apiClient.post('/events', {
    uid,
    item_id: itemId,
    event_type: eventType,
  })
  return data
}
