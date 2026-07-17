import { expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import ProductCard from './ProductCard'

const item = {
  item_id: 17,
  name: 'Stoneware serving bowl',
  price: 42,
  category: 'Living',
  image_url: '',
}

it('selects the product from its primary action with pointer and keyboard input', async () => {
  const user = userEvent.setup()
  const onSelect = vi.fn()
  const { rerender } = render(
    <ProductCard
      item={item}
      index={0}
      mode="recommendations"
      selected={false}
      feedbackPending={false}
      interactionsDisabled={false}
      onSelect={onSelect}
      onFeedback={vi.fn()}
    />,
  )

  const selectButton = screen.getByRole('button', { name: 'Inspect Stoneware serving bowl' })
  await user.click(selectButton)
  expect(onSelect).toHaveBeenLastCalledWith(17)

  selectButton.focus()
  await user.keyboard('{Enter}')
  await user.keyboard(' ')
  expect(onSelect).toHaveBeenCalledTimes(3)

  rerender(
    <ProductCard
      item={item}
      index={0}
      mode="recommendations"
      selected
      feedbackPending={false}
      interactionsDisabled={false}
      onSelect={onSelect}
      onFeedback={vi.fn()}
    />,
  )
  expect(screen.getByRole('button', { name: 'Inspect Stoneware serving bowl' })).toHaveAttribute('aria-pressed', 'true')
})
