import { expect, it, vi } from 'vitest'
import { useState } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import MobileInspectorSheet from './MobileInspectorSheet'

it('isolates the dialog, traps focus, and restores the trigger after close', async () => {
  const user = userEvent.setup()
  const onClose = vi.fn()
  render(
    <>
      <button type="button">Page action</button>
      <MobileInspectorSheet open onClose={onClose}>
        <button type="button">Inspector action</button>
      </MobileInspectorSheet>
    </>,
  )

  const closeButton = screen.getByRole('button', { name: 'Close recommendation diagnostics' })
  const inspectorAction = screen.getByRole('button', { name: 'Inspector action' })
  expect(closeButton).toHaveFocus()
  expect(document.body.style.overflow).toBe('hidden')

  await user.tab()
  expect(inspectorAction).toHaveFocus()

  await user.tab()
  expect(closeButton).toHaveFocus()

  await user.tab({ shift: true })
  expect(inspectorAction).toHaveFocus()

  await user.keyboard('{Escape}')
  expect(onClose).toHaveBeenCalledTimes(1)
})

it('closes from the overlay and returns focus after unmounting', async () => {
  const user = userEvent.setup()
  const onClose = vi.fn()

  function InspectorHarness() {
    const [open, setOpen] = useState(false)
    const close = () => {
      onClose()
      setOpen(false)
    }

    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>Open diagnostics</button>
        <MobileInspectorSheet open={open} onClose={close}>
          Diagnostics
        </MobileInspectorSheet>
      </>
    )
  }

  render(<InspectorHarness />)

  const trigger = screen.getByRole('button', { name: 'Open diagnostics' })
  await user.click(trigger)
  expect(trigger.parentElement).toHaveAttribute('inert')

  await user.click(screen.getByTestId('inspector-backdrop'))
  expect(onClose).toHaveBeenCalledTimes(1)
  await waitFor(() => {
    expect(trigger).toHaveFocus()
  })
  expect(document.body.style.overflow).toBe('')
})
