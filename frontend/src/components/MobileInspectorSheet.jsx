import { useLayoutEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'

function MobileInspectorSheet({ open, onClose, children }) {
  const backdropRef = useRef(null)
  const closeButtonRef = useRef(null)
  const previouslyFocusedRef = useRef(null)

  useLayoutEffect(() => {
    if (!open) return undefined

    const previousOverflow = document.body.style.overflow
    previouslyFocusedRef.current = document.activeElement
    const backdrop = backdropRef.current
    const backgroundElements = Array.from(document.body.children).filter(
      (element) => element !== backdrop,
    )
    const backgroundState = backgroundElements.map((element) => ({
      element,
      inert: element.hasAttribute('inert'),
      ariaHidden: element.getAttribute('aria-hidden'),
    }))
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose()
    }

    document.body.style.overflow = 'hidden'
    backgroundElements.forEach((element) => {
      element.setAttribute('inert', '')
      element.setAttribute('aria-hidden', 'true')
    })
    closeButtonRef.current?.focus()
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      backgroundState.forEach(({ element, inert, ariaHidden }) => {
        if (!inert) element.removeAttribute('inert')
        if (ariaHidden === null) element.removeAttribute('aria-hidden')
        else element.setAttribute('aria-hidden', ariaHidden)
      })
      window.removeEventListener('keydown', handleKeyDown)
      const previouslyFocused = previouslyFocusedRef.current
      queueMicrotask(() => {
        if (previouslyFocused?.isConnected) previouslyFocused.focus()
      })
    }
  }, [open, onClose])

  if (!open) return null

  const keepFocusInside = (event) => {
    if (event.key !== 'Tab') return

    const focusable = Array.from(event.currentTarget.querySelectorAll(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ))
    if (focusable.length === 0) return

    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  return createPortal(
    <div
      ref={backdropRef}
      className="sheet-backdrop"
      data-testid="inspector-backdrop"
      onMouseDown={onClose}
    >
      <section
        className="inspector-sheet"
        role="dialog"
        aria-modal="true"
        aria-label="Recommendation diagnostics"
        onMouseDown={(event) => event.stopPropagation()}
        onKeyDown={keepFocusInside}
      >
        <button
          ref={closeButtonRef}
          type="button"
          className="sheet-close"
          aria-label="Close recommendation diagnostics"
          title="Close"
          onClick={onClose}
        >
          <X aria-hidden="true" />
        </button>
        {children}
      </section>
    </div>,
    document.body,
  )
}

export default MobileInspectorSheet
