import { h, ComponentChildren, JSX } from 'preact';
import { useEffect, useRef } from 'preact/hooks';

export type ModalSize = 'sm' | 'md' | 'lg' | 'xl';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: ComponentChildren;
  icon?: ComponentChildren;
  size?: ModalSize;
  footer?: ComponentChildren;
  closeOnBackdrop?: boolean;
  closeOnEsc?: boolean;
  children?: ComponentChildren;
  className?: string;
  cardClassName?: string;
  bodyClassName?: string;
  ariaLabel?: string;
}

/**
 * Universal Modal Dialog Component
 *
 * Implements accessible modal dialogs with backdrop dismissal, keyboard Escape handling,
 * tokenized header with icon, scrollable body, and action footer.
 */
export function Modal({
  isOpen,
  onClose,
  title,
  icon,
  size = 'md',
  footer,
  closeOnBackdrop = true,
  closeOnEsc = true,
  children,
  className,
  cardClassName,
  bodyClassName,
  ariaLabel,
}: ModalProps) {
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen || !closeOnEsc) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, closeOnEsc, onClose]);

  if (!isOpen) return null;

  const handleOverlayClick = (e: JSX.TargetedMouseEvent<HTMLDivElement>) => {
    if (closeOnBackdrop && e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className={`tiresias-modal-overlay ${className || ''}`}
      onClick={handleOverlayClick}
      role="presentation"
    >
      <div
        ref={cardRef}
        className={`tiresias-modal-card tiresias-modal-${size} ${cardClassName || ''}`}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel || (typeof title === 'string' ? title : undefined)}
      >
        <div className="tiresias-modal-header">
          <h3 className="tiresias-modal-title">
            {icon && <span className="tiresias-modal-icon" style={{ display: 'inline-flex', alignItems: 'center' }}>{icon}</span>}
            <span>{title}</span>
          </h3>
          <button
            type="button"
            className="tiresias-modal-close"
            onClick={onClose}
            aria-label="Close dialog"
            title="Close"
          >
            ✕
          </button>
        </div>

        <div className={`tiresias-modal-body ${bodyClassName || ''}`}>
          {children}
        </div>

        {footer && (
          <div className="tiresias-modal-footer">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}
