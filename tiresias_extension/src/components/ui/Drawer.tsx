import { h, ComponentChildren, JSX } from 'preact';
import { useEffect, useRef } from 'preact/hooks';

export interface DrawerProps {
  isOpen: boolean;
  onClose: () => void;
  title?: ComponentChildren;
  icon?: ComponentChildren;
  headerRight?: ComponentChildren;
  children?: ComponentChildren;
  closeOnEsc?: boolean;
  className?: string;
  contentClassName?: string;
  width?: string | number;
  ariaLabel?: string;
}

/**
 * Slide-out Lateral Drawer Panel Component
 *
 * Implements right-anchored lateral inspection drawers with slide animation,
 * Escape key listener, header actions, and scrollable content area.
 */
export function Drawer({
  isOpen,
  onClose,
  title,
  icon,
  headerRight,
  children,
  closeOnEsc = true,
  className,
  contentClassName,
  width,
  ariaLabel,
}: DrawerProps) {
  const drawerRef = useRef<HTMLDivElement>(null);

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

  const styleOverride: JSX.CSSProperties = {};
  if (width !== undefined) {
    styleOverride.width = typeof width === 'number' ? `${width}px` : width;
  }

  return (
    <div
      ref={drawerRef}
      className={`tiresias-drawer ${className || ''}`}
      style={styleOverride}
      role="complementary"
      aria-modal="true"
      aria-label={ariaLabel || (typeof title === 'string' ? title : undefined)}
    >
      <div className="tiresias-drawer-header">
        <h3 className="tiresias-drawer-title">
          {icon && <span className="tiresias-drawer-icon" style={{ display: 'inline-flex', alignItems: 'center' }}>{icon}</span>}
          <span>{title}</span>
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {headerRight}
          <button
            type="button"
            className="tiresias-drawer-close"
            onClick={onClose}
            aria-label="Close panel"
            title="Close"
          >
            ✕
          </button>
        </div>
      </div>

      <div className={`tiresias-drawer-content ${contentClassName || ''}`}>
        {children}
      </div>
    </div>
  );
}
