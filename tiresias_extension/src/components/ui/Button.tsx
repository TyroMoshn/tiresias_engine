import { h, ComponentChildren, JSX } from 'preact';

export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'success' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends Omit<JSX.HTMLAttributes<HTMLButtonElement>, 'size' | 'icon' | 'style'> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  disabled?: boolean;
  icon?: ComponentChildren;
  children?: ComponentChildren;
  className?: string;
  type?: 'button' | 'submit' | 'reset';
  fullWidth?: boolean;
  style?: JSX.CSSProperties;
}

function LoadingSpinner() {
  return (
    <svg
      style={{
        width: '1em',
        height: '1em',
        animation: 'tiresias-spin 0.8s linear infinite',
        flexShrink: 0,
      }}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        stroke="currentColor"
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * Standard Themed Button Component
 *
 * Supports semantic variants (primary, secondary, danger, success, ghost),
 * sizes (sm, md, lg), loading spinner state, and icon prefixes.
 */
export function Button({
  variant = 'secondary',
  size = 'md',
  loading = false,
  disabled = false,
  icon,
  children,
  className,
  type = 'button',
  fullWidth = false,
  style,
  ...restProps
}: ButtonProps) {
  const isActuallyDisabled = disabled || loading;

  const combinedStyles: JSX.CSSProperties = {
    ...(fullWidth ? { width: '100%' } : {}),
    ...(style || {}),
  };

  const classNames = [
    'tiresias-btn',
    `tiresias-btn-${variant}`,
    `tiresias-btn-${size}`,
    loading ? 'loading' : '',
    className || '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <button
      type={type}
      className={classNames}
      disabled={isActuallyDisabled}
      style={combinedStyles}
      {...restProps}
    >
      {loading ? (
        <LoadingSpinner />
      ) : icon ? (
        <span className="tiresias-btn-icon-prefix" style={{ display: 'inline-flex', alignItems: 'center', lineHeight: 1 }}>
          {icon}
        </span>
      ) : null}
      {children && <span>{children}</span>}
    </button>
  );
}
