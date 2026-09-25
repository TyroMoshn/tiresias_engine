import { h, ComponentChildren, JSX } from 'preact';

export type BooruRating = 's' | 'q' | 'e';
export type TagCategory = 'artist' | 'character' | 'general' | 'tester';
export type NetworkStatus = 'online' | 'offline' | 'checking';

export type BadgeVariant =
  | BooruRating
  | `rating-${BooruRating}`
  | TagCategory
  | NetworkStatus
  | `status-${NetworkStatus}`
  | 'default'
  | 'gold'
  | 'info';

export type BadgeSize = 'sm' | 'md';

export interface StatusDotProps {
  status: NetworkStatus;
  size?: number;
  className?: string;
  title?: string;
}

/**
 * Status Dot Component
 * Displays a glowing or pulsing indicator dot for network / service health.
 */
export function StatusDot({ status, size = 7, className, title }: StatusDotProps) {
  return (
    <span
      className={`tiresias-status-dot ${status} ${className || ''}`}
      style={{ width: `${size}px`, height: `${size}px` }}
      title={title || `Status: ${status}`}
      aria-label={`Status: ${status}`}
      role="status"
    />
  );
}

export interface BadgeProps {
  variant?: BadgeVariant;
  size?: BadgeSize;
  icon?: ComponentChildren;
  children?: ComponentChildren;
  className?: string;
  title?: string;
  onClick?: (e: JSX.TargetedMouseEvent<HTMLSpanElement>) => void;
  style?: JSX.CSSProperties | string;
}

/**
 * Standard Badge & Tag Component
 *
 * Implements tokenized badges for booru ratings (s, q, e), tag taxonomy categories
 * (artist, character, general, tester), and status indicators.
 */
export function Badge({
  variant = 'default',
  size = 'sm',
  icon,
  children,
  className,
  title,
  onClick,
  style,
}: BadgeProps) {
  const getVariantClass = (v: BadgeVariant): string => {
    switch (v) {
      case 's':
      case 'rating-s':
        return 'tiresias-badge-s';
      case 'q':
      case 'rating-q':
        return 'tiresias-badge-q';
      case 'e':
      case 'rating-e':
        return 'tiresias-badge-e';
      case 'artist':
        return 'tiresias-badge-artist';
      case 'character':
        return 'tiresias-badge-character';
      case 'general':
        return 'tiresias-badge-general';
      case 'tester':
        return 'tiresias-badge-tester';
      case 'online':
      case 'status-online':
      case 'offline':
      case 'status-offline':
      case 'checking':
      case 'status-checking':
        return 'tiresias-badge-status';
      case 'gold':
        return 'tiresias-badge-gold';
      case 'info':
        return 'tiresias-badge-info';
      default:
        return '';
    }
  };

  const isStatusDotVariant =
    variant === 'online' ||
    variant === 'offline' ||
    variant === 'checking' ||
    variant === 'status-online' ||
    variant === 'status-offline' ||
    variant === 'status-checking';

  const normalizedStatus: NetworkStatus | null = isStatusDotVariant
    ? (variant.replace('status-', '') as NetworkStatus)
    : null;

  const sizeStyles: Record<BadgeSize, JSX.CSSProperties> = {
    sm: { fontSize: '11px', padding: '2px 6px' },
    md: { fontSize: '12px', padding: '3px 8px' },
  };

  const variantClass = getVariantClass(variant);

  const customStyle: JSX.CSSProperties = {
    ...sizeStyles[size],
    ...(onClick ? { cursor: 'pointer' } : {}),
    ...(typeof style === 'object' ? style : {}),
  };

  return (
    <span
      className={`tiresias-badge ${variantClass} ${className || ''}`}
      style={customStyle}
      title={title}
      onClick={onClick}
    >
      {normalizedStatus && <StatusDot status={normalizedStatus} size={6} />}
      {icon && <span style={{ display: 'inline-flex', alignItems: 'center', lineHeight: 1 }}>{icon}</span>}
      {children && <span>{children}</span>}
    </span>
  );
}
