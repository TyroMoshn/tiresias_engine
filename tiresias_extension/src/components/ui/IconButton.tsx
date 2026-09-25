import { h, ComponentChildren, JSX } from 'preact';

export type IconButtonSize = 'sm' | 'md' | 'lg';
export type IconButtonVariant = 'default' | 'round' | 'ghost' | 'danger' | 'primary';
export type IconButtonActiveVariant = 'default' | 'like' | 'hide' | 'primary' | 'danger';

export interface IconButtonProps extends Omit<JSX.HTMLAttributes<HTMLButtonElement>, 'size' | 'icon' | 'style'> {
  icon: ComponentChildren;
  title: string;
  onClick?: (e: JSX.TargetedMouseEvent<HTMLButtonElement>) => void;
  size?: IconButtonSize;
  variant?: IconButtonVariant;
  active?: boolean;
  activeVariant?: IconButtonActiveVariant;
  disabled?: boolean;
  className?: string;
  type?: 'button' | 'submit' | 'reset';
  style?: JSX.CSSProperties;
}

/**
 * Compact Icon-Only Button Component
 *
 * Designed for headers, post actions, floating bars, and toolbar buttons.
 * Guarantees accessible ARIA labels, tooltips, and thematic hover/active states.
 */
export function IconButton({
  icon,
  title,
  onClick,
  size = 'md',
  variant = 'default',
  active = false,
  activeVariant = 'default',
  disabled = false,
  className,
  type = 'button',
  style,
  ...restProps
}: IconButtonProps) {
  const sizeStyles: Record<IconButtonSize, JSX.CSSProperties> = {
    sm: { padding: '2px 5px', fontSize: '11px', minWidth: '22px', minHeight: '22px' },
    md: { padding: '3px 6px', fontSize: '13px', minWidth: '28px', minHeight: '28px' },
    lg: { padding: '5px 8px', fontSize: '15px', minWidth: '34px', minHeight: '34px' },
  };

  const activeClasses = [];
  if (active) {
    activeClasses.push('active');
    if (activeVariant === 'like') activeClasses.push('active-like');
    else if (activeVariant === 'hide') activeClasses.push('active-hide');
    else if (activeVariant === 'primary') activeClasses.push('tiresias-act-primary');
    else if (activeVariant === 'danger') activeClasses.push('tiresias-act-danger');
  }

  const classNames = [
    'tiresias-btn-icon',
    variant === 'round' ? 'tiresias-btn-icon-round' : '',
    variant === 'danger' ? 'tiresias-act-danger' : '',
    variant === 'primary' ? 'tiresias-act-primary' : '',
    ...activeClasses,
    className || '',
  ]
    .filter(Boolean)
    .join(' ');

  const combinedStyles: JSX.CSSProperties = {
    ...sizeStyles[size],
    ...(style || {}),
  };

  return (
    <button
      type={type}
      className={classNames}
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={title}
      style={combinedStyles}
      {...restProps}
    >
      <span style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', lineHeight: 1 }}>
        {icon}
      </span>
    </button>
  );
}
