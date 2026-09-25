import { h, ComponentChildren, JSX } from 'preact';

export interface SegmentedOption<T = string> {
  value: T;
  label: ComponentChildren;
  icon?: ComponentChildren;
  disabled?: boolean;
  title?: string;
  badge?: ComponentChildren;
}

export interface SegmentedControlProps<T = string> {
  options: Array<SegmentedOption<T>>;
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  size?: 'sm' | 'md';
  className?: string;
  ariaLabel?: string;
  fullWidth?: boolean;
}

/**
 * Standardized Segmented Control Component
 *
 * Replaces ad-hoc button groups (e.g., Privacy: Private / Public, Ratings: s / q / e)
 * with a unified, accessible, and theme-consistent segment switcher.
 */
export function SegmentedControl<T = string>({
  options,
  value,
  onChange,
  disabled = false,
  size = 'md',
  className,
  ariaLabel,
  fullWidth = true,
}: SegmentedControlProps<T>) {
  const sizeStyles: Record<'sm' | 'md', JSX.CSSProperties> = {
    sm: { padding: '4px 8px', fontSize: '11px' },
    md: { padding: '6px 12px', fontSize: '12px' },
  };

  return (
    <div
      className={`tiresias-segmented ${className || ''}`}
      role="radiogroup"
      aria-label={ariaLabel}
      style={{
        display: 'grid',
        gridAutoFlow: 'column',
        gridAutoColumns: fullWidth ? '1fr' : 'auto',
        opacity: disabled ? 0.6 : 1,
      }}
    >
      {options.map((opt) => {
        const isSelected = opt.value === value;
        const isOptionDisabled = disabled || opt.disabled;

        return (
          <button
            key={String(opt.value)}
            type="button"
            role="radio"
            aria-checked={isSelected}
            disabled={isOptionDisabled}
            title={opt.title}
            className={`tiresias-segmented-btn ${isSelected ? 'active' : ''}`}
            style={{
              ...sizeStyles[size],
              cursor: isOptionDisabled ? 'not-allowed' : 'pointer',
            }}
            onClick={() => {
              if (!isOptionDisabled && !isSelected) {
                onChange(opt.value);
              }
            }}
          >
            {opt.icon && (
              <span style={{ display: 'inline-flex', alignItems: 'center', lineHeight: 1 }}>
                {opt.icon}
              </span>
            )}
            <span>{opt.label}</span>
            {opt.badge && (
              <span style={{ marginLeft: '4px', opacity: 0.85 }}>{opt.badge}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
