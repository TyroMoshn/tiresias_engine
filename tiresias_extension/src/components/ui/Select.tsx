import { h, JSX } from 'preact';

export type SelectSize = 'sm' | 'md' | 'lg';

export interface SelectOption {
  value: string | number;
  label: string;
  disabled?: boolean;
}

export interface SelectProps extends Omit<JSX.HTMLAttributes<HTMLSelectElement>, 'size' | 'style'> {
  options: Array<SelectOption | string | number>;
  value?: string | number;
  onChange?: (e: JSX.TargetedEvent<HTMLSelectElement>) => void;
  disabled?: boolean;
  error?: boolean | string;
  fullWidth?: boolean;
  size?: SelectSize;
  className?: string;
  placeholder?: string;
  style?: JSX.CSSProperties;
}

/**
 * Standard Themed Dropdown Select Component
 *
 * Implements tokenized select styling with focus ring and error highlights.
 */
export function Select({
  options,
  value,
  onChange,
  disabled = false,
  error = false,
  fullWidth = true,
  size = 'md',
  className,
  placeholder,
  style,
  ...restProps
}: SelectProps) {
  const sizeStyles: Record<SelectSize, JSX.CSSProperties> = {
    sm: { padding: '4px 8px', fontSize: '12px' },
    md: { padding: '7px 10px', fontSize: '13px' },
    lg: { padding: '10px 14px', fontSize: '14px' },
  };

  const hasError = Boolean(error);

  const errorStyles: JSX.CSSProperties = hasError
    ? {
        borderColor: 'var(--tiresias-danger)',
        boxShadow: '0 0 0 2px var(--tiresias-danger-subtle)',
      }
    : {};

  const selectStyle: JSX.CSSProperties = {
    ...sizeStyles[size],
    ...(fullWidth ? { width: '100%' } : {}),
    ...errorStyles,
    ...(style || {}),
  };

  const normalizedOptions: SelectOption[] = options.map((opt) => {
    if (typeof opt === 'object' && opt !== null && 'value' in opt) {
      return opt as SelectOption;
    }
    return {
      value: opt,
      label: String(opt),
    };
  });

  return (
    <select
      value={value}
      onChange={onChange}
      disabled={disabled}
      className={`tiresias-select ${hasError ? 'tiresias-select-error' : ''} ${className || ''}`}
      style={selectStyle}
      aria-invalid={hasError}
      {...restProps}
    >
      {placeholder && (
        <option value="" disabled selected={value === undefined || value === ''}>
          {placeholder}
        </option>
      )}
      {normalizedOptions.map((opt) => (
        <option key={String(opt.value)} value={opt.value} disabled={opt.disabled}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}
