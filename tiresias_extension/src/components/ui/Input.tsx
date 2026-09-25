import { h, ComponentChildren, JSX } from 'preact';

export type InputSize = 'sm' | 'md' | 'lg';

export interface InputProps extends Omit<JSX.HTMLAttributes<HTMLInputElement>, 'size' | 'style'> {
  value?: string | number;
  onInput?: (e: JSX.TargetedEvent<HTMLInputElement>) => void;
  onChange?: (e: JSX.TargetedEvent<HTMLInputElement>) => void;
  placeholder?: string;
  type?: string;
  disabled?: boolean;
  required?: boolean;
  autoFocus?: boolean;
  name?: string;
  autoComplete?: string;
  maxLength?: number;
  minLength?: number;
  min?: number | string;
  max?: number | string;
  step?: number | string;
  error?: boolean | string;
  fullWidth?: boolean;
  size?: InputSize;
  leftIcon?: ComponentChildren;
  rightIcon?: ComponentChildren;
  className?: string;
  containerClassName?: string;
  style?: JSX.CSSProperties;
}

/**
 * Standard Themed Text / Number Input Component
 *
 * Implements tokenized input styling with focus ring, error highlights,
 * and optional prefix/suffix icons.
 */
export function Input({
  value,
  onInput,
  onChange,
  placeholder,
  type = 'text',
  disabled = false,
  error = false,
  fullWidth = true,
  size = 'md',
  leftIcon,
  rightIcon,
  className,
  containerClassName,
  style,
  ...restProps
}: InputProps) {
  const sizeStyles: Record<InputSize, JSX.CSSProperties> = {
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

  const inputStyle: JSX.CSSProperties = {
    ...sizeStyles[size],
    ...(fullWidth ? { width: '100%' } : {}),
    ...(leftIcon ? { paddingLeft: '32px' } : {}),
    ...(rightIcon ? { paddingRight: '32px' } : {}),
    ...errorStyles,
    ...(style || {}),
  };

  const inputElement = (
    <input
      type={type}
      value={value}
      onInput={onInput}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      className={`tiresias-input ${hasError ? 'tiresias-input-error' : ''} ${className || ''}`}
      style={inputStyle}
      aria-invalid={hasError}
      {...restProps}
    />
  );

  if (!leftIcon && !rightIcon) {
    return inputElement;
  }

  return (
    <div
      className={`tiresias-input-wrapper ${containerClassName || ''}`}
      style={{
        position: 'relative',
        display: fullWidth ? 'flex' : 'inline-flex',
        alignItems: 'center',
        width: fullWidth ? '100%' : 'auto',
      }}
    >
      {leftIcon && (
        <span
          style={{
            position: 'absolute',
            left: '10px',
            display: 'inline-flex',
            alignItems: 'center',
            color: 'var(--tiresias-text-muted)',
            pointerEvents: 'none',
            zIndex: 1,
          }}
        >
          {leftIcon}
        </span>
      )}
      {inputElement}
      {rightIcon && (
        <span
          style={{
            position: 'absolute',
            right: '10px',
            display: 'inline-flex',
            alignItems: 'center',
            color: 'var(--tiresias-text-muted)',
            pointerEvents: 'none',
            zIndex: 1,
          }}
        >
          {rightIcon}
        </span>
      )}
    </div>
  );
}
