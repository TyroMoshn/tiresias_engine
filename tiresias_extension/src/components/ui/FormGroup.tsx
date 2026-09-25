import { h, ComponentChildren, JSX } from 'preact';

export interface FormGroupProps {
  label?: ComponentChildren;
  htmlFor?: string;
  error?: ComponentChildren;
  hint?: ComponentChildren;
  required?: boolean;
  children?: ComponentChildren;
  className?: string;
  style?: JSX.CSSProperties | string;
}

/**
 * Standard Form Field Group Container
 *
 * Wraps form controls with accessible labels, subtle hints, and error captions.
 */
export function FormGroup({
  label,
  htmlFor,
  error,
  hint,
  required = false,
  children,
  className,
  style,
}: FormGroupProps) {
  const customStyles: JSX.CSSProperties = typeof style === 'object' ? style : {};

  return (
    <div className={`tiresias-form-group ${className || ''}`} style={customStyles}>
      {label && (
        <label className="tiresias-form-label" htmlFor={htmlFor}>
          {label}
          {required && (
            <span
              style={{
                color: 'var(--tiresias-danger-light)',
                marginLeft: '4px',
                fontWeight: 'bold',
              }}
              aria-hidden="true"
            >
              *
            </span>
          )}
        </label>
      )}

      {children}

      {hint && !error && (
        <div className="tiresias-form-hint">
          {hint}
        </div>
      )}

      {error && (
        <div className="tiresias-form-error" role="alert">
          <span style={{ flexShrink: 0 }}>⚠️</span>
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
