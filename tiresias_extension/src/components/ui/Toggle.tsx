import { h, ComponentChildren, JSX } from 'preact';

export type ToggleType = 'switch' | 'pill';

export interface ToggleProps {
  type?: ToggleType;
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: ComponentChildren;
  onText?: string;
  offText?: string;
  disabled?: boolean;
  className?: string;
  title?: string;
  ariaLabel?: string;
  id?: string;
}

/**
 * Standard Toggle Component
 *
 * Supports both switch (iOS style toggle) and pill (two-state sliding badge like Crop / Full).
 */
export function Toggle({
  type = 'switch',
  checked,
  onChange,
  label,
  onText = 'On',
  offText = 'Off',
  disabled = false,
  className,
  title,
  ariaLabel,
  id,
}: ToggleProps) {
  const handleChange = (e: JSX.TargetedEvent<HTMLInputElement>) => {
    if (disabled) return;
    onChange(e.currentTarget.checked);
  };

  if (type === 'pill') {
    return (
      <label
        className={`tiresias-toggle-pill ${className || ''}`}
        title={title}
        style={{ opacity: disabled ? 0.5 : 1, cursor: disabled ? 'not-allowed' : 'pointer' }}
      >
        <input
          id={id}
          type="checkbox"
          checked={checked}
          onChange={handleChange}
          disabled={disabled}
          aria-label={ariaLabel || (typeof label === 'string' ? label : undefined)}
        />
        <span className="pill-option pill-off">{offText}</span>
        <span className="pill-option pill-on">{onText}</span>
      </label>
    );
  }

  // Switch Toggle
  return (
    <label
      className={`tiresias-toggle-switch ${checked ? 'checked' : ''} ${className || ''}`}
      title={title}
      style={{ opacity: disabled ? 0.5 : 1, cursor: disabled ? 'not-allowed' : 'pointer' }}
    >
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={handleChange}
        disabled={disabled}
        aria-label={ariaLabel || (typeof label === 'string' ? label : undefined)}
      />
      <span className="tiresias-toggle-switch-track">
        <span className="tiresias-toggle-switch-thumb" />
      </span>
      {label && (
        <span
          className="tiresias-toggle-label"
          style={{
            fontSize: '13px',
            color: 'var(--tiresias-text-primary)',
            userSelect: 'none',
          }}
        >
          {label}
        </span>
      )}
    </label>
  );
}
