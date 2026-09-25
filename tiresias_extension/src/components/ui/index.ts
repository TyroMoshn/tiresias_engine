/**
 * Tiresias UI Component Catalog
 *
 * Single Source of Truth for base reusable UI primitives, adhering to
 * Tiresias Design System tokens and cross-platform Chrome MV3 standards.
 */

// Toast notification system
export { Toast, ToastContainer } from './Toast';
export type { ToastProps, ToastContainerProps } from './Toast';
export { toast } from '../../lib/toast';
export type {
  ToastOptions,
  ToastType,
  ToastEventDetail,
  ToastDismissDetail,
} from '../../lib/toast';

// Modal dialogs
export { Modal } from './Modal';
export type { ModalProps, ModalSize } from './Modal';

// Drawer panels
export { Drawer } from './Drawer';
export type { DrawerProps } from './Drawer';

// Buttons
export { Button } from './Button';
export type { ButtonProps, ButtonVariant, ButtonSize } from './Button';
export { IconButton } from './IconButton';
export type {
  IconButtonProps,
  IconButtonSize,
  IconButtonVariant,
  IconButtonActiveVariant,
} from './IconButton';

// Form controls
export { Input } from './Input';
export type { InputProps, InputSize } from './Input';
export { Select } from './Select';
export type { SelectProps, SelectOption, SelectSize } from './Select';
export { FormGroup } from './FormGroup';
export type { FormGroupProps } from './FormGroup';

// Toggles & Segmented controls
export { Toggle } from './Toggle';
export type { ToggleProps, ToggleType } from './Toggle';
export { SegmentedControl } from './SegmentedControl';
export type { SegmentedControlProps, SegmentedOption } from './SegmentedControl';

// Badges & Status indicators
export { Badge, StatusDot } from './Badge';
export type {
  BadgeProps,
  BadgeVariant,
  BadgeSize,
  StatusDotProps,
  BooruRating,
  TagCategory,
  NetworkStatus,
} from './Badge';
