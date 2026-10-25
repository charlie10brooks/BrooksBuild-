// Minimal ambient typings for React 19 / react-dom, used ONLY because @types/react could not be installed in the build
// environment (offline package registry). With `npm install` and @types/react present, delete this file.
declare module 'react' {
  export type ReactNode = any;
  export type Dispatch<A> = (a: A) => void;
  export type SetStateAction<S> = S | ((prev: S) => S);
  export type CSSProperties = { [k: string]: string | number | undefined };
  export type RefObject<T> = { current: T };
  export interface SyntheticEvent<T = any> {
    target: any; currentTarget: T; preventDefault(): void; stopPropagation(): void; nativeEvent: any;
    clientX: number; clientY: number; button: number; buttons: number; shiftKey: boolean; ctrlKey: boolean; metaKey: boolean; altKey: boolean;
    key: string; code: string; deltaY: number; deltaX: number; pointerId: number; detail: number; dataTransfer: any; pointerType: string;
    movementX: number; movementY: number; repeat: boolean;
  }
  export type FC<P = {}> = (props: P & { children?: ReactNode }) => any;
  export function useState<S>(init: S | (() => S)): [S, Dispatch<SetStateAction<S>>];
  export function useEffect(fn: () => void | (() => void), deps?: readonly unknown[]): void;
  export function useLayoutEffect(fn: () => void | (() => void), deps?: readonly unknown[]): void;
  export function useMemo<T>(fn: () => T, deps: readonly unknown[]): T;
  export function useCallback<T extends (...a: any[]) => any>(fn: T, deps: readonly unknown[]): T;
  export function useRef<T>(init: T): { current: T };
  export function useRef<T>(init: T | null): { current: T | null };
  export function useSyncExternalStore<T>(subscribe: (cb: () => void) => () => void, get: () => T): T;
  export function useContext<T>(c: any): T;
  export function createContext<T>(v: T): any;
  export function memo<T>(c: T): T;
  export function startTransition(fn: () => void): void;
  export const Fragment: any;
  export const StrictMode: any;
  const React: any;
  export default React;
}
declare module 'react/jsx-runtime' {
  export const Fragment: any;
  export function jsx(type: any, props: any, key?: any): any;
  export function jsxs(type: any, props: any, key?: any): any;
  import type { SyntheticEvent } from 'react';
  type H = (e: SyntheticEvent) => void;
  interface DomProps {
    [attr: string]: any;
    onClick?: H; onChange?: H; onInput?: H; onKeyDown?: H; onKeyUp?: H; onPointerDown?: H; onPointerMove?: H; onPointerUp?: H; onPointerLeave?: H;
    onDoubleClick?: H; onContextMenu?: H; onDragOver?: H; onDragLeave?: H; onDrop?: H; onScroll?: H; onMouseMove?: H; onMouseLeave?: H; onMouseDown?: H;
    onMouseUp?: H; onWheel?: H; onSubmit?: H; onFocus?: H; onBlur?: H; onLoad?: H; onError?: H;
  }
  export namespace JSX {
    type Element = any;
    interface ElementChildrenAttribute { children: {} }
    interface IntrinsicAttributes { key?: string | number }
    interface IntrinsicElements { [tag: string]: DomProps }
  }
}
declare module 'react-dom/client' {
  export function createRoot(el: Element): { render(node: any): void };
}
declare module 'react-dom' {
  export function createPortal(children: any, container: Element): any;
}
