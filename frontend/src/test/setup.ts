import '@testing-library/jest-dom'

// jsdom 25+ doesn't fully implement localStorage when no URL is given.
// Even with a URL the Storage methods (clear/removeItem) can be missing in some
// vitest worker configurations. Provide a complete in-memory implementation.
class InMemoryStorage implements Storage {
  private store: Map<string, string> = new Map()

  get length(): number { return this.store.size }
  key(index: number): string | null { return [...this.store.keys()][index] ?? null }
  getItem(key: string): string | null { return this.store.get(key) ?? null }
  setItem(key: string, value: string): void { this.store.set(key, String(value)) }
  removeItem(key: string): void { this.store.delete(key) }
  clear(): void { this.store.clear() }
}

Object.defineProperty(globalThis, 'localStorage', {
  value: new InMemoryStorage(),
  writable: true,
})
