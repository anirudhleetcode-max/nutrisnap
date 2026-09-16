export function Mark({ size = 26 }: { size?: number }) {
  // a plate seen from above, with a leaf — drawn, not an emoji
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="14.5" fill="#c8442b" />
      <circle cx="16" cy="16" r="9.5" fill="none" stroke="#faf7f1" strokeWidth="1.6" />
      <path d="M16.2 10.6c3.4.2 5.4 2.4 5.2 5.9-3.6.1-5.6-2-5.2-5.9Z" fill="#faf7f1" />
      <path d="M13 20.8c.6-2.6 1.9-4.4 4-5.6" stroke="#faf7f1" strokeWidth="1.3" fill="none" strokeLinecap="round" />
    </svg>
  );
}

export function Wordmark() {
  return (
    <span className="wordmark">
      <Mark />
      <span>NutriSnap</span>
    </span>
  );
}
