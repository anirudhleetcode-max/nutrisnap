import { useEffect, useRef } from "react";

type Props = {
  open: boolean;
  title: string;
  body: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
  busy?: boolean;
};

export default function ConfirmDialog({ open, title, body, confirmLabel, onConfirm, onCancel, busy }: Props) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    cancelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onCancel(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);
  if (!open) return null;
  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel(); }}>
      <div className="modal confirm" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" aria-describedby="confirm-body">
        <h2 id="confirm-title">{title}</h2>
        <p id="confirm-body" className="muted">{body}</p>
        <div className="confirm-actions">
          <button ref={cancelRef} className="btn" onClick={onCancel}>Cancel</button>
          <button className="btn btn-primary btn-destructive" onClick={onConfirm} disabled={busy}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}
