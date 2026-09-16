import { Search } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { api } from "../lib/api";
import { kcal } from "../lib/nutrition";
import type { Food } from "../lib/types";

export default function FoodSearch({ onPick, placeholder = "Search 180+ foods — dal, dosa, pizza…", autoFocus }: {
  onPick: (f: Food) => void; placeholder?: string; autoFocus?: boolean;
}) {
  const [q, setQ] = useState("");
  const [items, setItems] = useState<Food[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [loading, setLoading] = useState(false);
  const listId = useId();
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const term = q.trim();
    if (!term) { setItems([]); return; }
    setLoading(true);
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      api<{ items: Food[] }>(`/api/foods?limit=8&q=${encodeURIComponent(term)}`, { signal: ctrl.signal })
        .then((r) => { setItems(r.items); setActive(0); })
        .catch(() => undefined)
        .finally(() => setLoading(false));
    }, 160);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [q]);

  useEffect(() => {
    const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const pick = (f: Food) => { onPick(f); setQ(""); setOpen(false); };

  return (
    <div className="search" ref={box}>
      <Search className="search-icon" aria-hidden="true" />
      <input
        className="input search-input" type="search" value={q} placeholder={placeholder} autoFocus={autoFocus}
        role="combobox" aria-expanded={open && items.length > 0} aria-controls={listId} aria-autocomplete="list"
        aria-label="Search foods"
        onChange={(e) => { setQ(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(items.length - 1, a + 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
          else if (e.key === "Enter" && items[active]) { e.preventDefault(); pick(items[active]); }
          else if (e.key === "Escape") setOpen(false);
        }}
      />
      {open && q.trim() && (
        <ul className="search-list" id={listId} role="listbox">
          {items.map((f, i) => (
            <li key={f.id} role="option" aria-selected={i === active}
              className={i === active ? "active" : ""} onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => { e.preventDefault(); pick(f); }}>
              <span>{f.name}</span>
              <span className="faint xs num">{kcal(f.per100g.kcal)} kcal / 100 g</span>
            </li>
          ))}
          {!loading && items.length === 0 && <li className="search-empty">No match for “{q}”. Try a simpler name.</li>}
        </ul>
      )}
    </div>
  );
}
