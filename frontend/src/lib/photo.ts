import { useEffect, useState } from "react";
import { tokenStore } from "./api";

const cache = new Map<string, string>();

/** <img> can't send a bearer token, so thumbnails are fetched as blobs. */
export function usePhoto(photoId: string | null | undefined): string | null {
  const [url, setUrl] = useState<string | null>(photoId ? cache.get(photoId) ?? null : null);
  useEffect(() => {
    if (!photoId) { setUrl(null); return; }
    const hit = cache.get(photoId);
    if (hit) { setUrl(hit); return; }
    let alive = true;
    fetch(`/api/photos/${photoId}`, { headers: { Authorization: `Bearer ${tokenStore.get() ?? ""}` } })
      .then((r) => (r.ok ? r.blob() : null))
      .then((b) => {
        if (!b) return;
        const u = URL.createObjectURL(b);
        cache.set(photoId, u);
        if (alive) setUrl(u);
      })
      .catch(() => undefined);
    return () => { alive = false; };
  }, [photoId]);
  return url;
}
