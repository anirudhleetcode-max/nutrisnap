import { usePhoto } from "../lib/photo";

const TONES = ["#f3e2d6", "#e8ecd9", "#f6ecd0", "#e3e8ee", "#dfeee3"];

export default function FoodThumb({ photoId, name, size = 44 }: { photoId: string | null; name: string; size?: number }) {
  const url = usePhoto(photoId);
  if (url) return <img className="thumb" src={url} alt="" width={size} height={size} style={{ width: size, height: size }} />;
  const tone = TONES[[...name].reduce((a, c) => a + c.charCodeAt(0), 0) % TONES.length];
  return (
    <span className="thumb thumb-letter" style={{ width: size, height: size, background: tone }} aria-hidden="true">
      {name.slice(0, 1)}
    </span>
  );
}
