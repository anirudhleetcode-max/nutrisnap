import { Camera, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export default function Webcam({ onCapture, onClose }: { onCapture: (f: File) => void; onClose: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let stream: MediaStream | null = null;
    navigator.mediaDevices?.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false })
      .then((s) => {
        stream = s;
        if (video.current) { video.current.srcObject = s; video.current.play().then(() => setReady(true)).catch(() => undefined); }
      })
      .catch(() => setError("Couldn't open the camera. Check the browser's camera permission, or upload a photo instead."));
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => { stream?.getTracks().forEach((t) => t.stop()); window.removeEventListener("keydown", onKey); };
  }, [onClose]);

  const capture = () => {
    const v = video.current;
    if (!v) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d")?.drawImage(v, 0, 0);
    c.toBlob((b) => { if (b) onCapture(new File([b], "webcam.jpg", { type: "image/jpeg" })); }, "image/jpeg", 0.9);
  };

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Webcam">
      <div className="modal webcam">
        <div className="modal-head">
          <h2>Point at your plate</h2>
          <button className="btn btn-ghost icon-btn" onClick={onClose} aria-label="Close"><X aria-hidden="true" /></button>
        </div>
        {error ? <div className="alert alert-error" role="alert">{error}</div> : <video ref={video} playsInline muted />}
        <button className="btn btn-primary btn-lg btn-block" onClick={capture} disabled={!ready}>
          <Camera aria-hidden="true" /> Capture
        </button>
      </div>
    </div>
  );
}
