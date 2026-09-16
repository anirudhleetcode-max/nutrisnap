import { AlertTriangle } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { Wordmark } from "../components/Logo";
import { ApiError } from "../lib/api";
import { useAuth } from "../lib/auth";

export default function Auth({ mode }: { mode: "login" | "register" }) {
  const { login, register } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      // PublicOnly (App.tsx) redirects once the user is set: to Snap, or to Profile for new accounts
      if (mode === "login") await login(email, password);
      else await register(name, email, password);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't reach the server.");
    } finally { setBusy(false); }
  };

  return (
    <div className="auth">
      <div className="auth-side" aria-hidden="true">
        <div className="auth-side-inner">
          <Wordmark />
          <div className="plate-note">
            <p className="note-line"><span>2 × roti</span><span className="num">224</span></p>
            <p className="note-line"><span>1 katori dal tadka</span><span className="num">158</span></p>
            <p className="note-line"><span>1 katori bhindi masala</span><span className="num">165</span></p>
            <p className="note-line total"><span>Lunch</span><span className="num">547 kcal</span></p>
          </div>
          <p className="auth-quote">Roti, dal, dosa or pizza — photograph it, confirm it, done.</p>
        </div>
      </div>
      <main className="auth-main">
        <form className="auth-form" onSubmit={submit}>
          <div className="auth-brand"><Wordmark /></div>
          <h1>{mode === "login" ? "Sign in" : "Create your account"}</h1>
          <p className="muted">Log meals from a photo and see calories and macros against your own target.</p>
          {mode === "register" && (
            <label className="field"><span>Name</span>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} required maxLength={60} autoComplete="name" name="name" />
            </label>
          )}
          <label className="field"><span>Email</span>
            <input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" name="email" />
          </label>
          <label className="field"><span>Password</span>
            <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={6}
              autoComplete={mode === "login" ? "current-password" : "new-password"} name="password" />
          </label>
          {error && <div className="alert alert-error" role="alert"><AlertTriangle aria-hidden="true" />{error}</div>}
          <button className="btn btn-primary btn-lg btn-block" disabled={busy} type="submit">
            {busy ? <span className="spinner" /> : mode === "login" ? "Sign in" : "Create account"}
          </button>
          {mode === "login" ? (
            <p className="small muted">
              New here? <Link to="/register">Create an account</Link> ·{" "}
              <button type="button" className="linklike" onClick={() => { setEmail("demo@nutrisnap.app"); setPassword("demo1234"); }}>
                Use demo account
              </button>
            </p>
          ) : (
            <p className="small muted">Already have an account? <Link to="/login">Sign in</Link></p>
          )}
        </form>
      </main>
    </div>
  );
}
