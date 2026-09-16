import { BarChart3, BookOpen, Camera, Cpu, LogOut, UserRound } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { Wordmark } from "./Logo";

const NAV = [
  { to: "/", label: "Snap", icon: Camera, end: true },
  { to: "/diary", label: "Diary", icon: BookOpen },
  { to: "/trends", label: "Trends", icon: BarChart3 },
  { to: "/profile", label: "Profile", icon: UserRound },
  { to: "/model", label: "Model", icon: Cpu },
];

export const DEMO_EMAIL = "demo@nutrisnap.app";

export default function Shell() {
  const { user, logout } = useAuth();
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar-top"><Wordmark /></div>
        <nav className="side-nav" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className="side-link">
              <Icon aria-hidden="true" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="who">
            <span className="avatar" aria-hidden="true">{user?.name.slice(0, 1).toUpperCase()}</span>
            <span className="who-text">
              <strong>{user?.name}</strong>
              <span className="faint xs">{user?.email}</span>
            </span>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={logout}><LogOut aria-hidden="true" /> Sign out</button>
        </div>
      </aside>
      <header className="mobile-top">
        <Wordmark />
      </header>
      <main className="main" id="main">
        {user?.email === DEMO_EMAIL && (
          <p className="demo-banner" role="note">
            Demo account — the meals and trends here are generated sample data, not a real person’s log.
          </p>
        )}
        <Outlet />
      </main>
      <nav className="tabbar" aria-label="Main">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className="tab">
            <Icon aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
