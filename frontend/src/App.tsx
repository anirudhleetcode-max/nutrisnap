import type { ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Shell from "./components/Shell";
import { useAuth } from "./lib/auth";
import Auth from "./pages/Auth";
import Diary from "./pages/Diary";
import Profile from "./pages/Profile";
import Snap from "./pages/Snap";
import Trends from "./pages/Trends";

function Protected({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="boot"><span className="spinner" /></div>;
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function PublicOnly({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="boot"><span className="spinner" /></div>;
  if (user) return <Navigate to={user.has_profile ? "/" : "/profile"} replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<PublicOnly><Auth mode="login" /></PublicOnly>} />
      <Route path="/register" element={<PublicOnly><Auth mode="register" /></PublicOnly>} />
      <Route element={<Protected><Shell /></Protected>}>
        <Route index element={<Snap />} />
        <Route path="diary" element={<Diary />} />
        <Route path="trends" element={<Trends />} />
        <Route path="profile" element={<Profile />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
