import React, { createContext, useContext, useEffect, useState } from "react";

export type ThemeMode = "light" | "dark" | "system";

type ThemeContextType = {
  theme: ThemeMode;
  resolvedTheme: "light" | "dark";
  setTheme: (theme: ThemeMode) => void;
};

const THEME_STORAGE_KEY = "route-resilience-theme";

const ThemeContext = createContext<ThemeContextType>({
  theme: "system",
  resolvedTheme: "light",
  setTheme: () => {},
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "system";
    try {
      const stored = window.localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode | null;
      return stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
    } catch {
      return "system";
    }
  });

  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">(() => {
    if (typeof window === "undefined") return "light";
    try {
      const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
      if (stored === "dark") return "dark";
      if (stored === "light") return "light";
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    } catch {
      return "light";
    }
  });

  useEffect(() => {
    const root = document.documentElement;
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const applyTheme = () => {
      const isDark = theme === "dark" || (theme === "system" && mediaQuery.matches);
      const active: "light" | "dark" = isDark ? "dark" : "light";
      setResolvedTheme(active);

      if (isDark) {
        root.classList.add("dark");
      } else {
        root.classList.remove("dark");
      }
      root.style.colorScheme = active;
    };

    applyTheme();

    const listener = () => {
      if (theme === "system") {
        applyTheme();
      }
    };

    mediaQuery.addEventListener("change", listener);
    return () => mediaQuery.removeEventListener("change", listener);
  }, [theme]);

  const setTheme = (newTheme: ThemeMode) => {
    setThemeState(newTheme);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, newTheme);
    } catch {
      // Ignore localStorage write failures
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
