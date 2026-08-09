// Theme toggle: dark/light with persisted preference (inspired by the
// gpt2api-web useDark store). The pre-paint inline script in each page's
// <head> has already applied the saved/system theme; this wires the button.
(function () {
  const KEY = "ara.theme";
  const root = document.documentElement;

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("themeToggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const next = root.classList.contains("dark") ? "light" : "dark";
      root.classList.toggle("dark", next === "dark");
      try {
        localStorage.setItem(KEY, next);
      } catch {
        // storage unavailable (private mode); theme still toggles, just not persisted
      }
    });
  });
})();
