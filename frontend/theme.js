// Shared Light / Dark mode toggle with smooth transition & localStorage persistence
(function () {
  const saved = localStorage.getItem("adaptive_rag_theme");
  const initial = saved === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", initial);

  function updateButtons(theme) {
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.textContent = theme === "dark" ? "Light" : "Dark";
      btn.title = theme === "dark" ? "Switch to Light mode" : "Switch to Dark mode";
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    updateButtons(document.documentElement.getAttribute("data-theme") || "light");

    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") || "light";
        const next = current === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("adaptive_rag_theme", next);
        updateButtons(next);
      });
    });
  });
})();
