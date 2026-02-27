/**
 * main.js — SolarGraph AI
 * Shared utilities: auto-resize textarea.
 */
document.addEventListener("DOMContentLoaded", () => {
  const ta = document.getElementById("queryInput");
  if (!ta) return;
  ta.addEventListener("input", () => {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  });
});
