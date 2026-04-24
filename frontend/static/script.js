document.addEventListener("DOMContentLoaded", () => {
  // Predict form loading state
  const form = document.getElementById("predictForm");
  if (form) {
    form.addEventListener("submit", (e) => {
      const src = document.getElementById("source")?.value;
      const dst = document.getElementById("destination")?.value;
      if (src && dst && src === dst) {
        e.preventDefault();
        showError("Source and destination cannot be the same.");
        return;
      }
      const btn = document.getElementById("submitBtn");
      if (btn) {
        btn.querySelector("span").textContent = "Predicting...";
        btn.querySelector(".btn-loader")?.classList.remove("hidden");
        btn.querySelector(".fa-search")?.classList.add("hidden");
        btn.disabled = true;
      }
    });
  }

  // Animate metric bars on dashboard
  document.querySelectorAll(".metric-bar, .confidence-bar").forEach((bar) => {
    const target = bar.style.width;
    bar.style.width = "0%";
    setTimeout(() => { bar.style.width = target; }, 100);
  });

  // Animate stat values
  document.querySelectorAll(".stat-value").forEach((el) => {
    const val = parseInt(el.textContent);
    if (!isNaN(val)) {
      let current = 0;
      const step = Math.ceil(val / 30);
      const timer = setInterval(() => {
        current = Math.min(current + step, val);
        el.textContent = current;
        if (current >= val) clearInterval(timer);
      }, 30);
    }
  });
});

function showError(msg) {
  let alert = document.querySelector(".alert-error");
  if (!alert) {
    alert = document.createElement("div");
    alert.className = "alert alert-error";
    alert.innerHTML = `<i class="fas fa-exclamation-circle"></i> <span></span>`;
    document.querySelector(".container")?.prepend(alert);
  }
  alert.querySelector("span").textContent = msg;
  alert.scrollIntoView({ behavior: "smooth", block: "center" });
}
