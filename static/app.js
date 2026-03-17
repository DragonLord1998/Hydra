/* ================================================================
   Hydra — Character Developer (Frontend)
   ================================================================ */

(function () {
  "use strict";

  // --- DOM ---
  const promptInput  = document.getElementById("promptInput");
  const modeToggle   = document.getElementById("modeToggle");
  const imageDisplay = document.getElementById("imageDisplay");
  const placeholder  = document.getElementById("placeholder");
  const loraFile     = document.getElementById("loraFile");
  const loraBtn      = document.getElementById("loraBtn");
  const triggerWord  = document.getElementById("triggerWord");
  const loraStatus   = document.getElementById("loraStatus");

  const modelBtns    = document.querySelectorAll(".model-btn");

  // --- State ---
  let mode = "generate";        // "generate" | "edit"
  let zimageVariant = "deturbo"; // "deturbo" | "base"
  let currentImageUrl = null;
  let busy = false;

  // ---------------------------------------------------------------
  // Model selector (De-Turbo / Base)
  // ---------------------------------------------------------------

  modelBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      if (busy) return;
      zimageVariant = btn.dataset.model;
      modelBtns.forEach(b => b.classList.toggle("active", b === btn));
    });
  });

  // ---------------------------------------------------------------
  // Mode toggle
  // ---------------------------------------------------------------

  modeToggle.addEventListener("click", () => {
    if (busy) return;
    mode = mode === "generate" ? "edit" : "generate";
    modeToggle.classList.toggle("edit", mode === "edit");

    if (mode === "generate") {
      modeToggle.title = "Generate mode (click to switch)";
      promptInput.placeholder = "describe your character...";
    } else {
      modeToggle.title = "Edit mode (click to switch)";
      promptInput.placeholder = currentImageUrl
        ? "describe your edit..."
        : "generate an image first...";
    }
  });

  // ---------------------------------------------------------------
  // LoRA upload
  // ---------------------------------------------------------------

  loraFile.addEventListener("change", async () => {
    const file = loraFile.files[0];
    if (!file) return;

    const trigger = triggerWord.value.trim() || "chrx";
    const fd = new FormData();
    fd.append("lora", file);
    fd.append("trigger_word", trigger);

    try {
      const resp = await fetch("/api/upload-lora", { method: "POST", body: fd });
      const data = await resp.json();
      if (resp.ok) {
        loraBtn.classList.add("loaded");
        loraStatus.textContent = data.name;
        loraStatus.classList.add("visible");
      } else {
        showToast(data.error || "Upload failed");
      }
    } catch (err) {
      showToast("Upload failed: " + err.message);
    }

    // Reset file input so re-uploading the same file triggers change
    loraFile.value = "";
  });

  // ---------------------------------------------------------------
  // Prompt submission
  // ---------------------------------------------------------------

  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });

  async function submit() {
    const prompt = promptInput.value.trim();
    if (!prompt || busy) return;

    if (mode === "edit" && !currentImageUrl) {
      showToast("Generate an image first");
      return;
    }

    busy = true;
    promptInput.disabled = true;
    modeToggle.classList.add("loading");
    imageDisplay.classList.add("loading");
    if (placeholder) placeholder.textContent = mode === "generate" ? "Generating..." : "Editing...";

    const endpoint = mode === "generate" ? "/api/generate" : "/api/edit";
    const payload = mode === "generate"
      ? { prompt, model: zimageVariant }
      : { prompt };

    try {
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();

      if (resp.ok && data.image_url) {
        showImage(data.image_url);
      } else {
        showToast(data.error || "Request failed");
        resetPlaceholder();
      }
    } catch (err) {
      showToast("Request failed: " + err.message);
      resetPlaceholder();
    } finally {
      busy = false;
      promptInput.disabled = false;
      modeToggle.classList.remove("loading");
      imageDisplay.classList.remove("loading");
      promptInput.focus();
    }
  }

  // ---------------------------------------------------------------
  // Image display
  // ---------------------------------------------------------------

  function resetPlaceholder() {
    if (!placeholder) return;
    // Only restore if no image has been shown yet
    if (!currentImageUrl) {
      placeholder.style.display = "";
      placeholder.textContent = "Generate an image to begin";
    }
  }

  function showImage(url) {
    currentImageUrl = url;

    if (placeholder) placeholder.style.display = "none";

    let img = imageDisplay.querySelector("img");
    if (!img) {
      img = document.createElement("img");
      imageDisplay.appendChild(img);
    }
    img.src = url + "?t=" + Date.now();
    img.alt = "Generated image";

    // Update edit mode placeholder if needed
    if (mode === "edit" || mode === "generate") {
      promptInput.placeholder = mode === "generate"
        ? "describe your character..."
        : "describe your edit...";
    }
  }

  // ---------------------------------------------------------------
  // Toast notifications
  // ---------------------------------------------------------------

  let toastEl = null;
  let toastTimer = null;

  function showToast(message) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.className = "toast";
      document.body.appendChild(toastEl);
    }

    toastEl.textContent = message;
    toastEl.classList.add("visible");

    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastEl.classList.remove("visible");
    }, 3500);
  }

})();
