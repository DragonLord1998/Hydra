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
  const settingsBtn  = document.getElementById("settingsBtn");
  const settingsPanel = document.getElementById("settingsPanel");
  const resolutionSelect = document.getElementById("resolutionSelect");
  const resolutionRow = document.getElementById("resolutionRow");
  const stepsRange   = document.getElementById("stepsRange");
  const stepsValue   = document.getElementById("stepsValue");
  const uploadImgBtn = document.getElementById("uploadImgBtn");
  const imageFile    = document.getElementById("imageFile");

  // --- State ---
  let mode = "generate";        // "generate" | "edit"
  let genVariant = "deturbo"; // "deturbo" | "base" | "srpo"
  let currentImageUrl = null;
  let busy = false;

  // Restore state from server on page load
  fetch("/api/status").then(r => r.json()).then(data => {
    if (data.lora) {
      loraBtn.classList.add("loaded");
      loraStatus.textContent = data.lora.name;
      loraStatus.classList.add("visible");
      if (data.lora.trigger) triggerWord.value = data.lora.trigger;
    }
    if (data.gen_variant) {
      genVariant = data.gen_variant;
      modelBtns.forEach(b => b.classList.toggle("active", b.dataset.model === genVariant));
    }
    if (data.has_image && data.mode === "edit") {
      mode = "edit";
      modeToggle.classList.add("edit");
      modeToggle.title = "Edit mode (click to switch)";
      promptInput.placeholder = "describe your edit...";
      uploadImgBtn.style.display = "";
      resolutionRow.style.display = "none";
      syncStepsForMode("edit");
    }
  }).catch(() => {});

  // ---------------------------------------------------------------
  // Model selector (De-Turbo / Base / SRPO)
  // ---------------------------------------------------------------

  modelBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      if (busy) return;
      genVariant = btn.dataset.model;
      modelBtns.forEach(b => b.classList.toggle("active", b === btn));
      if (mode === "generate") syncDefaultSteps(genVariant);
    });
  });

  // ---------------------------------------------------------------
  // Settings panel
  // ---------------------------------------------------------------

  settingsBtn.addEventListener("click", () => {
    const open = settingsPanel.style.display !== "none";
    settingsPanel.style.display = open ? "none" : "";
    settingsBtn.classList.toggle("active", !open);
  });

  stepsRange.addEventListener("input", () => {
    stepsValue.textContent = stepsRange.value;
  });

  // Sync default steps when switching model variant or mode
  function syncDefaultSteps(variant) {
    var defaults = { deturbo: 25, base: 50, srpo: 50 };
    stepsRange.value = defaults[variant] || 25;
    stepsValue.textContent = stepsRange.value;
  }

  function syncStepsForMode(m) {
    if (m === "edit") {
      stepsRange.value = 20;
    } else {
      var defaults = { deturbo: 25, base: 50, srpo: 50 };
      stepsRange.value = defaults[genVariant] || 25;
    }
    stepsValue.textContent = stepsRange.value;
  }

  // ---------------------------------------------------------------
  // Mode toggle
  // ---------------------------------------------------------------

  modeToggle.addEventListener("click", () => {
    if (busy) return;
    mode = mode === "generate" ? "edit" : "generate";
    modeToggle.classList.toggle("edit", mode === "edit");
    uploadImgBtn.style.display = mode === "edit" ? "" : "none";
    resolutionRow.style.display = mode === "edit" ? "none" : "";
    syncStepsForMode(mode);

    if (mode === "generate") {
      modeToggle.title = "Generate mode (click to switch)";
      promptInput.placeholder = "describe your character...";
    } else {
      modeToggle.title = "Edit mode (click to switch)";
      promptInput.placeholder = currentImageUrl
        ? "describe your edit..."
        : "upload or generate an image first...";
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
  // Image upload (for edit mode)
  // ---------------------------------------------------------------

  imageFile.addEventListener("change", async () => {
    const file = imageFile.files[0];
    if (!file) return;

    const fd = new FormData();
    fd.append("image", file);

    try {
      const resp = await fetch("/api/upload-image", { method: "POST", body: fd });
      const data = await resp.json();
      if (resp.ok && data.image_url) {
        showImage(data.image_url);
        promptInput.placeholder = "describe your edit...";
      } else {
        showToast(data.error || "Upload failed");
      }
    } catch (err) {
      showToast("Upload failed: " + err.message);
    }

    imageFile.value = "";
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
      showToast("Upload or generate an image first");
      return;
    }

    busy = true;
    promptInput.disabled = true;
    modeToggle.classList.add("loading");
    imageDisplay.classList.add("loading");
    if (placeholder) placeholder.textContent = mode === "generate" ? "Generating..." : "Editing...";

    const endpoint = mode === "generate" ? "/api/generate" : "/api/edit";
    const [w, h] = resolutionSelect.value.split("x").map(Number);
    const steps = parseInt(stepsRange.value, 10);
    const payload = mode === "generate"
      ? { prompt, model: genVariant, width: w, height: h, steps: steps }
      : { prompt, steps: steps };

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
  // SSE connection for live previews & model status
  // ---------------------------------------------------------------

  const stepCounter  = document.getElementById("stepCounter");
  const stepText     = document.getElementById("stepText");
  const loadOverlay  = document.getElementById("loadingOverlay");
  const loadText     = document.getElementById("loadingText");

  var evtSource = new EventSource("/api/stream");

  function attachSSEListeners(src) {
    src.addEventListener("preview", function (e) {
      var data = JSON.parse(e.data);
      if (!busy) return; // ignore stale previews
      if (placeholder) placeholder.style.display = "none";

      var img = imageDisplay.querySelector("img");
      if (!img) {
        img = document.createElement("img");
        img.alt = "Preview";
        imageDisplay.appendChild(img);
      }
      img.src = data.image;
      img.classList.add("preview-img");

      if (stepCounter && stepText) {
        stepCounter.style.display = "";
        stepText.textContent = data.step + " / " + data.total;
      }
    });

    src.addEventListener("model_status", function (e) {
      var data = JSON.parse(e.data);
      if (data.action === "loading") {
        if (loadOverlay && loadText) {
          loadText.textContent = "Loading " + data.name + "...";
          loadOverlay.style.display = "";
        }
      } else if (data.action === "ready") {
        if (loadOverlay) loadOverlay.style.display = "none";
      }
    });

    src.addEventListener("error", function (e) {
      var data;
      try { data = JSON.parse(e.data); } catch (_) { return; }
      if (data && data.message) showToast(data.message);
    });

    // Resync state on reconnection
    src.onopen = function () {
      if (loadOverlay) loadOverlay.style.display = "none";
      fetch("/api/status").then(function (r) { return r.json(); }).then(function (data) {
        if (data.lora) {
          loraBtn.classList.add("loaded");
          loraStatus.textContent = data.lora.name;
          loraStatus.classList.add("visible");
        }
      }).catch(function () {});
    };
  }

  attachSSEListeners(evtSource);

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
    if (stepCounter) stepCounter.style.display = "none";

    var img = imageDisplay.querySelector("img");
    if (!img) {
      img = document.createElement("img");
      imageDisplay.appendChild(img);
    }
    img.src = url + "?t=" + Date.now();
    img.alt = "Generated image";
    img.classList.remove("preview-img");

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
