/* ================================================================
   Hydra — Infinity Canvas
   ================================================================ */

(function () {
  "use strict";

  // --- DOM ---
  var canvasViewport = document.getElementById("canvasViewport");
  var canvasWorld    = document.getElementById("canvasWorld");
  var promptInput    = document.getElementById("promptInput");
  var modeToggle     = document.getElementById("modeToggle");
  var loraFile       = document.getElementById("loraFile");
  var loraBtn        = document.getElementById("loraBtn");
  var triggerWord    = document.getElementById("triggerWord");
  var loraStatus     = document.getElementById("loraStatus");
  var modelBtns      = document.querySelectorAll(".model-btn");
  var settingsBtn    = document.getElementById("settingsBtn");
  var settingsPanel  = document.getElementById("settingsPanel");
  var resolutionSelect = document.getElementById("resolutionSelect");
  var resolutionRow  = document.getElementById("resolutionRow");
  var stepsRange     = document.getElementById("stepsRange");
  var stepsValue     = document.getElementById("stepsValue");
  var uploadImgBtn   = document.getElementById("uploadImgBtn");
  var imageFile      = document.getElementById("imageFile");

  // --- State ---
  var mode = "generate";
  var genVariant = "srpo";
  var busy = false;
  var placementCounter = 0;

  // --- Canvas State ---
  var camera = { x: window.innerWidth / 2, y: window.innerHeight / 2, zoom: 1 };
  var ZOOM_MIN = 0.1, ZOOM_MAX = 5.0, ZOOM_SPEED = 0.002;
  var nodes = new Map();
  var selectedId = null;
  var generatingId = null;

  // ---------------------------------------------------------------
  // Camera
  // ---------------------------------------------------------------

  function applyCameraTransform() {
    canvasWorld.style.transform =
      "translate(" + camera.x + "px," + camera.y + "px) scale(" + camera.zoom + ")";
    requestCull();
  }

  function screenToWorld(sx, sy) {
    return { x: (sx - camera.x) / camera.zoom, y: (sy - camera.y) / camera.zoom };
  }

  function viewportCenter() {
    return screenToWorld(window.innerWidth / 2, window.innerHeight / 2);
  }

  applyCameraTransform();

  // ---------------------------------------------------------------
  // Pan & Zoom
  // ---------------------------------------------------------------

  var isPanning = false;
  var panStart = { x: 0, y: 0 };
  var spaceHeld = false;
  var dragging = null;

  canvasViewport.addEventListener("wheel", function (e) {
    e.preventDefault();
    var delta = -e.deltaY * ZOOM_SPEED;
    var oldZoom = camera.zoom;
    camera.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, camera.zoom * (1 + delta)));
    var ratio = camera.zoom / oldZoom;
    camera.x = e.clientX - (e.clientX - camera.x) * ratio;
    camera.y = e.clientY - (e.clientY - camera.y) * ratio;
    applyCameraTransform();
    saveCanvasState();
  }, { passive: false });

  canvasViewport.addEventListener("pointerdown", function (e) {
    if (e.button === 1 || (e.button === 0 && spaceHeld)) {
      isPanning = true;
      panStart.x = e.clientX - camera.x;
      panStart.y = e.clientY - camera.y;
      canvasViewport.classList.add("panning");
      canvasViewport.setPointerCapture(e.pointerId);
      e.preventDefault();
    } else if (e.button === 0 && !spaceHeld && e.target === canvasViewport) {
      // Click on empty canvas = deselect
      selectNode(null);
    }
  });

  canvasViewport.addEventListener("pointermove", function (e) {
    if (isPanning) {
      camera.x = e.clientX - panStart.x;
      camera.y = e.clientY - panStart.y;
      applyCameraTransform();
    }
    if (dragging) {
      var world = screenToWorld(e.clientX, e.clientY);
      var node = nodes.get(dragging.nodeId);
      if (!node) return;
      node.x = world.x - dragging.offsetX;
      node.y = world.y - dragging.offsetY;
      node.el.style.transform =
        "translate(" + (node.x - node.width / 2) + "px," + (node.y - node.height / 2) + "px)";
      dragging.moved = true;
    }
  });

  canvasViewport.addEventListener("pointerup", function (e) {
    if (isPanning) {
      isPanning = false;
      canvasViewport.classList.remove("panning");
      saveCanvasState();
    }
    if (dragging) {
      var node = nodes.get(dragging.nodeId);
      if (node && node.el) node.el.classList.remove("dragging");
      if (!dragging.moved) selectNode(dragging.nodeId);
      dragging = null;
      saveCanvasState();
    }
  });

  // Spacebar pan (only when prompt not focused)
  document.addEventListener("keydown", function (e) {
    if (e.code === "Space" && document.activeElement !== promptInput) {
      spaceHeld = true;
      canvasViewport.style.cursor = "grab";
      e.preventDefault();
    }
    // Delete selected node
    if ((e.code === "Delete" || e.code === "Backspace") && document.activeElement !== promptInput) {
      if (selectedId && !busy) {
        removeNode(selectedId);
        e.preventDefault();
      }
    }
  });
  document.addEventListener("keyup", function (e) {
    if (e.code === "Space") {
      spaceHeld = false;
      if (!isPanning) canvasViewport.style.cursor = "";
    }
  });

  // Touch: single-finger pan, pinch-to-zoom
  var touchState = { type: null, lastDist: 0, lastCenter: null };

  canvasViewport.addEventListener("touchstart", function (e) {
    if (e.touches.length === 2) {
      e.preventDefault();
      var a = e.touches[0], b = e.touches[1];
      touchState.type = "pinch";
      touchState.lastDist = Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);
      touchState.lastCenter = { x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 };
    } else if (e.touches.length === 1 && e.target === canvasViewport) {
      touchState.type = "pan";
      panStart.x = e.touches[0].clientX - camera.x;
      panStart.y = e.touches[0].clientY - camera.y;
    }
  }, { passive: false });

  canvasViewport.addEventListener("touchmove", function (e) {
    if (touchState.type === "pinch" && e.touches.length === 2) {
      e.preventDefault();
      var a = e.touches[0], b = e.touches[1];
      var dist = Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);
      var center = { x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 };
      var scaleDelta = dist / touchState.lastDist;
      var oldZoom = camera.zoom;
      camera.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, camera.zoom * scaleDelta));
      var ratio = camera.zoom / oldZoom;
      camera.x = center.x - (center.x - camera.x) * ratio;
      camera.y = center.y - (center.y - camera.y) * ratio;
      touchState.lastDist = dist;
      applyCameraTransform();
    } else if (touchState.type === "pan" && e.touches.length === 1) {
      camera.x = e.touches[0].clientX - panStart.x;
      camera.y = e.touches[0].clientY - panStart.y;
      applyCameraTransform();
    }
  }, { passive: false });

  canvasViewport.addEventListener("touchend", function () {
    touchState.type = null;
    saveCanvasState();
  });

  // ---------------------------------------------------------------
  // Node Management
  // ---------------------------------------------------------------

  function createNode(opts) {
    opts = opts || {};
    var id = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2);
    var center = viewportCenter();
    var offset = placementCounter * 30;
    placementCounter++;

    var node = {
      id: id,
      url: opts.url || null,
      previewSrc: null,
      x: opts.x != null ? opts.x : center.x + offset,
      y: opts.y != null ? opts.y : center.y + offset,
      width: opts.width || 340,
      height: opts.height || 340,
      prompt: opts.prompt || "",
      mode: opts.mode || "generate",
      model: opts.model || genVariant,
      parentId: opts.parentId || null,
      seed: opts.seed || null,
      timestamp: Date.now(),
      state: opts.state || "generating",
      el: null,
    };

    nodes.set(id, node);
    renderNode(node);
    return node;
  }

  function renderNode(node) {
    var el = document.createElement("div");
    el.className = "canvas-node";
    el.dataset.nodeId = node.id;
    el.style.transform = "translate(" + (node.x - node.width / 2) + "px," + (node.y - node.height / 2) + "px)";
    el.style.width = node.width + "px";
    el.style.height = node.height + "px";

    if (node.url) {
      var img = document.createElement("img");
      img.src = node.url + "?t=" + node.timestamp;
      img.alt = "Generated";
      img.draggable = false;
      el.appendChild(img);
    } else if (node.state === "generating") {
      var ph = document.createElement("span");
      ph.className = "node-placeholder";
      ph.textContent = "Generating...";
      el.appendChild(ph);
    }

    if (node.state === "generating") el.classList.add("generating");
    if (selectedId === node.id) {
      el.classList.add(mode === "edit" ? "edit-selected" : "selected");
    }

    // Node drag/select
    el.addEventListener("pointerdown", function (e) {
      if (e.button === 0 && !spaceHeld) {
        e.stopPropagation();
        var world = screenToWorld(e.clientX, e.clientY);
        dragging = {
          nodeId: node.id,
          offsetX: world.x - node.x,
          offsetY: world.y - node.y,
          moved: false,
        };
        el.classList.add("dragging");
        canvasViewport.setPointerCapture(e.pointerId);
      }
    });

    canvasWorld.appendChild(el);
    node.el = el;
  }

  function selectNode(id) {
    if (selectedId && nodes.has(selectedId)) {
      var prev = nodes.get(selectedId);
      if (prev.el) {
        prev.el.classList.remove("selected", "edit-selected");
      }
    }
    selectedId = id;
    if (id && nodes.has(id)) {
      var node = nodes.get(id);
      if (node.el) {
        node.el.classList.add(mode === "edit" ? "edit-selected" : "selected");
      }
    }
    saveCanvasState();
  }

  function removeNode(id) {
    var node = nodes.get(id);
    if (node && node.el) node.el.remove();
    nodes.delete(id);
    if (selectedId === id) selectedId = null;
    if (generatingId === id) generatingId = null;
    saveCanvasState();
  }

  // ---------------------------------------------------------------
  // Per-node overlays
  // ---------------------------------------------------------------

  function showNodeStepCounter(node, step, total) {
    var counter = node.el.querySelector(".node-step-counter");
    if (!counter) {
      counter = document.createElement("div");
      counter.className = "node-step-counter";
      node.el.appendChild(counter);
    }
    counter.textContent = step + " / " + total;
  }

  function showNodeLoading(node, text) {
    var overlay = node.el.querySelector(".node-loading-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.className = "node-loading-overlay";
      node.el.appendChild(overlay);
    }
    overlay.textContent = text;
    overlay.style.display = "";
  }

  function hideNodeLoading(node) {
    var overlay = node.el.querySelector(".node-loading-overlay");
    if (overlay) overlay.style.display = "none";
  }

  function clearNodeOverlays(node) {
    var counter = node.el.querySelector(".node-step-counter");
    if (counter) counter.remove();
    var overlay = node.el.querySelector(".node-loading-overlay");
    if (overlay) overlay.remove();
    var ph = node.el.querySelector(".node-placeholder");
    if (ph) ph.remove();
  }

  // ---------------------------------------------------------------
  // Viewport culling
  // ---------------------------------------------------------------

  var cullTimer = null;
  function requestCull() {
    if (cullTimer) return;
    cullTimer = requestAnimationFrame(function () {
      cullTimer = null;
      var vw = window.innerWidth, vh = window.innerHeight, margin = 300;
      nodes.forEach(function (node) {
        if (!node.el) return;
        var sx = node.x * camera.zoom + camera.x;
        var sy = node.y * camera.zoom + camera.y;
        var hw = (node.width * camera.zoom) / 2;
        var hh = (node.height * camera.zoom) / 2;
        var visible = sx + hw > -margin && sx - hw < vw + margin &&
                      sy + hh > -margin && sy - hh < vh + margin;
        node.el.style.display = visible ? "" : "none";
      });
    });
  }

  // ---------------------------------------------------------------
  // Persistence (localStorage)
  // ---------------------------------------------------------------

  var STORAGE_KEY = "hydra_canvas";

  function saveCanvasState() {
    var data = {
      camera: { x: camera.x, y: camera.y, zoom: camera.zoom },
      nodes: [],
      selectedId: selectedId,
    };
    nodes.forEach(function (n) {
      if (n.state !== "complete") return;
      data.nodes.push({
        id: n.id, url: n.url, x: n.x, y: n.y,
        width: n.width, height: n.height,
        prompt: n.prompt, mode: n.mode, model: n.model,
        parentId: n.parentId, seed: n.seed, timestamp: n.timestamp,
        state: n.state,
      });
    });
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); } catch (_) {}
  }

  function loadCanvasState() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return false;
      var saved = JSON.parse(raw);
      camera.x = saved.camera.x;
      camera.y = saved.camera.y;
      camera.zoom = saved.camera.zoom;
      applyCameraTransform();
      for (var i = 0; i < saved.nodes.length; i++) {
        var nd = saved.nodes[i];
        nd.el = null;
        nodes.set(nd.id, nd);
        renderNode(nd);
      }
      if (saved.selectedId && nodes.has(saved.selectedId)) {
        selectNode(saved.selectedId);
      }
      return saved.nodes.length > 0;
    } catch (_) { return false; }
  }

  // ---------------------------------------------------------------
  // Model selector
  // ---------------------------------------------------------------

  modelBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      if (busy) return;
      genVariant = btn.dataset.model;
      modelBtns.forEach(function (b) { b.classList.toggle("active", b === btn); });
      if (mode === "generate") syncDefaultSteps(genVariant);
    });
  });

  // ---------------------------------------------------------------
  // Settings panel
  // ---------------------------------------------------------------

  settingsBtn.addEventListener("click", function () {
    var open = settingsPanel.style.display !== "none";
    settingsPanel.style.display = open ? "none" : "";
    settingsBtn.classList.toggle("active", !open);
  });

  stepsRange.addEventListener("input", function () {
    stepsValue.textContent = stepsRange.value;
  });

  function syncDefaultSteps(variant) {
    var defaults = { srpo: 50, base: 50 };
    stepsRange.value = defaults[variant] || 50;
    stepsValue.textContent = stepsRange.value;
  }

  function syncStepsForMode(m) {
    if (m === "edit") {
      stepsRange.value = 20;
    } else {
      var defaults = { srpo: 50, base: 50 };
      stepsRange.value = defaults[genVariant] || 50;
    }
    stepsValue.textContent = stepsRange.value;
  }

  // ---------------------------------------------------------------
  // Mode toggle
  // ---------------------------------------------------------------

  modeToggle.addEventListener("click", function () {
    if (busy) return;
    mode = mode === "generate" ? "edit" : "generate";
    modeToggle.classList.toggle("edit", mode === "edit");
    promptInput.classList.toggle("edit-mode", mode === "edit");
    uploadImgBtn.style.display = mode === "edit" ? "" : "none";
    resolutionRow.style.display = mode === "edit" ? "none" : "";
    syncStepsForMode(mode);

    // Update node selection styling
    if (selectedId && nodes.has(selectedId)) {
      var node = nodes.get(selectedId);
      if (node.el) {
        node.el.classList.remove("selected", "edit-selected");
        node.el.classList.add(mode === "edit" ? "edit-selected" : "selected");
      }
    }

    if (mode === "generate") {
      modeToggle.title = "Generate mode (click to switch)";
      promptInput.placeholder = "describe your character...";
    } else {
      modeToggle.title = "Edit mode (click to switch)";
      var hasSelected = selectedId && nodes.has(selectedId) && nodes.get(selectedId).url;
      promptInput.placeholder = hasSelected ? "describe your edit..." : "select an image to edit...";
    }
  });

  // ---------------------------------------------------------------
  // LoRA upload
  // ---------------------------------------------------------------

  loraFile.addEventListener("change", async function () {
    var file = loraFile.files[0];
    if (!file) return;
    var trigger = triggerWord.value.trim() || "chrx";
    var fd = new FormData();
    fd.append("lora", file);
    fd.append("trigger_word", trigger);
    try {
      var resp = await fetch("/api/upload-lora", { method: "POST", body: fd });
      var data = await resp.json();
      if (resp.ok) {
        loraBtn.classList.add("loaded");
        loraStatus.textContent = data.name;
        loraStatus.classList.add("visible");
      } else {
        showToast(data.error || "Upload failed");
      }
    } catch (err) { showToast("Upload failed: " + err.message); }
    loraFile.value = "";
  });

  // ---------------------------------------------------------------
  // Image upload (edit mode)
  // ---------------------------------------------------------------

  imageFile.addEventListener("change", async function () {
    var file = imageFile.files[0];
    if (!file) return;
    var fd = new FormData();
    fd.append("image", file);
    try {
      var resp = await fetch("/api/upload-image", { method: "POST", body: fd });
      var data = await resp.json();
      if (resp.ok && data.image_url) {
        var node = createNode({
          url: data.image_url,
          state: "complete",
          mode: "edit",
          prompt: "(uploaded)",
        });
        selectNode(node.id);
        promptInput.placeholder = "describe your edit...";
      } else {
        showToast(data.error || "Upload failed");
      }
    } catch (err) { showToast("Upload failed: " + err.message); }
    imageFile.value = "";
  });

  // ---------------------------------------------------------------
  // Prompt submission
  // ---------------------------------------------------------------

  promptInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });

  async function submit() {
    var prompt = promptInput.value.trim();
    if (!prompt || busy) return;

    var selectedUrl = null;
    if (mode === "edit") {
      if (selectedId && nodes.has(selectedId)) {
        selectedUrl = nodes.get(selectedId).url;
      }
      if (!selectedUrl) {
        showToast("Select an image to edit");
        return;
      }
    }

    busy = true;
    promptInput.disabled = true;
    modeToggle.classList.add("loading");
    placementCounter = 0;

    var rParts = resolutionSelect.value.split("x");
    var w = parseInt(rParts[0], 10);
    var h = parseInt(rParts[1], 10);
    var steps = parseInt(stepsRange.value, 10);

    // Compute display size
    var maxSide = 380;
    var aspect = w / h;
    var dispW, dispH;
    if (aspect >= 1) { dispW = maxSide; dispH = maxSide / aspect; }
    else { dispH = maxSide; dispW = maxSide * aspect; }

    var newNode = createNode({
      prompt: prompt,
      mode: mode,
      model: genVariant,
      width: dispW,
      height: dispH,
      parentId: mode === "edit" ? selectedId : null,
      state: "generating",
    });
    generatingId = newNode.id;
    selectNode(newNode.id);

    var endpoint = mode === "generate" ? "/api/generate" : "/api/edit";
    var payload = mode === "generate"
      ? { prompt: prompt, model: genVariant, width: w, height: h, steps: steps }
      : { prompt: prompt, steps: steps, source_image: selectedUrl };

    try {
      var resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      var data = await resp.json();

      if (resp.ok && data.image_url) {
        newNode.url = data.image_url;
        newNode.seed = data.seed || null;
        newNode.state = "complete";

        var img = newNode.el.querySelector("img");
        if (!img) {
          img = document.createElement("img");
          img.draggable = false;
          newNode.el.appendChild(img);
        }
        img.src = data.image_url + "?t=" + Date.now();
        img.alt = "Generated";
        img.classList.remove("preview-img");

        newNode.el.classList.remove("generating");
        clearNodeOverlays(newNode);
      } else {
        showToast(data.error || "Request failed");
        removeNode(newNode.id);
      }
    } catch (err) {
      showToast("Request failed: " + err.message);
      removeNode(newNode.id);
    } finally {
      busy = false;
      generatingId = null;
      promptInput.disabled = false;
      modeToggle.classList.remove("loading");
      promptInput.focus();
      saveCanvasState();
    }
  }

  // ---------------------------------------------------------------
  // SSE
  // ---------------------------------------------------------------

  var evtSource = new EventSource("/api/stream");

  function attachSSEListeners(src) {
    src.addEventListener("preview", function (e) {
      var data = JSON.parse(e.data);

      // Pick up mid-generation on reconnect
      if (!busy) {
        busy = true;
        promptInput.disabled = true;
        modeToggle.classList.add("loading");
        if (!generatingId) {
          var gNode = createNode({ state: "generating" });
          generatingId = gNode.id;
          selectNode(gNode.id);
        }
      }

      var node = generatingId && nodes.get(generatingId);
      if (!node || !node.el) return;

      var img = node.el.querySelector("img");
      if (!img) {
        // Remove placeholder text
        var ph = node.el.querySelector(".node-placeholder");
        if (ph) ph.remove();
        img = document.createElement("img");
        img.alt = "Preview";
        img.draggable = false;
        node.el.appendChild(img);
      }
      img.src = data.image;
      img.classList.add("preview-img");

      showNodeStepCounter(node, data.step, data.total);
    });

    src.addEventListener("model_status", function (e) {
      var data = JSON.parse(e.data);
      var node = generatingId && nodes.get(generatingId);
      if (data.action === "loading" && node) {
        showNodeLoading(node, "Loading " + data.name + "...");
      } else if (data.action === "ready" && node) {
        hideNodeLoading(node);
      }
    });

    src.addEventListener("error", function (e) {
      var data;
      try { data = JSON.parse(e.data); } catch (_) { return; }
      if (data && data.message) showToast(data.message);
    });

    src.onopen = function () {
      fetch("/api/status").then(function (r) { return r.json(); }).then(function (data) {
        if (data.lora) {
          loraBtn.classList.add("loaded");
          loraStatus.textContent = data.lora.name;
          loraStatus.classList.add("visible");
        }
        // If server finished while we were disconnected
        if (!data.busy && busy) {
          busy = false;
          promptInput.disabled = false;
          modeToggle.classList.remove("loading");
          if (data.image_url && generatingId) {
            var node = nodes.get(generatingId);
            if (node) {
              node.url = data.image_url;
              node.state = "complete";
              var img = node.el.querySelector("img") || document.createElement("img");
              img.src = data.image_url + "?t=" + Date.now();
              img.classList.remove("preview-img");
              img.draggable = false;
              if (!img.parentNode) node.el.appendChild(img);
              node.el.classList.remove("generating");
              clearNodeOverlays(node);
            }
          }
          generatingId = null;
          saveCanvasState();
        }
      }).catch(function () {});
    };
  }

  attachSSEListeners(evtSource);

  // ---------------------------------------------------------------
  // Restore state
  // ---------------------------------------------------------------

  function restoreState(data) {
    if (data.lora) {
      loraBtn.classList.add("loaded");
      loraStatus.textContent = data.lora.name;
      loraStatus.classList.add("visible");
      if (data.lora.trigger) triggerWord.value = data.lora.trigger;
    }
    if (data.gen_variant) {
      genVariant = data.gen_variant;
      modelBtns.forEach(function (b) { b.classList.toggle("active", b.dataset.model === genVariant); });
    }

    // If no nodes from localStorage, restore last server image
    if (data.image_url && nodes.size === 0) {
      var node = createNode({ url: data.image_url, state: "complete" });
      selectNode(node.id);
    }

    if (data.busy) {
      busy = true;
      promptInput.disabled = true;
      modeToggle.classList.add("loading");
      if (!generatingId) {
        var gNode = createNode({ state: "generating" });
        generatingId = gNode.id;
        selectNode(gNode.id);
      }
    }

    if (data.mode === "edit") {
      mode = "edit";
      modeToggle.classList.add("edit");
      modeToggle.title = "Edit mode (click to switch)";
      promptInput.classList.add("edit-mode");
      promptInput.placeholder = "describe your edit...";
      uploadImgBtn.style.display = "";
      resolutionRow.style.display = "none";
      syncStepsForMode("edit");
    }
  }

  // Load localStorage first, then server state
  var hadSaved = loadCanvasState();
  fetch("/api/status").then(function (r) { return r.json(); }).then(restoreState).catch(function () {});

  // ---------------------------------------------------------------
  // Toast
  // ---------------------------------------------------------------

  var toastEl = null;
  var toastTimer = null;

  function showToast(message) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.className = "toast";
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = message;
    toastEl.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove("visible"); }, 3500);
  }

})();
