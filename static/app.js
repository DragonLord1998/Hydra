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
  var settingsBtn    = document.getElementById("settingsBtn");
  var settingsPanel  = document.getElementById("settingsPanel");
  var resolutionSelect = document.getElementById("resolutionSelect");
  var resolutionRow  = document.getElementById("resolutionRow");
  var stepsRange     = document.getElementById("stepsRange");
  var stepsValue     = document.getElementById("stepsValue");
  var cfgRange       = document.getElementById("cfgRange");
  var cfgValue       = document.getElementById("cfgValue");
  var cfgRow         = document.getElementById("cfgRow");
  var uploadImgBtn   = document.getElementById("uploadImgBtn");
  var imageFile      = document.getElementById("imageFile");
  var uploadProgress     = document.getElementById("uploadProgress");
  var uploadProgressFill = document.getElementById("uploadProgressFill");
  var uploadProgressText = document.getElementById("uploadProgressText");

  // --- State ---
  var mode = "generate";
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
  var resizing = null; // { nodeId, corner, startX, startY, startW, startH, startNodeX, startNodeY }

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
    } else if (e.button === 0 && !spaceHeld && (e.target === canvasViewport || e.target === canvasWorld)) {
      // Click on empty canvas = deselect
      selectNode(null);
    }
  });

  canvasViewport.addEventListener("pointermove", function (e) {
    if (isPanning) {
      camera.x = e.clientX - panStart.x;
      camera.y = e.clientY - panStart.y;
      applyCameraTransform();
      return;
    }
    if (resizing) {
      var world = screenToWorld(e.clientX, e.clientY);
      var node = nodes.get(resizing.nodeId);
      if (!node) return;
      var dx = world.x - resizing.startX;
      var dy = world.y - resizing.startY;
      var c = resizing.corner;
      // Use the larger delta to maintain aspect ratio
      var delta;
      if (c === "se") delta = Math.max(dx, dy / resizing.aspect * resizing.aspect);
      else if (c === "sw") delta = Math.max(-dx, dy / resizing.aspect * resizing.aspect);
      else if (c === "ne") delta = Math.max(dx, -dy / resizing.aspect * resizing.aspect);
      else delta = Math.max(-dx, -dy / resizing.aspect * resizing.aspect);
      // Compute new size preserving aspect ratio
      var newW, newH;
      if (c === "se") {
        newW = Math.max(80, resizing.startW + dx);
        newH = newW / resizing.aspect;
      } else if (c === "sw") {
        newW = Math.max(80, resizing.startW - dx);
        newH = newW / resizing.aspect;
      } else if (c === "ne") {
        newW = Math.max(80, resizing.startW + dx);
        newH = newW / resizing.aspect;
      } else { // nw
        newW = Math.max(80, resizing.startW - dx);
        newH = newW / resizing.aspect;
      }
      // Adjust position so the opposite corner stays fixed
      if (c === "nw") {
        node.x = resizing.startNodeX + (resizing.startW - newW) / 2;
        node.y = resizing.startNodeY + (resizing.startH - newH) / 2;
      } else if (c === "ne") {
        node.x = resizing.startNodeX + (newW - resizing.startW) / 2;
        node.y = resizing.startNodeY + (resizing.startH - newH) / 2;
      } else if (c === "sw") {
        node.x = resizing.startNodeX + (resizing.startW - newW) / 2;
        node.y = resizing.startNodeY + (newH - resizing.startH) / 2;
      } else { // se
        node.x = resizing.startNodeX + (newW - resizing.startW) / 2;
        node.y = resizing.startNodeY + (newH - resizing.startH) / 2;
      }
      node.width = newW;
      node.height = newH;
      node.el.style.width = newW + "px";
      node.el.style.height = newH + "px";
      node.el.style.transform = "translate(" + (node.x - newW / 2) + "px," + (node.y - newH / 2) + "px)";
      return;
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
    if (resizing) {
      var resizedNode = nodes.get(resizing.nodeId);
      resizing = null;
      // Trigger upscale if resized significantly larger than original
      if (resizedNode && resizedNode.url && !resizedNode.upscaling &&
          resizedNode.width > resizedNode.originalWidth * 1.5) {
        triggerUpscale(resizedNode);
      }
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
    var activeTag = document.activeElement && document.activeElement.tagName;
    if (e.code === "Space" && activeTag !== "INPUT" && activeTag !== "SELECT" && activeTag !== "TEXTAREA") {
      spaceHeld = true;
      canvasViewport.style.cursor = "grab";
      e.preventDefault();
    }
    // Delete selected node (only when no input is focused)
    var tag = document.activeElement && document.activeElement.tagName;
    if ((e.code === "Delete" || e.code === "Backspace") && tag !== "INPUT" && tag !== "SELECT" && tag !== "TEXTAREA") {
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
      model: opts.model || "flux2",
      parentId: opts.parentId || null,
      seed: opts.seed || null,
      timestamp: Date.now(),
      state: opts.state || "generating",
      originalUrl: opts.url || null,
      originalWidth: opts.width || 340,
      originalHeight: opts.height || 340,
      upscaling: false,
      previousVersions: opts.previousVersions || [],
      poseData: opts.poseData || null,
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

    // Show upscale badge if image was previously upscaled
    if (node.url && node.originalUrl && node.url !== node.originalUrl) {
      var badge = document.createElement("div");
      badge.className = "node-upscale-badge";
      badge.textContent = "UPSCALED";
      el.appendChild(badge);
    }

    // Human (pose) button — only on completed image nodes
    if (node.url && node.state === "complete") {
      var humanBtn = document.createElement("button");
      humanBtn.className = "node-human-btn";
      humanBtn.title = "Extract & edit pose";
      humanBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="3"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/><line x1="9" y1="22" x2="12" y2="16"/><line x1="15" y1="22" x2="12" y2="16"/></svg>';
      humanBtn.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      humanBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        extractAndEditPose(node);
      });
      el.appendChild(humanBtn);
    }

    // Revert button — only if node has previous versions
    if (node.previousVersions && node.previousVersions.length > 0) {
      var revertBtn = document.createElement("button");
      revertBtn.className = "node-revert-btn";
      revertBtn.title = "Revert to previous version (" + node.previousVersions.length + ")";
      revertBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>';
      revertBtn.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      revertBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        revertNode(node);
      });
      el.appendChild(revertBtn);
    }

    if (node.state === "generating") el.classList.add("generating");
    if (selectedId === node.id) {
      el.classList.add(mode === "edit" ? "edit-selected" : "selected");
    }

    // Resize handles (4 corners)
    ["nw", "ne", "sw", "se"].forEach(function (corner) {
      var handle = document.createElement("div");
      handle.className = "node-resize-handle handle-" + corner;
      handle.addEventListener("pointerdown", function (e) {
        if (e.button !== 0 || spaceHeld) return;
        e.stopPropagation();
        var world = screenToWorld(e.clientX, e.clientY);
        resizing = {
          nodeId: node.id,
          corner: corner,
          startX: world.x,
          startY: world.y,
          startW: node.width,
          startH: node.height,
          startNodeX: node.x,
          startNodeY: node.y,
          aspect: node.width / node.height,
        };
        canvasViewport.setPointerCapture(e.pointerId);
      });
      el.appendChild(handle);
    });

    // Node drag/select
    el.addEventListener("pointerdown", function (e) {
      if (e.button === 0 && !spaceHeld && !resizing) {
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
  // Pose: Extract, Edit, Regenerate
  // ---------------------------------------------------------------

  async function extractAndEditPose(node) {
    if (busy || !node.url) return;

    showNodeLoading(node, "Extracting pose...");

    try {
      var resp = await fetch("/api/extract-pose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_image: node.url }),
      });
      var data = await resp.json();

      if (!resp.ok) {
        showToast(data.error || "Pose extraction failed");
        hideNodeLoading(node);
        return;
      }

      hideNodeLoading(node);

      // Open the fullscreen pose editor
      if (!window.HydraPoseEditor) {
        showToast("Pose editor not loaded — check Three.js");
        return;
      }

      window.HydraPoseEditor.open(
        data,
        node.url,
        node.url,
        node.prompt || "",
        function onRegenerate(poseImage, prompt, characterImageUrl) {
          regenerateWithPose(node, poseImage, prompt, characterImageUrl);
        },
        function onClose() {
          // Nothing to do — editor cleans itself up
        }
      );
    } catch (err) {
      showToast("Pose extraction failed: " + err.message);
      hideNodeLoading(node);
    }
  }

  async function regenerateWithPose(node, poseImage, prompt, characterImageUrl) {
    if (busy) return;
    busy = true; // Set immediately to prevent double-fire

    promptInput.disabled = true;
    modeToggle.classList.add("loading");

    // Save current version before overwriting
    if (node.url) {
      if (!node.previousVersions) node.previousVersions = [];
      node.previousVersions.push({
        url: node.url,
        prompt: node.prompt,
        timestamp: Date.now(),
      });
      // Cap at 5 versions
      if (node.previousVersions.length > 5) {
        node.previousVersions = node.previousVersions.slice(-5);
      }
    }

    showNodeLoading(node, "Regenerating pose...");
    node.el.classList.add("generating");
    generatingId = node.id;

    try {
      var resp = await fetch("/api/generate-posed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          character_image: characterImageUrl,
          pose_image: poseImage,
          prompt: prompt,
          steps: parseInt(stepsRange.value, 10) || 28,
        }),
      });
      var data = await resp.json();

      if (resp.ok && data.image_url) {
        node.url = data.image_url;
        node.prompt = prompt;
        node.timestamp = Date.now();
        node.state = "complete";

        // Update image in the node
        var img = node.el.querySelector("img");
        if (!img) {
          img = document.createElement("img");
          img.draggable = false;
          node.el.appendChild(img);
        }
        img.src = data.image_url + "?t=" + Date.now();
        img.alt = "Posed";
        img.classList.remove("preview-img");
        node.el.classList.remove("generating");
        clearNodeOverlays(node);

        // Re-render the node to add/update revert button
        refreshNodeButtons(node);
      } else {
        showToast(data.error || "Posed generation failed");
        // Revert the version push since generation failed
        if (node.previousVersions && node.previousVersions.length > 0) {
          node.previousVersions.pop();
        }
        node.el.classList.remove("generating");
        hideNodeLoading(node);
      }
    } catch (err) {
      showToast("Posed generation failed: " + err.message);
      if (node.previousVersions && node.previousVersions.length > 0) {
        node.previousVersions.pop();
      }
      node.el.classList.remove("generating");
      hideNodeLoading(node);
    } finally {
      busy = false;
      generatingId = null;
      promptInput.disabled = false;
      modeToggle.classList.remove("loading");
      promptInput.focus();
      saveCanvasState();
    }
  }

  function revertNode(node) {
    if (!node.previousVersions || node.previousVersions.length === 0) return;

    var prev = node.previousVersions.pop();
    node.url = prev.url;
    node.prompt = prev.prompt;
    node.timestamp = Date.now();

    var img = node.el.querySelector("img");
    if (img) {
      img.src = prev.url + "?t=" + Date.now();
    }

    refreshNodeButtons(node);
    saveCanvasState();
    showToast("Reverted (" + node.previousVersions.length + " versions left)");
  }

  function refreshNodeButtons(node) {
    // Remove existing human/revert buttons and re-add them
    var oldHuman = node.el.querySelector(".node-human-btn");
    if (oldHuman) oldHuman.remove();
    var oldRevert = node.el.querySelector(".node-revert-btn");
    if (oldRevert) oldRevert.remove();

    // Re-add human button
    if (node.url && node.state === "complete") {
      var humanBtn = document.createElement("button");
      humanBtn.className = "node-human-btn";
      humanBtn.title = "Extract & edit pose";
      humanBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="3"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/><line x1="9" y1="22" x2="12" y2="16"/><line x1="15" y1="22" x2="12" y2="16"/></svg>';
      humanBtn.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      humanBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        extractAndEditPose(node);
      });
      node.el.appendChild(humanBtn);
    }

    // Re-add revert button if there are versions
    if (node.previousVersions && node.previousVersions.length > 0) {
      var revertBtn = document.createElement("button");
      revertBtn.className = "node-revert-btn";
      revertBtn.title = "Revert to previous version (" + node.previousVersions.length + ")";
      revertBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>';
      revertBtn.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      revertBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        revertNode(node);
      });
      node.el.appendChild(revertBtn);
    }
  }

  // ---------------------------------------------------------------
  // SeedVR2 Upscale
  // ---------------------------------------------------------------

  async function triggerUpscale(node) {
    if (!node.originalUrl || node.upscaling) return;
    node.upscaling = true;

    // Compute target resolution based on how much larger the node is
    var scale = node.width / node.originalWidth;
    var targetRes = Math.round(Math.min(scale * 1024, 4096));

    showNodeLoading(node, "Upscaling...");

    try {
      var resp = await fetch("/api/upscale", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_image: node.originalUrl,
          resolution: targetRes,
        }),
      });
      var data = await resp.json();

      if (resp.ok && data.image_url) {
        // Keep original reference, update displayed image
        node.url = data.image_url;
        var img = node.el.querySelector("img");
        if (img) {
          img.src = data.image_url + "?t=" + Date.now();
        }
        // Add or update upscale badge
        var badge = node.el.querySelector(".node-upscale-badge");
        if (!badge) {
          badge = document.createElement("div");
          badge.className = "node-upscale-badge";
          node.el.appendChild(badge);
        }
        badge.textContent = "UPSCALED " + targetRes + "p";
        showToast("Upscaled to " + targetRes + "p");
      } else {
        showToast(data.error || "Upscale failed");
      }
    } catch (err) {
      showToast("Upscale failed: " + err.message);
    } finally {
      node.upscaling = false;
      hideNodeLoading(node);
      saveCanvasState();
    }
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
        originalUrl: n.originalUrl, originalWidth: n.originalWidth,
        originalHeight: n.originalHeight,
        previousVersions: n.previousVersions || [],
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
        nd.upscaling = false;
        // Backfill fields for nodes saved before upscale/pose support
        if (!nd.originalUrl) nd.originalUrl = nd.url;
        if (!nd.originalWidth) nd.originalWidth = nd.width;
        if (!nd.originalHeight) nd.originalHeight = nd.height;
        if (!nd.previousVersions) nd.previousVersions = [];
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

  // Model selector removed -- Flux 2 is the only model

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

  cfgRange.addEventListener("input", function () {
    cfgValue.textContent = cfgRange.value;
  });

  function syncDefaultSteps() {
    stepsRange.value = 50;
    stepsValue.textContent = "50";
    cfgRange.value = 4.0;
    cfgValue.textContent = "4";
  }

  function syncStepsForMode(m) {
    stepsRange.value = m === "edit" ? 50 : 50;
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
    cfgRow.style.display = mode === "edit" ? "none" : "";
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

  function showUploadProgress(pct) {
    uploadProgress.classList.add("active");
    loraStatus.classList.remove("visible");
    uploadProgressFill.style.width = pct + "%";
    uploadProgressText.textContent = pct + "%";
  }

  function hideUploadProgress() {
    uploadProgress.classList.remove("active");
  }

  loraFile.addEventListener("change", function () {
    var file = loraFile.files[0];
    if (!file) return;
    var trigger = triggerWord.value.trim() || "chrx";
    var fd = new FormData();
    fd.append("lora", file);
    fd.append("trigger_word", trigger);

    showUploadProgress(0);

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload-lora");

    xhr.upload.onprogress = function (e) {
      if (e.lengthComputable) {
        showUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = function () {
      hideUploadProgress();
      try {
        var data = JSON.parse(xhr.responseText);
        if (xhr.status === 200) {
          loraBtn.classList.add("loaded");
          loraStatus.textContent = data.name;
          loraStatus.classList.add("visible");
        } else {
          showToast(data.error || "Upload failed");
        }
      } catch (_) { showToast("Upload failed"); }
    };

    xhr.onerror = function () {
      hideUploadProgress();
      showToast("Upload failed");
    };

    xhr.send(fd);
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
      model: "flux2",
      width: dispW,
      height: dispH,
      parentId: mode === "edit" ? selectedId : null,
      state: "generating",
    });
    generatingId = newNode.id;
    selectNode(newNode.id);

    var cfg = parseFloat(cfgRange.value);
    var endpoint = mode === "generate" ? "/api/generate" : "/api/edit";
    var payload = mode === "generate"
      ? { prompt: prompt, width: w, height: h, steps: steps, cfg: cfg }
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
        newNode.originalUrl = data.image_url;
        newNode.seed = data.seed || null;
        newNode.state = "complete";
        newNode.originalWidth = newNode.width;
        newNode.originalHeight = newNode.height;

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
        refreshNodeButtons(newNode);
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
      cfgRow.style.display = "none";
      syncStepsForMode("edit");
    }
  }

  // Load localStorage first, then server state
  loadCanvasState();
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
