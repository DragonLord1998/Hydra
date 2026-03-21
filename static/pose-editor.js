/* ================================================================
   Hydra — 3D Pose Editor (Three.js)
   SAM 3D Body joint visualization + IK manipulation
   ================================================================ */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

class PoseEditor {
  constructor() {
    this.overlay = null;
    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.controls = null;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.dragPlane = new THREE.Plane();

    // State
    this.joints = [];           // Array of { name, index, position:[x,y,z], draggable }
    this.bones = [];            // Array of [i, j] index pairs
    this.ikChains = [];         // Array of [root, mid, end] index triples
    this.draggableIndices = []; // Which joints can be dragged

    // Three.js objects
    this.jointMeshes = new Map();  // index → THREE.Mesh
    this.boneMeshes = [];          // Array of THREE.Mesh (cylinders)
    this.jointGroup = null;
    this.boneGroup = null;

    // Interaction
    this.selectedJoint = null;
    this.isDragging = false;
    this.dragOffset = new THREE.Vector3();
    this.originalPositions = null; // For reset

    // Callbacks
    this.onRegenerate = null;  // (poseImageBase64, prompt) => void
    this.onClose = null;

    // Reference data
    this.sourceImageUrl = null;
    this.originalPrompt = "";
    this.characterImageUrl = null;

    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this._onKeyDown = this._onKeyDown.bind(this);
    this._animate = this._animate.bind(this);
    this._animId = null;
  }

  // ---------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------

  open(poseData, sourceImageUrl, characterImageUrl, prompt, onRegenerate, onClose) {
    this.joints = poseData.joints;
    this.bones = poseData.bones;
    this.ikChains = poseData.ik_chains || [];
    this.draggableIndices = poseData.draggable_indices || [];
    this.sourceImageUrl = sourceImageUrl;
    this.characterImageUrl = characterImageUrl;
    this.originalPrompt = prompt || "";
    this.onRegenerate = onRegenerate;
    this.onClose = onClose;

    // Save original positions for reset
    this.originalPositions = this.joints.map(j => [...j.position]);

    this._buildUI();
    this._buildScene();
    this._buildMannequin();
    this._startLoop();

    document.addEventListener("keydown", this._onKeyDown);
  }

  close() {
    this._stopLoop();
    document.removeEventListener("keydown", this._onKeyDown);
    if (this._onResize) {
      window.removeEventListener("resize", this._onResize);
      this._onResize = null;
    }

    // Remove pointer listeners before disposing renderer
    if (this.renderer) {
      this.renderer.domElement.removeEventListener("pointerdown", this._onPointerDown);
      this.renderer.domElement.removeEventListener("pointermove", this._onPointerMove);
      this.renderer.domElement.removeEventListener("pointerup", this._onPointerUp);
    }

    // Dispose all GPU resources (geometries + materials)
    if (this.scene) {
      this.scene.traverse(function (obj) {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(function (m) { m.dispose(); });
          else obj.material.dispose();
        }
      });
    }

    if (this._materials) {
      this._materials.joint.dispose();
      this._materials.draggable.dispose();
      this._materials.selected.dispose();
      this._materials = null;
    }

    if (this.controls) { this.controls.dispose(); this.controls = null; }
    if (this.renderer) { this.renderer.dispose(); this.renderer = null; }
    if (this.overlay) { this.overlay.remove(); this.overlay = null; }

    this.scene = null;
    this.camera = null;
    this.jointMeshes.clear();
    this.boneMeshes = [];
    this._limbMeshes = [];
    this.selectedJoint = null;
    this.isDragging = false;

    if (this.onClose) this.onClose();
  }

  renderPoseImage() {
    if (!this.renderer || !this.scene || !this.camera) return null;

    // Render from front view for AnyPose
    var frontCam = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    frontCam.position.set(0, 0, 3);
    frontCam.lookAt(0, 0, 0);

    // Temporarily resize renderer for square output at 1x pixel ratio
    var prevSize = new THREE.Vector2();
    this.renderer.getSize(prevSize);
    var prevPixelRatio = this.renderer.getPixelRatio();
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(512, 512);

    // White background for pose reference
    var prevBg = this.scene.background;
    this.scene.background = new THREE.Color(0xffffff);

    this.renderer.render(this.scene, frontCam);
    var dataUrl = this.renderer.domElement.toDataURL("image/png");

    // Restore
    this.scene.background = prevBg;
    this.renderer.setPixelRatio(prevPixelRatio);
    this.renderer.setSize(prevSize.x, prevSize.y);

    return dataUrl;
  }

  // ---------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------

  _buildUI() {
    if (this.overlay) this.overlay.remove();

    var overlay = document.createElement("div");
    overlay.className = "pose-editor-overlay";
    overlay.innerHTML = `
      <div class="pose-editor-layout">
        <div class="pose-viewport" id="poseViewport"></div>
        <div class="pose-sidebar">
          <div class="pose-sidebar-section">
            <div class="pose-sidebar-label">REFERENCE</div>
            <img class="pose-reference-img" id="poseRefImg" alt="Reference">
          </div>
          <div class="pose-sidebar-section">
            <div class="pose-sidebar-label">PRESETS</div>
            <div class="pose-presets">
              <button class="pose-preset-btn" data-preset="tpose">T-Pose</button>
              <button class="pose-preset-btn" data-preset="standing">Standing</button>
              <button class="pose-preset-btn" data-preset="sitting">Sitting</button>
              <button class="pose-preset-btn" data-preset="action">Action</button>
            </div>
          </div>
          <button class="pose-reset-btn" id="poseResetBtn">Reset to Original</button>
          <div class="pose-joint-info" id="poseJointInfo">Click a joint to select</div>
        </div>
      </div>
      <div class="pose-bottom-bar">
        <input type="text" class="pose-prompt-input" id="posePromptInput"
               placeholder="describe the character..." spellcheck="false">
        <button class="pose-regen-btn" id="poseRegenBtn">Regenerate</button>
        <button class="pose-cancel-btn" id="poseCancelBtn">Cancel</button>
      </div>
    `;

    // Set values programmatically to avoid XSS via innerHTML interpolation
    overlay.querySelector("#poseRefImg").src = this.sourceImageUrl;
    overlay.querySelector("#posePromptInput").value = this.originalPrompt;

    document.body.appendChild(overlay);
    this.overlay = overlay;

    // Wire events
    var self = this;
    overlay.querySelector("#poseCancelBtn").addEventListener("click", function () {
      self.close();
    });

    overlay.querySelector("#poseRegenBtn").addEventListener("click", function () {
      self._handleRegenerate();
    });

    overlay.querySelector("#poseResetBtn").addEventListener("click", function () {
      self._resetPose();
    });

    overlay.querySelector("#posePromptInput").addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        self._handleRegenerate();
      }
      e.stopPropagation(); // Don't trigger canvas shortcuts
    });

    overlay.querySelectorAll(".pose-preset-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        self._applyPreset(btn.dataset.preset);
      });
    });
  }

  _escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  _handleRegenerate() {
    // Disable button immediately to prevent double-click
    var regenBtn = this.overlay.querySelector("#poseRegenBtn");
    if (regenBtn) regenBtn.disabled = true;

    var prompt = this.overlay.querySelector("#posePromptInput").value.trim();
    var poseImage = this.renderPoseImage();
    if (!poseImage) {
      if (regenBtn) regenBtn.disabled = false;
      return;
    }

    if (this.onRegenerate) {
      this.onRegenerate(poseImage, prompt, this.characterImageUrl);
    }
    this.close();
  }

  // ---------------------------------------------------------------
  // Three.js scene
  // ---------------------------------------------------------------

  _buildScene() {
    var viewport = this.overlay.querySelector("#poseViewport");
    var w = viewport.clientWidth;
    var h = viewport.clientHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0f);

    // Camera
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
    this.camera.position.set(0, 0, 3.5);
    this.camera.lookAt(0, 0, 0);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    viewport.appendChild(this.renderer.domElement);

    // Orbit controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.enablePan = true;
    this.controls.mouseButtons = {
      LEFT: null,         // We handle left-click for joint selection
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.ROTATE,
    };

    // Lights
    var ambient = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambient);
    var dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(2, 3, 4);
    this.scene.add(dir);

    // Grid helper
    var grid = new THREE.GridHelper(4, 20, 0x222233, 0x111122);
    grid.rotation.x = 0; // XZ plane
    grid.position.y = -1.2;
    this.scene.add(grid);

    // Resize handler
    var self = this;
    this._onResize = function () {
      var vw = viewport.clientWidth, vh = viewport.clientHeight;
      if (vw === 0 || vh === 0) return;
      self.camera.aspect = vw / vh;
      self.camera.updateProjectionMatrix();
      self.renderer.setSize(vw, vh);
    };
    window.addEventListener("resize", this._onResize);

    // Interaction events
    this.renderer.domElement.addEventListener("pointerdown", this._onPointerDown);
    this.renderer.domElement.addEventListener("pointermove", this._onPointerMove);
    this.renderer.domElement.addEventListener("pointerup", this._onPointerUp);
  }

  // ---------------------------------------------------------------
  // Mannequin rendering
  // ---------------------------------------------------------------

  _buildMannequin() {
    this.jointGroup = new THREE.Group();
    this.boneGroup = new THREE.Group();
    this.scene.add(this.boneGroup);
    this.scene.add(this.jointGroup);

    // Center the pose around origin
    this._centerPose();

    // Create joint spheres
    var jointGeom = new THREE.SphereGeometry(0.025, 12, 12);
    var draggableGeom = new THREE.SphereGeometry(0.035, 16, 16);
    var jointMat = new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.4 });
    var draggableMat = new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.3, emissive: 0x10b981, emissiveIntensity: 0.2 });
    var selectedMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.3, emissive: 0x3b82f6, emissiveIntensity: 0.4 });

    this._materials = { joint: jointMat, draggable: draggableMat, selected: selectedMat };

    for (var i = 0; i < this.joints.length; i++) {
      var j = this.joints[i];
      var isDraggable = this.draggableIndices.indexOf(j.index) !== -1;

      // Skip finger joints for cleaner visualization
      if (!isDraggable && j.name.match(/(thumb|index|middle|ring|pinky)/)) continue;

      var geom = isDraggable ? draggableGeom : jointGeom;
      var mat = isDraggable ? draggableMat.clone() : jointMat.clone();
      var mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(j.position[0], j.position[1], j.position[2]);
      mesh.userData = { jointIndex: j.index, draggable: isDraggable, name: j.name };

      this.jointGroup.add(mesh);
      this.jointMeshes.set(j.index, mesh);
    }

    // Create bones (cylinders between joints)
    this._updateBones();

    // Add body volume (capsules for limbs, box for torso)
    this._buildBodyVolume();
  }

  _centerPose() {
    // Compute bounding box center of major joints and translate to origin
    var draggable = this.joints.filter(function (j) { return j.draggable; });
    if (draggable.length === 0) draggable = this.joints;

    var cx = 0, cy = 0, cz = 0;
    for (var i = 0; i < draggable.length; i++) {
      cx += draggable[i].position[0];
      cy += draggable[i].position[1];
      cz += draggable[i].position[2];
    }
    cx /= draggable.length;
    cy /= draggable.length;
    cz /= draggable.length;

    for (var i = 0; i < this.joints.length; i++) {
      this.joints[i].position[0] -= cx;
      this.joints[i].position[1] -= cy;
      this.joints[i].position[2] -= cz;
    }

    // Also update original positions
    if (this.originalPositions) {
      for (var i = 0; i < this.originalPositions.length; i++) {
        this.originalPositions[i][0] -= cx;
        this.originalPositions[i][1] -= cy;
        this.originalPositions[i][2] -= cz;
      }
    }
  }

  _buildBodyVolume() {
    // Simple capsule-like body parts for a recognizable mannequin silhouette
    var limbPairs = [
      [5, 7],   // left upper arm
      [7, 62],  // left forearm
      [6, 8],   // right upper arm
      [8, 41],  // right forearm
      [9, 11],  // left thigh
      [11, 13], // left shin
      [10, 12], // right thigh
      [12, 14], // right shin
    ];

    var bodyMat = new THREE.MeshStandardMaterial({
      color: 0x555566,
      roughness: 0.6,
      transparent: true,
      opacity: 0.35,
    });

    this._limbMeshes = [];

    for (var i = 0; i < limbPairs.length; i++) {
      var a = this._getJointPos(limbPairs[i][0]);
      var b = this._getJointPos(limbPairs[i][1]);
      if (!a || !b) continue;

      var limb = this._createCapsule(a, b, 0.03, bodyMat);
      if (limb) {
        limb.userData.limbPair = limbPairs[i];
        this.scene.add(limb);
        this._limbMeshes.push(limb);
      }
    }

    // Torso — connect shoulders to hips
    var ls = this._getJointPos(5);
    var rs = this._getJointPos(6);
    var lh = this._getJointPos(9);
    var rh = this._getJointPos(10);
    if (ls && rs && lh && rh) {
      var torsoCenter = new THREE.Vector3(
        (ls.x + rs.x + lh.x + rh.x) / 4,
        (ls.y + rs.y + lh.y + rh.y) / 4,
        (ls.z + rs.z + lh.z + rh.z) / 4,
      );
      var torsoW = ls.distanceTo(rs) * 0.9;
      var torsoH = new THREE.Vector3((ls.x + rs.x) / 2, (ls.y + rs.y) / 2, (ls.z + rs.z) / 2)
        .distanceTo(new THREE.Vector3((lh.x + rh.x) / 2, (lh.y + rh.y) / 2, (lh.z + rh.z) / 2));

      var torsoGeom = new THREE.BoxGeometry(torsoW, torsoH, 0.08);
      var torsoMesh = new THREE.Mesh(torsoGeom, bodyMat.clone());
      torsoMesh.position.copy(torsoCenter);
      torsoMesh.userData.isTorso = true;
      this.scene.add(torsoMesh);
      this._limbMeshes.push(torsoMesh);
    }

    // Head sphere
    var nosePos = this._getJointPos(0);
    var neckPos = this._getJointPos(69);
    if (nosePos && neckPos) {
      var headCenter = new THREE.Vector3().lerpVectors(neckPos, nosePos, 0.6);
      var headGeom = new THREE.SphereGeometry(0.08, 16, 16);
      var headMesh = new THREE.Mesh(headGeom, bodyMat.clone());
      headMesh.position.copy(headCenter);
      headMesh.userData.isHead = true;
      this.scene.add(headMesh);
      this._limbMeshes.push(headMesh);
    }
  }

  _createCapsule(a, b, radius, material) {
    var dir = new THREE.Vector3().subVectors(b, a);
    var length = dir.length();
    if (length < 0.001) return null;

    var geom = new THREE.CapsuleGeometry(radius, length, 4, 8);
    var mesh = new THREE.Mesh(geom, material.clone());

    // Position at midpoint
    mesh.position.lerpVectors(a, b, 0.5);

    // Orient along the bone direction
    var up = new THREE.Vector3(0, 1, 0);
    var quat = new THREE.Quaternion().setFromUnitVectors(up, dir.normalize());
    mesh.quaternion.copy(quat);

    return mesh;
  }

  _updateBones() {
    // Remove old bone lines
    for (var i = 0; i < this.boneMeshes.length; i++) {
      this.boneGroup.remove(this.boneMeshes[i]);
      this.boneMeshes[i].geometry.dispose();
    }
    this.boneMeshes = [];

    var boneMat = new THREE.LineBasicMaterial({ color: 0x666677, linewidth: 2 });

    for (var i = 0; i < this.bones.length; i++) {
      var a = this._getJointPos(this.bones[i][0]);
      var b = this._getJointPos(this.bones[i][1]);
      if (!a || !b) continue;

      var geom = new THREE.BufferGeometry().setFromPoints([a, b]);
      var line = new THREE.Line(geom, boneMat);
      this.boneGroup.add(line);
      this.boneMeshes.push(line);
    }
  }

  _updateLimbs() {
    if (!this._limbMeshes) return;

    for (var i = 0; i < this._limbMeshes.length; i++) {
      var mesh = this._limbMeshes[i];
      if (mesh.userData.limbPair) {
        var pair = mesh.userData.limbPair;
        var a = this._getJointPos(pair[0]);
        var b = this._getJointPos(pair[1]);
        if (!a || !b) continue;

        var dir = new THREE.Vector3().subVectors(b, a);
        var length = dir.length();
        if (length < 0.001) continue;

        mesh.position.lerpVectors(a, b, 0.5);
        var up = new THREE.Vector3(0, 1, 0);
        mesh.quaternion.setFromUnitVectors(up, dir.normalize());

        // Update capsule scale to match new bone length
        var origLength = mesh.geometry.parameters ? mesh.geometry.parameters.length : length;
        mesh.scale.y = length / (origLength || length);
      } else if (mesh.userData.isTorso) {
        var ls = this._getJointPos(5);
        var rs = this._getJointPos(6);
        var lh = this._getJointPos(9);
        var rh = this._getJointPos(10);
        if (ls && rs && lh && rh) {
          mesh.position.set(
            (ls.x + rs.x + lh.x + rh.x) / 4,
            (ls.y + rs.y + lh.y + rh.y) / 4,
            (ls.z + rs.z + lh.z + rh.z) / 4,
          );
        }
      } else if (mesh.userData.isHead) {
        var nose = this._getJointPos(0);
        var neck = this._getJointPos(69);
        if (nose && neck) {
          mesh.position.lerpVectors(neck, nose, 0.6);
        }
      }
    }
  }

  _getJointPos(index) {
    var mesh = this.jointMeshes.get(index);
    if (mesh) return mesh.position.clone();

    // Fallback: find in joints array
    for (var i = 0; i < this.joints.length; i++) {
      if (this.joints[i].index === index) {
        return new THREE.Vector3(
          this.joints[i].position[0],
          this.joints[i].position[1],
          this.joints[i].position[2],
        );
      }
    }
    return null;
  }

  // ---------------------------------------------------------------
  // Two-bone IK solver
  // ---------------------------------------------------------------

  _solveIK(rootIdx, midIdx, endIdx, targetPos) {
    var rootMesh = this.jointMeshes.get(rootIdx);
    var midMesh = this.jointMeshes.get(midIdx);
    var endMesh = this.jointMeshes.get(endIdx);
    if (!rootMesh || !midMesh || !endMesh) return;

    var root = rootMesh.position;
    var L1 = root.distanceTo(midMesh.position);
    var L2 = midMesh.position.distanceTo(endMesh.position);
    var totalLength = L1 + L2;

    var toTarget = new THREE.Vector3().subVectors(targetPos, root);
    var dist = toTarget.length();

    // Clamp to reachable range
    if (dist > totalLength * 0.999) {
      dist = totalLength * 0.999;
      toTarget.normalize().multiplyScalar(dist);
      targetPos = new THREE.Vector3().addVectors(root, toTarget);
    }
    if (dist < Math.abs(L1 - L2) * 1.001) {
      dist = Math.abs(L1 - L2) * 1.001;
      toTarget.normalize().multiplyScalar(dist);
      targetPos = new THREE.Vector3().addVectors(root, toTarget);
    }

    // Law of cosines for elbow/knee angle
    var cosAngle = (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist);
    cosAngle = Math.max(-1, Math.min(1, cosAngle));
    var angle = Math.acos(cosAngle);

    // Direction from root to target
    var dir = toTarget.normalize();

    // Pole vector: prefer current mid-joint direction as hint
    var currentMidDir = new THREE.Vector3().subVectors(midMesh.position, root).normalize();
    var poleHint = new THREE.Vector3().crossVectors(dir, currentMidDir);
    if (poleHint.lengthSq() < 0.0001) {
      // Fallback pole: use world Z
      poleHint.set(0, 0, 1);
    }
    var perpendicular = new THREE.Vector3().crossVectors(dir, poleHint).normalize();

    // Mid-joint position
    var midPos = new THREE.Vector3()
      .copy(root)
      .addScaledVector(dir, Math.cos(angle) * L1)
      .addScaledVector(perpendicular, Math.sin(angle) * L1);

    midMesh.position.copy(midPos);
    endMesh.position.copy(targetPos);

    // Sync back to joints array
    this._syncJointPosition(midIdx, midPos);
    this._syncJointPosition(endIdx, targetPos);
  }

  _findIKChain(jointIndex) {
    for (var i = 0; i < this.ikChains.length; i++) {
      var chain = this.ikChains[i];
      if (chain[2] === jointIndex) return chain; // dragging end effector
    }
    return null;
  }

  _syncJointPosition(index, pos) {
    for (var i = 0; i < this.joints.length; i++) {
      if (this.joints[i].index === index) {
        this.joints[i].position = [pos.x, pos.y, pos.z];
        break;
      }
    }
  }

  // ---------------------------------------------------------------
  // Interaction
  // ---------------------------------------------------------------

  _onPointerDown(e) {
    if (e.button !== 0) return; // Left click only

    var rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);

    // Test against draggable joints
    var draggableMeshes = [];
    var self = this;
    this.jointMeshes.forEach(function (mesh) {
      if (mesh.userData.draggable) draggableMeshes.push(mesh);
    });

    var intersects = this.raycaster.intersectObjects(draggableMeshes);
    if (intersects.length > 0) {
      var hit = intersects[0].object;
      this._selectJoint(hit);
      this.isDragging = true;

      // Set up drag plane perpendicular to camera
      var camDir = new THREE.Vector3();
      this.camera.getWorldDirection(camDir);
      this.dragPlane.setFromNormalAndCoplanarPoint(camDir, hit.position);

      // Disable orbit controls while dragging
      this.controls.enabled = false;

      this.renderer.domElement.setPointerCapture(e.pointerId);
      e.preventDefault();
    } else {
      // Clicked empty space — deselect and let orbit controls handle it
      this._selectJoint(null);
      this.controls.enabled = true;
    }
  }

  _onPointerMove(e) {
    if (!this.isDragging || !this.selectedJoint) return;

    var rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);

    var intersection = new THREE.Vector3();
    if (this.raycaster.ray.intersectPlane(this.dragPlane, intersection)) {
      var jointIndex = this.selectedJoint.userData.jointIndex;

      // Check if this joint is an IK end effector
      var chain = this._findIKChain(jointIndex);
      if (chain) {
        this._solveIK(chain[0], chain[1], chain[2], intersection);
      } else {
        // Direct move (FK)
        this.selectedJoint.position.copy(intersection);
        this._syncJointPosition(jointIndex, intersection);
      }

      this._updateBones();
      this._updateLimbs();
    }
  }

  _onPointerUp(e) {
    if (this.isDragging) {
      this.isDragging = false;
      this.controls.enabled = true;
    }
  }

  _selectJoint(mesh) {
    // Deselect previous — dispose old material to prevent GPU leak
    if (this.selectedJoint && this.selectedJoint !== mesh) {
      this.selectedJoint.material.dispose();
      this.selectedJoint.material = this._materials.draggable.clone();
    }

    this.selectedJoint = mesh;

    if (mesh) {
      mesh.material.dispose();
      mesh.material = this._materials.selected.clone();
      var info = this.overlay.querySelector("#poseJointInfo");
      if (info) info.textContent = mesh.userData.name;
    } else {
      var info = this.overlay.querySelector("#poseJointInfo");
      if (info) info.textContent = "Click a joint to select";
    }
  }

  _onKeyDown(e) {
    if (e.code === "Escape") {
      e.preventDefault();
      this.close();
    }
  }

  // ---------------------------------------------------------------
  // Presets
  // ---------------------------------------------------------------

  _applyPreset(name) {
    // Presets define offsets relative to a normalized skeleton
    // These are approximate — good enough for pose reference
    var presets = {
      tpose: function (joints) {
        // Arms out horizontal, legs straight
        var neck = _findJoint(joints, 69);
        if (!neck) return;
        var baseY = neck.position[1];

        _setIf(joints, 5,  [-0.25, baseY, 0]);       // left shoulder
        _setIf(joints, 6,  [0.25, baseY, 0]);         // right shoulder
        _setIf(joints, 7,  [-0.55, baseY, 0]);        // left elbow
        _setIf(joints, 8,  [0.55, baseY, 0]);         // right elbow
        _setIf(joints, 62, [-0.85, baseY, 0]);        // left wrist
        _setIf(joints, 41, [0.85, baseY, 0]);         // right wrist
        _setIf(joints, 9,  [-0.12, baseY - 0.55, 0]); // left hip
        _setIf(joints, 10, [0.12, baseY - 0.55, 0]);  // right hip
        _setIf(joints, 11, [-0.12, baseY - 1.0, 0]);  // left knee
        _setIf(joints, 12, [0.12, baseY - 1.0, 0]);   // right knee
        _setIf(joints, 13, [-0.12, baseY - 1.45, 0]); // left ankle
        _setIf(joints, 14, [0.12, baseY - 1.45, 0]);  // right ankle
      },
      standing: function (joints) {
        var neck = _findJoint(joints, 69);
        if (!neck) return;
        var baseY = neck.position[1];

        _setIf(joints, 5,  [-0.2, baseY - 0.02, 0]);
        _setIf(joints, 6,  [0.2, baseY - 0.02, 0]);
        _setIf(joints, 7,  [-0.22, baseY - 0.3, 0.02]);
        _setIf(joints, 8,  [0.22, baseY - 0.3, 0.02]);
        _setIf(joints, 62, [-0.18, baseY - 0.55, 0.05]);
        _setIf(joints, 41, [0.18, baseY - 0.55, 0.05]);
        _setIf(joints, 9,  [-0.12, baseY - 0.55, 0]);
        _setIf(joints, 10, [0.12, baseY - 0.55, 0]);
        _setIf(joints, 11, [-0.13, baseY - 1.0, 0.02]);
        _setIf(joints, 12, [0.13, baseY - 1.0, 0.02]);
        _setIf(joints, 13, [-0.13, baseY - 1.45, 0]);
        _setIf(joints, 14, [0.13, baseY - 1.45, 0]);
      },
      sitting: function (joints) {
        var neck = _findJoint(joints, 69);
        if (!neck) return;
        var baseY = neck.position[1];

        _setIf(joints, 5,  [-0.2, baseY - 0.02, 0]);
        _setIf(joints, 6,  [0.2, baseY - 0.02, 0]);
        _setIf(joints, 7,  [-0.22, baseY - 0.3, 0.05]);
        _setIf(joints, 8,  [0.22, baseY - 0.3, 0.05]);
        _setIf(joints, 62, [-0.2, baseY - 0.5, 0.12]);
        _setIf(joints, 41, [0.2, baseY - 0.5, 0.12]);
        _setIf(joints, 9,  [-0.12, baseY - 0.55, 0]);
        _setIf(joints, 10, [0.12, baseY - 0.55, 0]);
        _setIf(joints, 11, [-0.18, baseY - 0.6, 0.45]);
        _setIf(joints, 12, [0.18, baseY - 0.6, 0.45]);
        _setIf(joints, 13, [-0.18, baseY - 1.0, 0.55]);
        _setIf(joints, 14, [0.18, baseY - 1.0, 0.55]);
      },
      action: function (joints) {
        var neck = _findJoint(joints, 69);
        if (!neck) return;
        var baseY = neck.position[1];

        _setIf(joints, 5,  [-0.2, baseY - 0.02, 0]);
        _setIf(joints, 6,  [0.2, baseY - 0.02, 0]);
        _setIf(joints, 7,  [-0.45, baseY + 0.15, -0.1]);
        _setIf(joints, 8,  [0.35, baseY - 0.25, 0.15]);
        _setIf(joints, 62, [-0.55, baseY + 0.35, -0.05]);
        _setIf(joints, 41, [0.25, baseY - 0.15, 0.3]);
        _setIf(joints, 9,  [-0.12, baseY - 0.55, 0]);
        _setIf(joints, 10, [0.12, baseY - 0.55, 0]);
        _setIf(joints, 11, [-0.2, baseY - 0.95, 0.15]);
        _setIf(joints, 12, [0.08, baseY - 1.0, -0.1]);
        _setIf(joints, 13, [-0.25, baseY - 1.3, 0.3]);
        _setIf(joints, 14, [0.1, baseY - 1.45, -0.05]);
      },
    };

    function _findJoint(joints, idx) {
      for (var i = 0; i < joints.length; i++) {
        if (joints[i].index === idx) return joints[i];
      }
      return null;
    }

    function _setIf(joints, idx, pos) {
      var j = _findJoint(joints, idx);
      if (j) j.position = pos;
    }

    var fn = presets[name];
    if (!fn) return;

    fn(this.joints);

    // Update Three.js meshes
    var self = this;
    this.jointMeshes.forEach(function (mesh, index) {
      for (var i = 0; i < self.joints.length; i++) {
        if (self.joints[i].index === index) {
          mesh.position.set(
            self.joints[i].position[0],
            self.joints[i].position[1],
            self.joints[i].position[2],
          );
          break;
        }
      }
    });

    this._updateBones();
    this._updateLimbs();
  }

  _resetPose() {
    if (!this.originalPositions) return;

    for (var i = 0; i < this.joints.length; i++) {
      this.joints[i].position = [...this.originalPositions[i]];
    }

    var self = this;
    this.jointMeshes.forEach(function (mesh, index) {
      for (var i = 0; i < self.joints.length; i++) {
        if (self.joints[i].index === index) {
          mesh.position.set(
            self.joints[i].position[0],
            self.joints[i].position[1],
            self.joints[i].position[2],
          );
          break;
        }
      }
    });

    this._updateBones();
    this._updateLimbs();
  }

  // ---------------------------------------------------------------
  // Render loop
  // ---------------------------------------------------------------

  _startLoop() {
    var self = this;
    function loop() {
      self._animId = requestAnimationFrame(loop);
      if (self.controls) self.controls.update();
      if (self.renderer && self.scene && self.camera) {
        self.renderer.render(self.scene, self.camera);
      }
    }
    loop();
  }

  _stopLoop() {
    if (this._animId) {
      cancelAnimationFrame(this._animId);
      this._animId = null;
    }
  }
}

// Expose to global scope for app.js
window.HydraPoseEditor = new PoseEditor();
