// main.js — Logika Antarmuka Dashboard DOC Sexing AI Monitor

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const portSelect = document.getElementById("serial-port-select");
  const btnRefreshPorts = document.getElementById("btn-refresh-ports");
  const btnPing = document.getElementById("btn-ping");
  const pingIndicator = document.getElementById("ping-indicator");

  // Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Camera elements
  const btnToggleCam = document.getElementById("btn-toggle-cam");
  const btnCaptureInfer = document.getElementById("btn-capture-infer");
  const chkAutoInfer = document.getElementById("chk-auto-infer");
  const video = document.getElementById("webcam-video");
  const canvas = document.getElementById("capture-canvas");
  const cameraPlaceholder = document.getElementById("camera-placeholder");
  const cameraOverlay = document.querySelector(".camera-overlay");

  // File elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const filePreviewImg = document.getElementById("file-preview-img");
  const dropzoneContent = document.querySelector(".dropzone-content");
  const presetContainer = document.getElementById("preset-container");
  const btnSendFile = document.getElementById("btn-send-file");

  // Results elements
  const statusPill = document.getElementById("status-pill");
  const classResultText = document.getElementById("class-result-text");
  const confidenceVal = document.getElementById("confidence-val");
  const confidenceBar = document.getElementById("confidence-bar");
  const valNpuMs = document.getElementById("val-npu-ms");
  const valTotalMs = document.getElementById("val-total-ms");
  const valI2cStatus = document.getElementById("val-i2c-status");
  const valChecksum = document.getElementById("val-checksum");
  const imgNpuPreview = document.getElementById("img-npu-preview");
  const rawLogsTerminal = document.getElementById("raw-logs-terminal");

  // State
  let mediaStream = null;
  let autoInferTimer = null;
  let isInferring = false;
  let selectedFileBase64 = null;
  let selectedSamplePath = null;

  // ==========================================
  // 1. SERIAL PORT & HARDWARE CONNECTION
  // ==========================================
  async function loadSerialPorts() {
    try {
      const res = await fetch("/api/ports");
      const data = await res.json();
      portSelect.innerHTML = "";

      if (data.success && data.ports.length > 0) {
        data.ports.forEach(p => {
          const opt = document.createElement("option");
          opt.value = p.device;
          opt.textContent = `${p.device} (${p.description || "USB Serial"})`;
          if (p.is_recommended) {
            opt.textContent = "⭐ " + opt.textContent;
            opt.selected = true;
          }
          portSelect.appendChild(opt);
        });
      } else {
        const opt = document.createElement("option");
        opt.value = "/dev/ttyACM0";
        opt.textContent = "/dev/ttyACM0 (Default)";
        portSelect.appendChild(opt);
      }
    } catch (err) {
      console.error("Gagal memuat serial ports:", err);
    }
  }

  async function pingGroveHardware() {
    const port = portSelect.value;
    pingIndicator.className = "status-dot";
    btnPing.disabled = true;

    try {
      const res = await fetch("/api/ping", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ port })
      });
      const data = await res.json();

      if (data.success) {
        pingIndicator.className = "status-dot online";
        statusPill.textContent = "Grove Terhubung (" + data.status + ")";
        statusPill.className = "status-pill ready";
      } else {
        pingIndicator.className = "status-dot offline";
        statusPill.textContent = "Grove Offline";
        statusPill.className = "status-pill";
        alert("Gagal menghubungi Grove Vision AI V2 di " + port + ":\n" + (data.error || "Timeout"));
      }
    } catch (err) {
      pingIndicator.className = "status-dot offline";
      console.error("Ping error:", err);
    } finally {
      btnPing.disabled = false;
    }
  }

  btnRefreshPorts.addEventListener("click", loadSerialPorts);
  btnPing.addEventListener("click", pingGroveHardware);

  // ==========================================
  // 2. TAB SWITCHER
  // ==========================================
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");

      // Matikan auto infer jika berpindah dari tab kamera
      if (targetId !== "tab-camera" && autoInferTimer) {
        chkAutoInfer.checked = false;
        clearInterval(autoInferTimer);
        autoInferTimer = null;
      }
    });
  });

  // ==========================================
  // 3. MODE 1: LIVE WEBCAM CAMERA
  // ==========================================
  async function startCamera() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "environment"
        },
        audio: false
      });

      video.srcObject = mediaStream;
      video.style.display = "block";
      cameraPlaceholder.style.display = "none";
      cameraOverlay.style.display = "flex";

      btnToggleCam.innerHTML = '<span class="icon">⏹️</span> Tutup Kamera';
      btnToggleCam.className = "btn btn-secondary";
      btnCaptureInfer.disabled = false;
      chkAutoInfer.disabled = false;
    } catch (err) {
      alert("Tidak dapat mengakses kamera:\n" + err.message + "\nPastikan izin kamera diizinkan di browser.");
      console.error("Camera access error:", err);
    }
  }

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
    video.srcObject = null;
    video.style.display = "none";
    cameraPlaceholder.style.display = "flex";
    cameraOverlay.style.display = "none";

    btnToggleCam.innerHTML = '<span class="icon">▶️</span> Buka Kamera';
    btnToggleCam.className = "btn btn-primary";
    btnCaptureInfer.disabled = true;
    chkAutoInfer.disabled = true;
    chkAutoInfer.checked = false;

    if (autoInferTimer) {
      clearInterval(autoInferTimer);
      autoInferTimer = null;
    }
  }

  btnToggleCam.addEventListener("click", () => {
    if (mediaStream) {
      stopCamera();
    } else {
      startCamera();
    }
  });

  // Ambil frame dari video, crop square, gambar ke canvas 160x160
  function captureSquare160FromVideo() {
    const vWidth = video.videoWidth || 640;
    const vHeight = video.videoHeight || 480;
    const minDim = Math.min(vWidth, vHeight);

    const startX = (vWidth - minDim) / 2;
    const startY = (vHeight - minDim) / 2;

    const ctx = canvas.getContext("2d");
    // Draw cropped center square scaled to 160x160
    ctx.drawImage(video, startX, startY, minDim, minDim, 0, 0, 160, 160);
    return canvas.toDataURL("image/jpeg", 0.92);
  }

  async function executeInference(payload) {
    if (isInferring) return;
    isInferring = true;
    statusPill.textContent = "Mengirim ke NPU...";
    statusPill.className = "status-pill busy";
    btnCaptureInfer.disabled = true;
    btnSendFile.disabled = true;

    try {
      payload.port = portSelect.value;
      const res = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      displayInferenceResult(data);
    } catch (err) {
      console.error("Inference request failed:", err);
      statusPill.textContent = "Error Komunikasi";
      statusPill.className = "status-pill";
      alert("Gagal mengirim data ke server UI:\n" + err.message);
    } finally {
      isInferring = false;
      if (mediaStream) btnCaptureInfer.disabled = false;
      if (selectedFileBase64 || selectedSamplePath) btnSendFile.disabled = false;
    }
  }

  btnCaptureInfer.addEventListener("click", () => {
    if (!mediaStream) return;
    const b64 = captureSquare160FromVideo();
    executeInference({ image: b64 });
  });

  chkAutoInfer.addEventListener("change", (e) => {
    if (e.target.checked) {
      autoInferTimer = setInterval(() => {
        if (!isInferring && mediaStream) {
          const b64 = captureSquare160FromVideo();
          executeInference({ image: b64 });
        }
      }, 2000);
    } else {
      if (autoInferTimer) {
        clearInterval(autoInferTimer);
        autoInferTimer = null;
      }
    }
  });

  // ==========================================
  // 4. MODE 2: FILE UPLOAD & PRESET SAMPLES
  // ==========================================
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--primary)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "var(--card-border)";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--card-border)";
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      alert("Silakan pilih file gambar yang valid (JPG/PNG).");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      selectedFileBase64 = e.target.result;
      selectedSamplePath = null;

      // Update preview UI
      filePreviewImg.src = selectedFileBase64;
      filePreviewImg.style.display = "block";
      dropzoneContent.style.display = "none";

      // Reset preset button active state
      document.querySelectorAll(".btn-sample").forEach(b => b.classList.remove("active"));
      btnSendFile.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  async function loadPresetSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      presetContainer.innerHTML = "";

      if (data.success && data.samples.length > 0) {
        data.samples.forEach(s => {
          const btn = document.createElement("button");
          btn.className = "btn-sample";
          btn.textContent = s.name;
          btn.addEventListener("click", () => {
            document.querySelectorAll(".btn-sample").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            selectedSamplePath = s.path;
            selectedFileBase64 = null;

            // Load sample preview
            filePreviewImg.src = `/api/sample_image?path=${encodeURIComponent(s.path)}`;
            filePreviewImg.style.display = "block";
            dropzoneContent.style.display = "none";

            btnSendFile.disabled = false;
          });
          presetContainer.appendChild(btn);
        });
      } else {
        presetContainer.textContent = "Tidak ada sampel preset ditemukan.";
      }
    } catch (err) {
      console.error("Gagal memuat sampel preset:", err);
      presetContainer.textContent = "Gagal memuat sampel.";
    }
  }

  btnSendFile.addEventListener("click", () => {
    if (selectedSamplePath) {
      executeInference({ sample_path: selectedSamplePath });
    } else if (selectedFileBase64) {
      executeInference({ image: selectedFileBase64 });
    }
  });

  // ==========================================
  // 5. DISPLAY INFERENCE RESULTS
  // ==========================================
  function displayInferenceResult(res) {
    if (!res.success) {
      statusPill.textContent = "Gagal Inferensi";
      statusPill.className = "status-pill";
      classResultText.textContent = "ERROR: " + (res.error || "Gagal");
      classResultText.className = "class-result";

      if (res.logs && res.logs.length > 0) {
        rawLogsTerminal.textContent = res.logs.join("\n");
      }
      return;
    }

    // Success State
    statusPill.textContent = "Inferensi Sukses";
    statusPill.className = "status-pill ready";

    // Class Name & Badge
    classResultText.textContent = res.label_display || res.class_name;
    classResultText.className = "class-result " + (res.badge_type || "");

    // Confidence
    const conf = res.confidence || 0.0;
    confidenceVal.textContent = conf.toFixed(1) + "%";
    confidenceBar.style.width = Math.min(Math.max(conf, 0), 100) + "%";

    if (res.badge_type === "female" || res.badge_type === "hen") {
      confidenceBar.style.backgroundColor = "var(--female-color)";
    } else if (res.badge_type === "male" || res.badge_type === "cock") {
      confidenceBar.style.backgroundColor = "var(--male-color)";
    } else {
      confidenceBar.style.backgroundColor = "var(--warning)";
    }

    // Metrics
    valNpuMs.textContent = (res.infer_ms !== undefined ? res.infer_ms : "-") + " ms";
    valTotalMs.textContent = (res.total_ms || "-") + " ms";
    valI2cStatus.textContent = res.i2c_status || "-";
    valI2cStatus.style.color = res.i2c_status === "ACK_OK" ? "var(--success)" : "var(--danger)";
    valChecksum.textContent = res.checksum || "-";

    // Image Preprocessed Preview
    if (res.preprocessed_image) {
      imgNpuPreview.src = res.preprocessed_image;
    }

    // Logs
    if (res.logs && res.logs.length > 0) {
      rawLogsTerminal.textContent = res.logs.join("\n");
    } else if (res.raw_json) {
      rawLogsTerminal.textContent = JSON.stringify(res.raw_json, null, 2);
    }
  }

  // ==========================================
  // INITIALIZATION
  // ==========================================
  loadSerialPorts();
  loadPresetSamples();
});

