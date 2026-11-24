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

  // Preprocessing Verification Elements
  const preprocessCardFile = document.getElementById("preprocess-card-file");
  const preprocessThumbFile = document.getElementById("preprocess-thumb-file");
  const preprocessOrigSize = document.getElementById("preprocess-orig-size");
  const preprocessQualityFile = document.getElementById("preprocess-quality-file");
  const preprocessCrcFile = document.getElementById("preprocess-crc-file");

  const preprocessCardCam = document.getElementById("preprocess-card-cam");
  const preprocessThumbCam = document.getElementById("preprocess-thumb-cam");
  const preprocessQualityCam = document.getElementById("preprocess-quality-cam");

  // Mode Switcher Elements
  const btnToggleUiMode = document.getElementById("btn-toggle-ui-mode");
  const modeIcon = document.getElementById("mode-icon");
  const modeText = document.getElementById("mode-text");
  const exhibitPill = document.getElementById("exhibit-pill");

  // Friendly Results Elements
  const classResultIcon = document.getElementById("class-result-icon");
  const classResultLabel = document.getElementById("class-result-label");
  const classResultSubtitle = document.getElementById("class-result-subtitle");
  const actionGuideIcon = document.getElementById("action-guide-icon");
  const actionGuideText = document.getElementById("action-guide-text");

  // Counter Elements
  const cntTotal = document.getElementById("cnt-total");
  const cntFemale = document.getElementById("cnt-female");
  const cntMale = document.getElementById("cnt-male");
  const pctFemale = document.getElementById("pct-female");
  const pctMale = document.getElementById("pct-male");
  const btnResetCounter = document.getElementById("btn-reset-counter");

  // State: Mode (Default: exhibition)
  let currentUiMode = localStorage.getItem("doc_ui_mode") || "exhibition";

  // State: Counter
  let counterData = {
    total: parseInt(localStorage.getItem("doc_count_total") || "0", 10),
    female: parseInt(localStorage.getItem("doc_count_female") || "0", 10),
    male: parseInt(localStorage.getItem("doc_count_male") || "0", 10)
  };

  function updateCounterUI() {
    if (!cntTotal) return;
    cntTotal.textContent = counterData.total;
    cntFemale.textContent = counterData.female;
    cntMale.textContent = counterData.male;
    const pFemale = counterData.total > 0 ? Math.round((counterData.female / counterData.total) * 100) : 0;
    const pMale = counterData.total > 0 ? Math.round((counterData.male / counterData.total) * 100) : 0;
    if (pctFemale) pctFemale.textContent = pFemale + "%";
    if (pctMale) pctMale.textContent = pMale + "%";

    localStorage.setItem("doc_count_total", counterData.total.toString());
    localStorage.setItem("doc_count_female", counterData.female.toString());
    localStorage.setItem("doc_count_male", counterData.male.toString());
  }

  function applyUiMode(mode) {
    currentUiMode = mode;
    localStorage.setItem("doc_ui_mode", mode);
    document.body.classList.remove("mode-exhibition", "mode-developer");
    document.body.classList.add("mode-" + mode);

    if (mode === "exhibition") {
      if (modeIcon) modeIcon.textContent = "🎪";
      if (modeText) modeText.textContent = "Mode Pameran";
      if (exhibitPill) exhibitPill.textContent = "Mode Pameran";
      if (btnToggleUiMode) btnToggleUiMode.title = "Klik untuk membuka Mode Ahli (Detail Teknis)";
    } else {
      if (modeIcon) modeIcon.textContent = "🎓";
      if (modeText) modeText.textContent = "Mode Ahli";
      if (exhibitPill) exhibitPill.textContent = "Mode Teknis / Ahli";
      if (btnToggleUiMode) btnToggleUiMode.title = "Klik untuk kembali ke Mode Pameran";
    }
  }

  if (btnToggleUiMode) {
    btnToggleUiMode.addEventListener("click", () => {
      const nextMode = currentUiMode === "exhibition" ? "developer" : "exhibition";
      applyUiMode(nextMode);
    });
  }

  if (btnResetCounter) {
    btnResetCounter.addEventListener("click", () => {
      if (confirm("Reset seluruh data hitungan sortir hari ini?")) {
        counterData = { total: 0, female: 0, male: 0 };
        updateCounterUI();
      }
    });
  }

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
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    // Draw cropped center square scaled to 160x160
    ctx.drawImage(video, startX, startY, minDim, minDim, 0, 0, 160, 160);
    const b64 = canvas.toDataURL("image/jpeg", 0.95);

    if (preprocessCardCam && preprocessThumbCam) {
      preprocessCardCam.style.display = "block";
      preprocessThumbCam.src = b64;
      if (preprocessQualityCam) {
        preprocessQualityCam.textContent = "160×160 RGB Tersinkronisasi Viewfinder";
      }
    }
    return b64;
  }

  async function fetchPreprocessPreview(payload) {
    try {
      if (preprocessCardFile) {
        preprocessCardFile.style.display = "block";
        if (preprocessQualityFile) preprocessQualityFile.textContent = "Memproses kalibrasi 160×160...";
      }
      const res = await fetch("/api/preprocess_preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.success) {
        if (preprocessThumbFile) preprocessThumbFile.src = data.preprocessed_image;
        if (preprocessOrigSize) preprocessOrigSize.textContent = `${data.orig_size[0]} × ${data.orig_size[1]} px`;
        if (preprocessQualityFile) {
          preprocessQualityFile.textContent = `${data.quality.status} — ${data.quality.message}`;
          preprocessQualityFile.style.color = data.quality.status === "OPTIMAL" ? "var(--female-color)" : "var(--danger)";
        }
        if (preprocessCrcFile) preprocessCrcFile.textContent = data.checksum;
      }
    } catch (err) {
      console.warn("Gagal mengambil preview preprocessing:", err);
    }
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
      btnSendFile.innerHTML = '<span class="icon">🚀</span> Kirim Ulang ke Grove (Re-Infer)';

      // Auto-Kirim langsung ke Grove Vision AI (NPU) saat file dipilih!
      executeInference({ image: selectedFileBase64 });
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
            btnSendFile.innerHTML = '<span class="icon">🚀</span> Kirim Ulang ke Grove (Re-Infer)';

            // Auto-Kirim langsung ke Grove Vision AI (NPU) saat sampel diklik!
            executeInference({ sample_path: s.path });
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

    // Class Name & Badge & Friendly Elements
    const badge = res.badge_type || "";
    classResultText.className = "class-result " + badge;

    if (badge === "female" || badge === "hen") {
      if (classResultIcon) classResultIcon.textContent = "🐥";
      if (classResultLabel) classResultLabel.textContent = "BETINA (FEMALE)";
      if (classResultSubtitle) classResultSubtitle.textContent = "Kategori: Bibit Ayam Petelur (Layer)";
      if (actionGuideIcon) actionGuideIcon.textContent = "➡️";
      if (actionGuideText) actionGuideText.textContent = "Pintu Sortir: Arahkan ke Jalur KIRI (Kotak Betina)";

      counterData.female++;
      counterData.total++;
      updateCounterUI();
    } else if (badge === "male" || badge === "cock") {
      if (classResultIcon) classResultIcon.textContent = "🐓";
      if (classResultLabel) classResultLabel.textContent = "JANTAN (MALE)";
      if (classResultSubtitle) classResultSubtitle.textContent = "Kategori: Bibit Ayam Pedaging (Broiler)";
      if (actionGuideIcon) actionGuideIcon.textContent = "➡️";
      if (actionGuideText) actionGuideText.textContent = "Pintu Sortir: Arahkan ke Jalur KANAN (Kotak Jantan)";

      counterData.male++;
      counterData.total++;
      updateCounterUI();
    } else {
      if (classResultIcon) classResultIcon.textContent = "❓";
      if (classResultLabel) classResultLabel.textContent = res.label_display || res.class_name || "TIDAK TERDETEKSI";
      if (classResultSubtitle) classResultSubtitle.textContent = "Bulu sayap belum berada tepat di area panduan";
      if (actionGuideIcon) actionGuideIcon.textContent = "⚠️";
      if (actionGuideText) actionGuideText.textContent = "Posisikan bulu sayap anak ayam tepat ke tengah kotak kamera";
    }

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
      if (preprocessThumbFile) preprocessThumbFile.src = res.preprocessed_image;
      if (preprocessCardFile) preprocessCardFile.style.display = "block";
      if (preprocessQualityFile && res.quality) {
        preprocessQualityFile.textContent = `${res.quality.status} — ${res.quality.message}`;
        preprocessQualityFile.style.color = res.quality.status === "OPTIMAL" ? "var(--female-color)" : "var(--danger)";
      }
      if (preprocessCrcFile && res.checksum) {
        preprocessCrcFile.textContent = res.checksum;
      }
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
  applyUiMode(currentUiMode);
  updateCounterUI();
  loadSerialPorts();
  loadPresetSamples();
});

