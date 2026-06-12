const video = document.getElementById("video");
const startBtn = document.getElementById("startBtn");
const captureBtn = document.getElementById("captureBtn");
const canvas = document.getElementById("canvas");
const capturedImage = document.getElementById("capturedImage");
const cameraStatus = document.getElementById("cameraStatus");
const predictStatus = document.getElementById("predictStatus");
const emotionNameEl = document.getElementById("emotionName");
const confidenceValueEl = document.getElementById("confidenceValue");
const confidenceArc = document.getElementById("confidenceArc");
const emotionBarsContainer = document.getElementById("emotionBars");
const detectionLog = document.getElementById("detectionLog");
const clearLogBtn = document.getElementById("clearLogBtn");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const emotionList = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const CIRCUMFERENCE = 282.74;

let logEntries = [];

function loadLog() {
    const saved = localStorage.getItem('emotionDetectionLog');
    if (saved) {
        try {
            logEntries = JSON.parse(saved);
        } catch(e) {
            logEntries = [];
        }
    }
    renderLog();
}
function saveLog() {
    localStorage.setItem('emotionDetectionLog', JSON.stringify(logEntries.slice(-20)));
}
function renderLog() {
    if (!detectionLog) return;
    
    if (logEntries.length === 0) {
        detectionLog.innerHTML = '<div class="log-empty">Chưa có kết quả nhận diện nào</div>';
        return;
    }
    detectionLog.innerHTML = logEntries.slice().reverse().map(entry => `
        <div class="log-item">
            <div>
                <strong class="log-emotion">${entry.emotion}</strong>
                <div class="log-time">${entry.time}</div>
            </div>
            <div class="log-confidence">${entry.confidence}%</div>
        </div>
    `).join('');
}
function addLogEntry(emotion, confidence) {
    const now = new Date();
    const timeStr = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`;
    
    logEntries.unshift({ emotion, confidence, time: timeStr });
    if (logEntries.length > 20) logEntries.pop();
    saveLog();
    renderLog();
}
function clearLog() {
    logEntries = [];
    saveLog();
    renderLog();
}
function resetUI() {
    if (emotionNameEl) emotionNameEl.textContent = 'Chưa nhận diện';
    if (confidenceValueEl) confidenceValueEl.textContent = '0';
    
    if (confidenceArc) {
        confidenceArc.style.strokeDashoffset = CIRCUMFERENCE;
    }
    if (predictStatus) {
        predictStatus.textContent = 'Chờ dự đoán';
        predictStatus.style.background = '#f1f5f9';
        predictStatus.style.color = '#475569';
    }
    renderEmotionBars(null);
}
function renderEmotionBars(predictions) {
    if (!emotionBarsContainer) return;
    
    const barsHtml = emotionList.map(emotion => {
        let confidence = 0;
        if (predictions) {
            const pred = predictions.find(p => p.label === emotion);
            if (pred) confidence = Math.round(pred.confidence * 100);
        }
        return `
            <div class="emotion-bar-item">
                <div class="emotion-bar-label">
                    <span class="emotion-name-label">${emotion}</span>
                    <span class="emotion-percent">${confidence}%</span>
                </div>
                <div class="emotion-bar-bg">
                    <div class="emotion-bar-fill" style="width: ${confidence}%;"></div>
                </div>
            </div>
        `;
    }).join('');
    emotionBarsContainer.innerHTML = barsHtml;
}
function updatePredictionUI(predictions) {
    if (!predictions || predictions.length === 0) {
        resetUI();
        return;
    }
    let maxIdx = 0;
    for (let i = 1; i < predictions.length; i++) {
        if (predictions[i].confidence > predictions[maxIdx].confidence) {
            maxIdx = i;
        }
    }
    const topEmotion = predictions[maxIdx].label;
    const topConfidence = Math.round(predictions[maxIdx].confidence * 100);
    if (emotionNameEl) emotionNameEl.textContent = topEmotion;
    if (confidenceValueEl) confidenceValueEl.textContent = topConfidence;
    if (confidenceArc) {
        const offset = CIRCUMFERENCE - (topConfidence / 100) * CIRCUMFERENCE;
        confidenceArc.style.strokeDashoffset = offset;
    }
    if (predictStatus) {
        predictStatus.textContent = 'Đã nhận diện';
        predictStatus.style.background = '#d1fae5';
        predictStatus.style.color = '#065f46';
    }
    renderEmotionBars(predictions);
    addLogEntry(topEmotion, topConfidence);
}
function setLoadingState(isLoading) {
    if (!predictStatus) return;
    if (isLoading) {
        predictStatus.textContent = 'Đang xử lý...';
        predictStatus.style.background = '#fef3c7';
        predictStatus.style.color = '#92400e';
        if (captureBtn) {
            captureBtn.disabled = true;
            captureBtn.textContent = 'ĐANG XỬ LÝ...';
        }
    } else {
        if (captureBtn) {
            captureBtn.disabled = false;
            captureBtn.textContent = 'NHẬN DIỆN';
        }
    }
}
function setCameraStatus(isConnected) {
    if (!cameraStatus) return;
    
    if (isConnected) {
        cameraStatus.textContent = 'Đã kết nối';
        cameraStatus.style.background = '#d1fae5';
        cameraStatus.style.color = '#065f46';
        if (startBtn) startBtn.disabled = true;
        if (captureBtn) captureBtn.disabled = false;
        if (videoPlaceholder) videoPlaceholder.style.display = 'none';
        video.style.display = 'block';
    } else {
        cameraStatus.textContent = 'Chưa kết nối';
        cameraStatus.style.background = '#f1f5f9';
        cameraStatus.style.color = '#475569';
        if (startBtn) startBtn.disabled = false;
        if (captureBtn) captureBtn.disabled = true;
        if (videoPlaceholder) videoPlaceholder.style.display = 'flex';
        video.style.display = 'none';
        resetUI();
    }
}
let currentStream = null;
async function startCamera() {
    try {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        currentStream = stream;
        video.srcObject = stream;
        setCameraStatus(true);
    } catch (error) {
        console.error("Camera error:", error);
        alert("Không thể mở camera. Vui lòng kiểm tra quyền truy cập.");
        if (cameraStatus) {
            cameraStatus.textContent = 'Lỗi camera';
            cameraStatus.style.background = '#fee2e2';
            cameraStatus.style.color = '#991b1b';
        }
    }
}
function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    video.srcObject = null;
    setCameraStatus(false);
}
async function captureAndDetect() {
    if (!currentStream || !video.videoWidth) {
        alert('Vui lòng bật camera trước!');
        return;
    }
    setLoadingState(true);
    try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL("image/png");
        if (capturedImage) {
            capturedImage.src = imageData;
            capturedImage.style.display = "block";
        }
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append("image", blob, "capture.png");
            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();
                const emotionElement = document.getElementById("emotion");
                const confidenceElement = document.getElementById("confidence");
                if (data.emotion === "Không tìm thấy khuôn mặt" || data.confidence === 0 || data.confidence < 5) {
                    // Không có khuôn mặt
                    if (emotionElement) emotionElement.innerText = "Không tìm thấy khuôn mặt";
                    if (confidenceElement) confidenceElement.innerText = "Độ tin cậy: --";
                    if (emotionNameEl) emotionNameEl.textContent = "Không thấy khuôn mặt";
                    if (confidenceValueEl) confidenceValueEl.textContent = "0";
                    if (confidenceArc) {
                        confidenceArc.style.strokeDashoffset = CIRCUMFERENCE;
                    }
                    if (predictStatus) {
                        predictStatus.textContent = "Không tìm thấy khuôn mặt";
                        predictStatus.style.background = "#fee2e2";
                        predictStatus.style.color = "#991b1b";
                    }
                    renderEmotionBars(null);
                } else {
                    if (emotionElement) emotionElement.innerText = data.emotion;
                    if (confidenceElement) confidenceElement.innerText = "Độ tin cậy: " + data.confidence + "%";
                    const predictions = emotionList.map(emotion => ({
                        label: emotion,
                        confidence: emotion === data.emotion ? data.confidence / 100 : 0.01
                    }));
                    updatePredictionUI(predictions);
                }
            } catch (err) {
                console.error("Fetch error:", err);
                if (predictStatus) {
                    predictStatus.textContent = 'Lỗi kết nối server';
                    predictStatus.style.background = '#fee2e2';
                    predictStatus.style.color = '#991b1b';
                }
            }
            setLoadingState(false);
        }, 'image/png');
    } catch (error) {
        console.error("Capture error:", error);
        setLoadingState(false);
        alert("Lỗi khi chụp ảnh!");
    }
}
if (startBtn) startBtn.addEventListener('click', startCamera);
if (captureBtn) captureBtn.addEventListener('click', captureAndDetect);
if (clearLogBtn) clearLogBtn.addEventListener('click', clearLog);
window.addEventListener('beforeunload', () => {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
});
if (videoPlaceholder) videoPlaceholder.style.display = 'flex';
video.style.display = 'none';
if (captureBtn) captureBtn.disabled = true;
loadLog();
resetUI();
console.log('Webcam handler initialized - ready for detection!');