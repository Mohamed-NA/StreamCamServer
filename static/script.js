document.addEventListener('DOMContentLoaded', () => {
    const host     = window.location.host;
    const socket   = io(`wss://${host}`);
    const video       = document.getElementById('video');
    const serverCanvas= document.getElementById('server-frame');
    const serverCtx   = serverCanvas.getContext('2d');

    // UI refs
    const dot          = document.getElementById('status-dot');
    const statusText   = document.getElementById('status-text');
    const connBadge    = document.getElementById('conn-badge');
    const recBadge     = document.getElementById('rec-badge');
    const pauseBtn     = document.getElementById('pause-btn');
    const pauseIcon    = document.getElementById('pause-icon');
    const overlay      = document.getElementById('overlay');
    const overlayIcon  = document.getElementById('overlay-icon');
    const overlayTitle = document.getElementById('overlay-title');
    const overlayDesc  = document.getElementById('overlay-desc');
    const frameCountEl = document.getElementById('frame-count');
    const fpsDisplay   = document.getElementById('fps-display');
    const stateDisplay = document.getElementById('state-display');
    const fpsSlider    = document.getElementById('fps-slider');
    const fpsLabel     = document.getElementById('fps-label');
    const qualitySlider= document.getElementById('quality-slider');
    const qualityLabel = document.getElementById('quality-label');
    const qualityDisp  = document.getElementById('quality-display');
    const resDisplay   = document.getElementById('res-display');
    const logEl        = document.getElementById('log');

    // State
    let frameCount  = 0;
    let paused      = false;
    let fps         = 15;
    let quality     = 0.8;
    let canvasW     = 640;
    let canvasH     = 480;
    let fpsFrames   = 0;
    let fpsTimer    = Date.now();
    let waiting     = false;   // ack-based flow control
    let loopHandle  = null;
    let canvas      = document.createElement('canvas');
    let ctx         = canvas.getContext('2d');

    function log(msg, type = 'info') {
        const now  = new Date();
        const time = now.toTimeString().slice(0, 8);
        const el   = document.createElement('div');
        el.className = `log-entry ${type}`;
        el.innerHTML = `<span class="log-time">${time}</span><span class="log-msg">${msg}</span>`;
        logEl.prepend(el);
        // keep log short
        while (logEl.children.length > 30) logEl.lastChild.remove();
    }

    function setCanvasSize(w, h) {
        canvasW = canvas.width  = w;
        canvasH = canvas.height = h;
        resDisplay.textContent = `${w} × ${h}`;
    }

    setCanvasSize(640, 480);

    function scheduleNext() {
        loopHandle = setTimeout(sendFrame, 1000 / fps);
    }

    function sendFrame() {
        if (paused || video.readyState < 2 || waiting) {
            scheduleNext();
            return;
        }
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvasW, 0, canvasW, canvasH);
        ctx.restore();
        const frame = canvas.toDataURL('image/jpeg', quality);
        waiting = true;
        socket.emit('video_frame', frame);
        frameCountEl.textContent = ++frameCount;

        // measure real fps
        fpsFrames++;
        const now = Date.now();
        if (now - fpsTimer >= 1000) {
            fpsDisplay.textContent = fpsFrames;
            fpsFrames = 0;
            fpsTimer  = now;
        }
    }

    // ── Camera ──
    function showOverlay(icon, title, desc, isError = false) {
        overlayIcon.className = 'overlay-icon' + (isError ? ' error' : '');
        overlayIcon.innerHTML = icon;
        overlayTitle.textContent = title;
        overlayDesc.textContent  = desc;
        overlay.classList.remove('hidden');
        video.classList.add('hidden');
        recBadge.classList.remove('visible');
        pauseBtn.classList.remove('visible');
        stateDisplay.textContent = isError ? 'Error' : '…';
    }

    function hideOverlay() {
        overlay.classList.add('hidden');
        recBadge.classList.add('visible');
        pauseBtn.classList.add('visible');
        stateDisplay.textContent = 'Waiting';
    }

    const camIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M15 10l4.55-2.27A1 1 0 0121 8.67v6.66a1 1 0 01-1.45.9L15 14"/>
        <rect x="1" y="8" width="14" height="10" rx="2"/>
    </svg>`;

    const errIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>`;

    showOverlay(camIcon, 'Requesting camera', 'Please allow camera access to begin streaming.');

    navigator.mediaDevices.getUserMedia({ video: { width: canvasW, height: canvasH } })
        .then(stream => {
            video.srcObject = stream;
            video.play();
            video.addEventListener('playing', () => {
                hideOverlay();
                scheduleNext();
                log('Camera started', 'success');
            });
        })
        .catch(err => {
            showOverlay(errIcon, 'Camera unavailable', err.message || 'Could not access the camera.', true);
            log('Camera error: ' + err.message, 'error');
        });

    // ── Pause / resume ──
    pauseBtn.addEventListener('click', () => {
        paused = !paused;
        pauseIcon.innerHTML = paused
            ? `<polygon points="5,3 19,12 5,21"/>`   // play icon
            : `<rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/>`;
        recBadge.style.opacity = paused ? '0.3' : '';
        stateDisplay.textContent = paused ? 'Paused' : 'Live';
        log(paused ? 'Stream paused' : 'Stream resumed', 'info');
    });

    // ── FPS slider ──
    fpsSlider.addEventListener('input', () => {
        fps = parseInt(fpsSlider.value);
        fpsLabel.textContent = `${fps} fps`;
    });

    // ── Quality slider ──
    qualitySlider.addEventListener('input', () => {
        quality = parseInt(qualitySlider.value) / 100;
        qualityLabel.textContent = `${qualitySlider.value}%`;
        qualityDisp.textContent  = `${qualitySlider.value}%`;
    });

    // ── Resolution buttons ──
    document.getElementById('res-group').addEventListener('click', e => {
        const btn = e.target.closest('.toggle-btn');
        if (!btn) return;
        document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const w = parseInt(btn.dataset.w);
        const h = parseInt(btn.dataset.h);
        setCanvasSize(w, h);
        log(`Resolution → ${w}×${h}`, 'info');
    });

    // ── Received frame from server ──
    socket.on('server_frame', (data) => {
        const img = new Image();
        img.onload = () => {
            const cw = serverCanvas.clientWidth;
            const ch = serverCanvas.clientHeight;
            serverCanvas.width  = cw;
            serverCanvas.height = ch;

            // letter-box: scale to fill width, center vertically
            const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
            const dw = img.naturalWidth  * scale;
            const dh = img.naturalHeight * scale;
            const dx = (cw - dw) / 2;
            const dy = (ch - dh) / 2;

            serverCtx.clearRect(0, 0, cw, ch);
            serverCtx.drawImage(img, dx, dy, dw, dh);

            if (serverCanvas.classList.contains('hidden')) {
                serverCanvas.classList.remove('hidden');
                stateDisplay.textContent = 'Live';
            }
        };
        img.src = data;

        // unblock send loop
        waiting = false;
        scheduleNext();
    });

    // ── Detection results ──
    socket.on('detections', (results) => {
        const panel = document.getElementById('detection-panel');
        const withMask    = results.filter(r => r.label === 'With Mask').length;
        const withoutMask = results.filter(r => r.label === 'Without Mask').length;

        document.getElementById('det-faces').textContent  = results.length;
        document.getElementById('det-masked').textContent = withMask;
        document.getElementById('det-no-mask').textContent = withoutMask;

        const list = document.getElementById('det-list');
        list.innerHTML = results.map((r, i) =>
            `<div class="det-item ${r.label === 'With Mask' ? 'safe' : 'danger'}">
                <span>Face ${i + 1}</span>
                <span>${r.label} &nbsp; ${(r.confidence * 100).toFixed(0)}%</span>
            </div>`
        ).join('') || '<div class="det-empty">No faces detected</div>';
    });

    // ── Socket ──
    socket.on('connect', () => {
        dot.className = 'dot connected';
        statusText.textContent = 'Connected';
        connBadge.classList.add('live');
        log('Socket connected', 'success');
    });

    socket.on('disconnect', () => {
        dot.className = 'dot disconnected';
        statusText.textContent = 'Disconnected';
        connBadge.classList.remove('live');
        log('Socket disconnected', 'error');
    });
});
