// Frontend Application Logic for Smart Waste Classification System

let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    initDragAndDrop();
    initFileInput();
    loadHistory();
    loadGuidanceReference();
});

// Navigation Tab Switcher
function switchTab(tabName) {
    const views = ['classifier', 'guidance', 'admin'];
    views.forEach(v => {
        const viewEl = document.getElementById(`view-${v}`);
        const tabEl = document.getElementById(`tab-${v}`);
        if (v === tabName) {
            viewEl.classList.remove('hidden');
            tabEl.classList.add('text-emerald-400', 'bg-slate-800', 'shadow');
            tabEl.classList.remove('text-slate-400');
        } else {
            viewEl.classList.add('hidden');
            tabEl.classList.remove('text-emerald-400', 'bg-slate-800', 'shadow');
            tabEl.classList.add('text-slate-400');
        }
    });

    if (tabName === 'admin') {
        loadAdminAnalytics();
    } else if (tabName === 'guidance') {
        loadGuidanceReference();
    }
}

// Drag & Drop Setup
function initDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
}

function initFileInput() {
    const fileInput = document.getElementById('file-input');
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    // Validate file type (FR-1)
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showAlert('Invalid file format. Only JPG and PNG images are allowed.', 'error');
        return;
    }

    // Validate size limit 10MB (FR-1)
    if (file.size > 10 * 1024 * 1024) {
        showAlert('File size exceeds maximum allowed limit of 10MB.', 'error');
        return;
    }

    hideAlert();
    selectedFile = file;

    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('image-preview').src = e.target.result;
        document.getElementById('preview-filename').textContent = file.name;
        document.getElementById('upload-prompt').classList.add('hidden');
        document.getElementById('preview-container').classList.remove('hidden');
        document.getElementById('btn-classify').disabled = false;
        document.getElementById('btn-reset').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('upload-prompt').classList.remove('hidden');
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('btn-classify').disabled = true;
    document.getElementById('btn-reset').classList.add('hidden');

    // Reset results view
    document.getElementById('result-placeholder').classList.remove('hidden');
    document.getElementById('result-loading').classList.add('hidden');
    document.getElementById('result-card').classList.add('hidden');
    hideAlert();
}

async function classifyImage() {
    if (!selectedFile) return;

    // Show loading state
    document.getElementById('result-placeholder').classList.add('hidden');
    document.getElementById('result-card').classList.add('hidden');
    document.getElementById('result-loading').classList.remove('hidden');
    document.getElementById('btn-classify').disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/api/classify', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        document.getElementById('result-loading').classList.add('hidden');
        document.getElementById('btn-classify').disabled = false;

        if (!response.ok || !data.success) {
            showAlert(data.error || 'Classification request failed.', 'error');
            document.getElementById('result-placeholder').classList.remove('hidden');
            return;
        }

        renderResult(data);
        loadHistory(); // Refresh history list
    } catch (err) {
        document.getElementById('result-loading').classList.add('hidden');
        document.getElementById('btn-classify').disabled = false;
        document.getElementById('result-placeholder').classList.remove('hidden');
        showAlert(`Network error: ${err.message}`, 'error');
    }
}

function renderResult(data) {
    document.getElementById('result-card').classList.remove('hidden');

    // Category & Badge
    document.getElementById('res-category').textContent = data.category;
    const badge = document.getElementById('res-category-badge');
    badge.textContent = data.guidance.display_name;

    // Reset badge classes
    badge.className = 'px-3 py-1 rounded-full text-xs font-bold shadow-sm ' + getBadgeClass(data.category);

    // Confidence & Meter
    const conf = data.confidence;
    document.getElementById('res-confidence-text').textContent = `${conf}%`;
    document.getElementById('res-confidence-percentage').textContent = `${conf}%`;
    
    const bar = document.getElementById('res-confidence-bar');
    bar.style.width = `${conf}%`;
    if (conf >= 80) {
        bar.className = 'h-full rounded-full bg-emerald-500 transition-all duration-700';
    } else if (conf >= 60) {
        bar.className = 'h-full rounded-full bg-yellow-500 transition-all duration-700';
    } else {
        bar.className = 'h-full rounded-full bg-red-500 transition-all duration-700';
    }

    // Low confidence alert (FR-6: <60%)
    const lowConfAlert = document.getElementById('res-low-confidence-alert');
    if (data.is_low_confidence) {
        lowConfAlert.classList.remove('hidden');
    } else {
        lowConfAlert.classList.add('hidden');
    }

    // Guidance Details
    const g = data.guidance;
    document.getElementById('res-bin-type').textContent = g.bin_type;
    document.getElementById('res-bin-color').textContent = g.bin_color;
    
    const colorInd = document.getElementById('res-bin-color-indicator');
    colorInd.style.backgroundColor = g.color;

    document.getElementById('res-guidance-text').textContent = g.guidance;

    // Examples list
    const exList = document.getElementById('res-examples-list');
    exList.innerHTML = g.examples.map(ex => 
        `<span class="px-2.5 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700">${ex}</span>`
    ).join('');
}

function getBadgeClass(category) {
    switch (category) {
        case 'Plastic': return 'badge-plastic';
        case 'Paper': return 'badge-paper';
        case 'Metal': return 'badge-metal';
        case 'Glass': return 'badge-glass';
        case 'Organic': return 'badge-organic';
        default: return 'badge-other';
    }
}

// Load Session History
async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        if (!data.success) return;

        const container = document.getElementById('history-list');
        if (data.history.length === 0) {
            container.innerHTML = `<p class="text-xs text-slate-500 col-span-full py-4 text-center">No past session items logged yet.</p>`;
            return;
        }

        container.innerHTML = data.history.map(item => `
            <div class="bg-slate-900/80 rounded-xl p-3 border border-slate-700/60 flex items-center space-x-3 hover:border-slate-600 transition-all">
                <img src="${item.image_url}" class="w-12 h-12 rounded-lg object-cover border border-slate-700" alt="Scanned item">
                <div class="flex-1 min-w-0">
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-bold text-white truncate">${item.category}</span>
                        <span class="text-[10px] font-bold ${item.is_low_confidence ? 'text-amber-400' : 'text-emerald-400'}">${item.confidence}%</span>
                    </div>
                    <p class="text-[11px] text-slate-400 truncate mt-0.5">${item.original_filename}</p>
                    <span class="text-[10px] text-slate-500 block mt-0.5">${item.created_at || ''}</span>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error('History load error:', e);
    }
}

// Load Guidance Reference
async function loadGuidanceReference() {
    try {
        const res = await fetch('/api/guidance');
        const data = await res.json();
        if (!data.success) return;

        const container = document.getElementById('guidance-cards-grid');
        container.innerHTML = Object.entries(data.categories).map(([catKey, cat]) => `
            <div class="bg-slate-800 border border-slate-700/80 rounded-2xl p-6 shadow-lg space-y-4 flex flex-col justify-between">
                <div class="space-y-3">
                    <div class="flex items-center justify-between border-b border-slate-700/70 pb-3">
                        <div class="flex items-center space-x-3">
                            <span class="w-3.5 h-3.5 rounded-full inline-block" style="background-color: ${cat.color}"></span>
                            <h3 class="text-lg font-bold text-white">${cat.display_name}</h3>
                        </div>
                        <span class="px-2.5 py-0.5 rounded-md text-xs font-bold bg-slate-900 border border-slate-700 text-slate-300">${cat.bin_color} Bin</span>
                    </div>
                    <p class="text-xs text-slate-300 leading-relaxed">${cat.guidance}</p>
                </div>
                <div class="pt-2">
                    <span class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">Common Examples:</span>
                    <div class="flex flex-wrap gap-1.5">
                        ${cat.examples.map(ex => `<span class="px-2 py-0.5 bg-slate-900 text-slate-300 rounded text-[11px] border border-slate-700">${ex}</span>`).join('')}
                    </div>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error('Guidance load error:', e);
    }
}

// Load Admin Analytics (FR-8)
async function loadAdminAnalytics() {
    try {
        const res = await fetch('/api/analytics');
        const data = await res.json();
        if (!data.success) return;

        document.getElementById('admin-total-scans').textContent = data.total_scans;
        document.getElementById('admin-avg-confidence').textContent = `${data.avg_confidence}%`;
        document.getElementById('admin-low-conf-scans').textContent = data.low_confidence_scans;
        document.getElementById('admin-low-conf-rate').textContent = `${data.low_confidence_rate}%`;

        const rowsContainer = document.getElementById('admin-category-rows');
        const total = data.total_scans || 1;

        rowsContainer.innerHTML = Object.entries(data.category_distribution).map(([cat, count]) => {
            const pct = data.total_scans > 0 ? ((count / total) * 100).toFixed(1) : 0;
            return `
                <tr class="hover:bg-slate-900/40">
                    <td class="py-3 px-4 font-bold text-white flex items-center space-x-2">
                        <span>${cat}</span>
                    </td>
                    <td class="py-3 px-4 font-medium text-slate-200">${count}</td>
                    <td class="py-3 px-4 text-xs font-bold text-emerald-400">${pct}%</td>
                    <td class="py-3 px-4 w-48">
                        <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-700">
                            <div class="h-full bg-emerald-500 rounded-full" style="width: ${pct}%"></div>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (e) {
        console.error('Analytics load error:', e);
    }
}

function showAlert(message, type = 'error') {
    const alertBox = document.getElementById('alert-box');
    const alertIcon = document.getElementById('alert-icon');
    const alertMsg = document.getElementById('alert-message');

    alertBox.classList.remove('hidden', 'bg-red-500/10', 'border-red-500/30', 'text-red-300', 'bg-emerald-500/10', 'border-emerald-500/30', 'text-emerald-300');

    if (type === 'error') {
        alertBox.classList.add('bg-red-500/10', 'border-red-500/30', 'text-red-300');
        alertIcon.className = 'fa-solid fa-circle-exclamation text-red-400 text-xl';
    } else {
        alertBox.classList.add('bg-emerald-500/10', 'border-emerald-500/30', 'text-emerald-300');
        alertIcon.className = 'fa-solid fa-circle-check text-emerald-400 text-xl';
    }

    alertMsg.textContent = message;
}

function hideAlert() {
    document.getElementById('alert-box').classList.add('hidden');
}
