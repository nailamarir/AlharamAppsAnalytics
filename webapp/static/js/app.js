/**
 * AlHaram Analytics - Preprocessing Pipeline Web App
 * Interactive data preprocessing with before/after visualization
 */

// State Management
const state = {
    sessionId: null,
    fileName: null,
    appliedSteps: [],
    originalColumns: [],
    currentColumns: []
};

// DOM Elements - initialized after DOM loads
let elements = {};

// Current preview step
let currentPreviewStep = null;

// =====================================================
// Initialization
// =====================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing...');

    // Initialize DOM elements
    elements = {
        uploadBox: document.getElementById('uploadBox'),
        fileInput: document.getElementById('fileInput'),
        fileInfo: document.getElementById('fileInfo'),
        fileName: document.getElementById('fileName'),
        fileStats: document.getElementById('fileStats'),
        changeFileBtn: document.getElementById('changeFileBtn'),
        pipelineSection: document.getElementById('pipelineSection'),
        dataSection: document.getElementById('dataSection'),
        applyAllBtn: document.getElementById('applyAllBtn'),
        resetBtn: document.getElementById('resetBtn'),
        downloadBtn: document.getElementById('downloadBtn'),
        previewModal: document.getElementById('previewModal'),
        modalClose: document.getElementById('modalClose'),
        modalCancelBtn: document.getElementById('modalCancelBtn'),
        modalApplyBtn: document.getElementById('modalApplyBtn'),
        modalTitle: document.getElementById('modalTitle'),
        beforeStats: document.getElementById('beforeStats'),
        afterStats: document.getElementById('afterStats'),
        beforeColHeader: document.getElementById('beforeColHeader'),
        afterColHeader: document.getElementById('afterColHeader'),
        comparisonBody: document.getElementById('comparisonBody'),
        dataTableHead: document.getElementById('dataTableHead'),
        dataTableBody: document.getElementById('dataTableBody'),
        totalRows: document.getElementById('totalRows'),
        totalCols: document.getElementById('totalCols'),
        newCols: document.getElementById('newCols'),
        appliedSteps: document.getElementById('appliedSteps'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        loadingText: document.getElementById('loadingText'),
        toastContainer: document.getElementById('toastContainer')
    };

    console.log('Elements initialized:', Object.keys(elements).length);

    initializeEventListeners();
});

function initializeEventListeners() {
    // File Upload
    elements.uploadBox.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.changeFileBtn.addEventListener('click', resetUpload);

    // Drag and Drop
    elements.uploadBox.addEventListener('dragover', handleDragOver);
    elements.uploadBox.addEventListener('dragleave', handleDragLeave);
    elements.uploadBox.addEventListener('drop', handleDrop);

    // Pipeline Actions
    elements.applyAllBtn.addEventListener('click', applyAllSteps);
    elements.resetBtn.addEventListener('click', resetData);
    elements.downloadBtn.addEventListener('click', downloadData);

    // Modal
    elements.modalClose.addEventListener('click', closeModal);
    elements.modalCancelBtn.addEventListener('click', closeModal);
    elements.modalApplyBtn.addEventListener('click', applyFromModal);
    elements.previewModal.addEventListener('click', (e) => {
        if (e.target === elements.previewModal) closeModal();
    });

    // Pipeline Cards
    document.querySelectorAll('.pipeline-card').forEach(card => {
        const stepId = card.dataset.step;

        card.querySelector('.preview-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            previewStep(stepId);
        });

        card.querySelector('.apply-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            applyStep(stepId);
        });
    });
}

// =====================================================
// File Upload Handlers
// =====================================================
function handleDragOver(e) {
    e.preventDefault();
    elements.uploadBox.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadBox.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadBox.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        elements.fileInput.files = files;
        handleFileSelect();
    }
}

async function handleFileSelect() {
    const file = elements.fileInput.files[0];
    if (!file) return;

    console.log('File selected:', file.name);

    // Validate file type
    const validTypes = ['.xlsx', '.xls', '.csv'];
    const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

    if (!validTypes.includes(fileExt)) {
        showToast('Please upload an Excel (.xlsx, .xls) or CSV file', 'error');
        return;
    }

    // Show loading
    showLoading('Uploading and processing file...');

    try {
        const formData = new FormData();
        formData.append('file', file);

        console.log('Uploading file...');

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        console.log('Response received:', response.status);

        const data = await response.json();
        console.log('Data received:', data);

        if (data.error) {
            throw new Error(data.error);
        }

        // Update state
        state.sessionId = data.session_id;
        state.fileName = data.filename;
        state.originalColumns = data.columns;
        state.currentColumns = data.columns;
        state.appliedSteps = [];

        // Update UI
        elements.fileName.textContent = data.filename;
        elements.fileStats.textContent = `${data.rows.toLocaleString()} rows • ${data.columns.length} columns`;

        elements.uploadBox.classList.add('hidden');
        elements.fileInfo.classList.remove('hidden');
        elements.pipelineSection.classList.remove('hidden');
        elements.dataSection.classList.remove('hidden');

        // Update data preview
        updateDataPreview(data.sample, data.columns);
        updateStats();

        showToast('File uploaded successfully!', 'success');

    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

function resetUpload() {
    elements.uploadBox.classList.remove('hidden');
    elements.fileInfo.classList.add('hidden');
    elements.pipelineSection.classList.add('hidden');
    elements.dataSection.classList.add('hidden');
    elements.fileInput.value = '';

    state.sessionId = null;
    state.appliedSteps = [];

    // Reset all cards
    document.querySelectorAll('.pipeline-card').forEach(card => {
        card.classList.remove('applied');
        card.querySelector('.status-badge').textContent = 'Pending';
        card.querySelector('.status-badge').classList.remove('applied');
        card.querySelector('.status-badge').classList.add('pending');
    });
}

// =====================================================
// Pipeline Operations
// =====================================================
async function previewStep(stepId) {
    if (!state.sessionId) {
        showToast('Please upload a file first', 'error');
        return;
    }

    showLoading('Generating preview...');

    try {
        const response = await fetch(`/preview/${state.sessionId}/${stepId}`);
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        currentPreviewStep = stepId;

        // Get step info
        const card = document.querySelector(`[data-step="${stepId}"]`);
        const stepName = card.querySelector('h3').textContent;
        const stepColor = getComputedStyle(card).getPropertyValue('--step-color');

        // Update modal title
        elements.modalTitle.innerHTML = `<i class="fas fa-eye"></i> Preview: ${stepName}`;

        // Update column headers
        elements.beforeColHeader.textContent = data.input_col;
        elements.afterColHeader.textContent = data.output_col;

        // Update stats
        updateModalStats(data.before_stats, data.after_stats);

        // Update comparison table
        updateComparisonTable(data.comparison);

        // Show modal
        elements.previewModal.classList.remove('hidden');

    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

function updateModalStats(before, after) {
    elements.beforeStats.innerHTML = `
        <div class="stat-item-modal">
            <span class="value">${before.total?.toLocaleString() || 'N/A'}</span>
            <span class="label">Total</span>
        </div>
        <div class="stat-item-modal">
            <span class="value">${before.unique?.toLocaleString() || 'N/A'}</span>
            <span class="label">Unique</span>
        </div>
        <div class="stat-item-modal">
            <span class="value">${before.missing?.toLocaleString() || 0}</span>
            <span class="label">Missing</span>
        </div>
        <div class="stat-item-modal">
            <span class="value">${before.missing_pct || 0}%</span>
            <span class="label">Missing %</span>
        </div>
    `;

    elements.afterStats.innerHTML = `
        <div class="stat-item-modal">
            <span class="value">${after.total?.toLocaleString() || 'N/A'}</span>
            <span class="label">Total</span>
        </div>
        <div class="stat-item-modal">
            <span class="value">${after.unique?.toLocaleString() || 'N/A'}</span>
            <span class="label">Unique</span>
        </div>
        <div class="stat-item-modal">
            <span class="value">${after.missing?.toLocaleString() || 0}</span>
            <span class="label">Missing</span>
        </div>
        <div class="stat-item-modal">
            <span class="value">${after.missing_pct || 0}%</span>
            <span class="label">Missing %</span>
        </div>
    `;
}

function updateComparisonTable(comparison) {
    elements.comparisonBody.innerHTML = comparison.map((row, i) => {
        const changed = row.before !== row.after;
        return `
            <tr>
                <td>${i + 1}</td>
                <td class="before-col">${escapeHtml(truncate(row.before, 50))}</td>
                <td class="after-col">${escapeHtml(truncate(row.after, 50))}</td>
                <td class="changed-col">
                    <span class="change-badge ${changed ? 'yes' : 'no'}">
                        <i class="fas ${changed ? 'fa-check' : 'fa-minus'}"></i>
                    </span>
                </td>
            </tr>
        `;
    }).join('');
}

function closeModal() {
    elements.previewModal.classList.add('hidden');
    currentPreviewStep = null;
}

async function applyFromModal() {
    if (currentPreviewStep) {
        closeModal();
        await applyStep(currentPreviewStep);
    }
}

async function applyStep(stepId) {
    if (!state.sessionId) {
        showToast('Please upload a file first', 'error');
        return;
    }

    if (state.appliedSteps.includes(stepId)) {
        showToast('This step has already been applied', 'info');
        return;
    }

    showLoading('Applying transformation...');

    try {
        const response = await fetch(`/apply/${state.sessionId}/${stepId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Update state
        state.appliedSteps = data.applied_steps;
        state.currentColumns = data.columns;

        // Update card UI
        updateCardStatus(stepId, true);

        // Update data preview
        updateDataPreview(data.sample, data.columns);
        updateStats();

        showToast('Step applied successfully!', 'success');

    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function applyAllSteps() {
    if (!state.sessionId) {
        showToast('Please upload a file first', 'error');
        return;
    }

    showLoading('Applying all transformations...');

    try {
        const response = await fetch(`/apply-all/${state.sessionId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Update state
        state.appliedSteps = data.applied_steps;
        state.currentColumns = data.columns;

        // Update all cards
        data.applied_steps.forEach(stepId => {
            updateCardStatus(stepId, true);
        });

        // Update data preview
        updateDataPreview(data.sample, data.columns);
        updateStats();

        showToast('All steps applied successfully!', 'success');

    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function resetData() {
    if (!state.sessionId) return;

    showLoading('Resetting data...');

    try {
        const response = await fetch(`/reset/${state.sessionId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Reset state
        state.appliedSteps = [];
        state.currentColumns = data.columns;

        // Reset all cards
        document.querySelectorAll('.pipeline-card').forEach(card => {
            updateCardStatus(card.dataset.step, false);
        });

        // Update data preview
        updateDataPreview(data.sample, data.columns);
        updateStats();

        showToast('Data reset to original state', 'success');

    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

function downloadData() {
    if (!state.sessionId) {
        showToast('No data to download', 'error');
        return;
    }

    window.location.href = `/download/${state.sessionId}`;
    showToast('Download started!', 'success');
}

// =====================================================
// UI Updates
// =====================================================
function updateCardStatus(stepId, applied) {
    const card = document.querySelector(`[data-step="${stepId}"]`);
    if (!card) return;

    const badge = card.querySelector('.status-badge');

    if (applied) {
        card.classList.add('applied');
        badge.textContent = 'Applied';
        badge.classList.remove('pending');
        badge.classList.add('applied');
    } else {
        card.classList.remove('applied');
        badge.textContent = 'Pending';
        badge.classList.remove('applied');
        badge.classList.add('pending');
    }
}

function updateDataPreview(sample, columns) {
    // Update table headers
    elements.dataTableHead.innerHTML = `
        <tr>
            ${columns.map(col => {
                const isNew = !state.originalColumns.includes(col);
                return `<th class="${isNew ? 'new-column' : ''}">${escapeHtml(col)}${isNew ? ' ✨' : ''}</th>`;
            }).join('')}
        </tr>
    `;

    // Update table body
    elements.dataTableBody.innerHTML = sample.map(row => `
        <tr>
            ${columns.map(col => {
                const isNew = !state.originalColumns.includes(col);
                const value = row[col] !== null && row[col] !== undefined ? row[col] : '';
                return `<td class="${isNew ? 'new-column' : ''}">${escapeHtml(truncate(String(value), 40))}</td>`;
            }).join('')}
        </tr>
    `).join('');
}

async function updateStats() {
    if (!state.sessionId) return;

    try {
        const response = await fetch(`/stats/${state.sessionId}`);
        const data = await response.json();

        elements.totalRows.textContent = data.total_rows.toLocaleString();
        elements.totalCols.textContent = data.total_columns;
        elements.newCols.textContent = data.new_columns;
        elements.appliedSteps.textContent = `${data.applied_steps}/6`;

    } catch (error) {
        console.error('Failed to fetch stats:', error);
    }
}

// =====================================================
// Utility Functions
// =====================================================
function showLoading(text = 'Processing...') {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    elements.loadingOverlay.classList.add('hidden');
}

function showToast(message, type = 'info') {
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas ${icons[type]}"></i>
        <span>${escapeHtml(message)}</span>
    `;

    elements.toastContainer.appendChild(toast);

    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function truncate(text, maxLength) {
    if (!text) return '';
    text = String(text);
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}
