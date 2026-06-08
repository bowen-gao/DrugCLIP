// ============================================
// DrugCLIP Web Application - Main JavaScript
// ============================================

// Global state
let appState = {
    currentTab: 'single',
    singleResult: null,
    batchResults: null,
    molecules: [],
    isLoading: false
};

// API endpoints
const API_BASE = '/api';
const endpoints = {
    health: '/api/health',
    predict: '/api/predict',
    batchPredict: '/api/batch_predict',
    uploadMolecules: '/api/upload_molecules',
    examples: '/api/examples'
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkServerHealth();
    loadExamples();
});

function initializeEventListeners() {
    // Tab navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Single prediction
    document.getElementById('predictBtn').addEventListener('click', singlePredict);
    document.getElementById('loadExampleBtn').addEventListener('click', loadSingleExample);

    // Batch prediction
    document.getElementById('molFile').addEventListener('change', handleFileUpload);
    document.getElementById('batchPredictBtn').addEventListener('click', batchPredict);
    document.getElementById('loadBatchExampleBtn').addEventListener('click', loadBatchExample);
    document.getElementById('downloadCsvBtn').addEventListener('click', downloadResults.bind(null, 'csv'));
    document.getElementById('downloadJsonBtn').addEventListener('click', downloadResults.bind(null, 'json'));

    // Enter key for predict
    document.getElementById('molSmiles').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            singlePredict();
        }
    });
}

// ============================================
// Server Health Check
// ============================================

async function checkServerHealth() {
    try {
        const response = await axios.get(endpoints.health);
        updateStatusIndicator(response.data.model_loaded);
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatusIndicator(false);
    }
}

function updateStatusIndicator(isHealthy) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    if (isHealthy) {
        statusDot.classList.remove('offline');
        statusText.textContent = '服务正常';
    } else {
        statusDot.classList.add('offline');
        statusText.textContent = '服务离线';
    }
}

// ============================================
// Tab Navigation
// ============================================

function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    const tabElement = document.getElementById(tabName);
    tabElement.classList.add('active');
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    appState.currentTab = tabName;

    // Scroll to top of the tab content
    tabElement.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Update charts if switching to results tab
    if (tabName === 'results' && appState.batchResults) {
        updateResultsCharts();
    }
}

// ============================================
// Single Prediction
// ============================================

async function singlePredict() {
    const smiles = document.getElementById('molSmiles').value.trim();
    const pocketId = document.getElementById('pocketId').value.trim();

    // Validate input
    if (!smiles) {
        showError('请输入 SMILES 字符串');
        return;
    }

    if (!pocketId) {
        showError('请输入蛋白质口袋 ID');
        return;
    }

    try {
        showLoading('正在预测...');

        const response = await axios.post(endpoints.predict, {
            mol_smi: smiles,
            pocket_id: pocketId
        });

        if (response.data.success) {
            appState.singleResult = response.data.data;
            displaySingleResult(response.data.data);
        } else {
            showError(response.data.message || '预测失败');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.response?.data?.message || '预测失败，请重试');
    } finally {
        hideLoading();
    }
}

function displaySingleResult(result) {
    const container = document.getElementById('singleResultContainer');
    document.getElementById('resultSmiles').textContent = result.mol_smi;
    document.getElementById('resultPocket').textContent = result.pocket_id;
    document.getElementById('resultScore').textContent = result.score.toFixed(2);
    document.getElementById('resultProb').textContent = (result.binding_probability * 100).toFixed(1);
    document.getElementById('resultConfidence').textContent = (result.confidence * 100).toFixed(1);

    container.classList.remove('hidden');

    // Draw chart
    drawSingleChart(result);
}

function drawSingleChart(result) {
    const chart = echarts.init(document.getElementById('singleChart'));

    const option = {
        title: {
            text: '结合评分分布',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis'
        },
        xAxis: {
            type: 'category',
            data: ['结合分数']
        },
        yAxis: {
            type: 'value',
            name: '分数'
        },
        series: [{
            type: 'bar',
            data: [result.score],
            itemStyle: {
                color: result.score < 0 ? '#10b981' : '#f59e0b'
            }
        }],
        grid: {
            left: '10%',
            right: '10%',
            bottom: '20%',
            containLabel: true
        }
    };

    chart.setOption(option);
}

function loadSingleExample() {
    document.getElementById('molSmiles').value = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O';
    document.getElementById('pocketId').value = '1a30';
}

// ============================================
// Batch Prediction
// ============================================

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        showLoading('正在解析文件...');

        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post(endpoints.uploadMolecules, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });

        if (response.data.success) {
            appState.molecules = response.data.molecules;
            displayMoleculeList(response.data.molecules);
        } else {
            showError(response.data.error || '文件解析失败');
        }
    } catch (error) {
        console.error('File upload error:', error);
        showError(error.response?.data?.error || '文件上传失败');
    } finally {
        hideLoading();
    }
}

function displayMoleculeList(molecules) {
    const container = document.getElementById('molListContainer');
    const preview = document.getElementById('molListPreview');

    let html = '';
    molecules.slice(0, 10).forEach((mol, idx) => {
        html += `
            <div class="item">
                <strong>#${idx + 1}:</strong> ${mol.name} 
                <code style="font-size: 11px; color: #6b7280;">SMILES: ${mol.smi.substring(0, 40)}...</code>
            </div>
        `;
    });

    if (molecules.length > 10) {
        html += `<div class="item"><em>... 还有 ${molecules.length - 10} 个分子</em></div>`;
    }

    preview.innerHTML = html;
    container.classList.remove('hidden');
}

async function batchPredict() {
    if (!appState.molecules || appState.molecules.length === 0) {
        showError('请先上传分子文件');
        return;
    }

    const pocketId = document.getElementById('batchPocketId').value.trim();
    if (!pocketId) {
        showError('请输入蛋白质口袋 ID');
        return;
    }

    try {
        const progressContainer = document.getElementById('progressContainer');
        progressContainer.classList.remove('hidden');

        const response = await axios.post(endpoints.batchPredict, {
            molecules: appState.molecules,
            pocket_id: pocketId
        }, {
            onUploadProgress: (progressEvent) => {
                if (progressEvent.total) {
                    const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
                    updateProgress(percent);
                }
            }
        });

        if (response.data.success) {
            appState.batchResults = response.data.results;
            displayBatchResults(response.data.results);

            // Auto-switch to results tab
            setTimeout(() => switchTab('results'), 500);
        } else {
            showError(response.data.error || '批量预测失败');
        }
    } catch (error) {
        console.error('Batch prediction error:', error);
        showError(error.response?.data?.error || '批量预测失败');
    }
}

function displayBatchResults(results) {
    const container = document.getElementById('batchResultsContainer');
    const tbody = document.getElementById('resultsTableBody');

    tbody.innerHTML = '';
    results.forEach((result, idx) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${idx + 1}</td>
            <td>${result.name}</td>
            <td style="font-family: monospace; font-size: 12px;">${result.smi.substring(0, 40)}</td>
            <td><strong style="color: #3b82f6;">${result.score ? result.score.toFixed(2) : 'N/A'}</strong></td>
            <td>${result.binding_probability ? (result.binding_probability * 100).toFixed(1) : 'N/A'}</td>
            <td>${result.confidence ? (result.confidence * 100).toFixed(1) : 'N/A'}</td>
        `;
        tbody.appendChild(row);
    });

    container.classList.remove('hidden');
}

function loadBatchExample() {
    appState.molecules = [
        { smi: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', name: 'Ibuprofen' },
        { smi: 'CC(=O)Oc1ccccc1C(=O)O', name: 'Aspirin' },
        { smi: 'c1ccccc1', name: 'Benzene' },
        { smi: 'CC(C)C', name: 'Isobutane' }
    ];
    displayMoleculeList(appState.molecules);
}

function updateProgress(percent) {
    document.getElementById('progressFill').style.width = percent + '%';
    document.getElementById('progressPercent').textContent = percent + '%';
}

// ============================================
// Results Analysis
// ============================================

function updateResultsCharts() {
    if (!appState.batchResults || appState.batchResults.length === 0) return;

    updateStatistics();
    drawScoresChart();
    drawProbabilityChart();
    drawDistributionChart();
}

function updateStatistics() {
    const results = appState.batchResults.filter(r => r.score !== null);
    const scores = results.map(r => r.score);

    if (scores.length === 0) return;

    const max = Math.max(...scores);
    const min = Math.min(...scores);
    const avg = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2);

    document.getElementById('statTotal').textContent = results.length;
    document.getElementById('statMax').textContent = max.toFixed(2);
    document.getElementById('statMin').textContent = min.toFixed(2);
    document.getElementById('statAvg').textContent = avg;
}

function drawScoresChart() {
    const chart = echarts.init(document.getElementById('scoresChart'));
    const results = appState.batchResults.filter(r => r.score !== null);
    const names = results.map(r => r.name);
    const scores = results.map(r => r.score);

    const option = {
        title: { text: '分子结合评分排行', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: {
            type: 'category',
            data: names.slice(0, 20),
            axisLabel: { rotate: 45 }
        },
        yAxis: { type: 'value', name: '评分' },
        series: [{
            type: 'bar',
            data: scores.slice(0, 20),
            itemStyle: {
                color: (params) => scores[params.dataIndex] < 0 ? '#10b981' : '#f59e0b'
            }
        }],
        grid: { left: '10%', right: '10%', bottom: '25%', containLabel: true }
    };

    chart.setOption(option);
}

function drawProbabilityChart() {
    const chart = echarts.init(document.getElementById('probabilityChart'));
    const results = appState.batchResults.filter(r => r.binding_probability !== null);
    const names = results.map(r => r.name);
    const probs = results.map(r => (r.binding_probability * 100).toFixed(1));

    const option = {
        title: { text: '结合概率分布', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: {
            type: 'category',
            data: names.slice(0, 20),
            axisLabel: { rotate: 45 }
        },
        yAxis: { type: 'value', name: '概率 (%)', max: 100 },
        series: [{
            type: 'line',
            data: probs.slice(0, 20),
            smooth: true,
            itemStyle: { color: '#3b82f6' },
            areaStyle: { color: 'rgba(59, 130, 246, 0.2)' }
        }],
        grid: { left: '10%', right: '10%', bottom: '25%', containLabel: true }
    };

    chart.setOption(option);
}

function drawDistributionChart() {
    const chart = echarts.init(document.getElementById('distributionChart'));
    const results = appState.batchResults.filter(r => r.score !== null);
    const scores = results.map(r => r.score);

    // Histogram
    const bins = 20;
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const binWidth = (max - min) / bins;
    const histogram = new Array(bins).fill(0);

    scores.forEach(score => {
        const binIndex = Math.floor((score - min) / binWidth);
        histogram[Math.min(binIndex, bins - 1)]++;
    });

    const binLabels = [];
    for (let i = 0; i < bins; i++) {
        binLabels.push((min + i * binWidth).toFixed(1));
    }

    const option = {
        title: { text: '评分分布直方图', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: binLabels, axisLabel: { rotate: 45 } },
        yAxis: { type: 'value', name: '频率' },
        series: [{
            type: 'bar',
            data: histogram,
            itemStyle: { color: '#8b5cf6' }
        }],
        grid: { left: '10%', right: '10%', bottom: '25%', containLabel: true }
    };

    chart.setOption(option);
}

// ============================================
// Data Export
// ============================================

function downloadResults(format) {
    if (!appState.batchResults) {
        showError('没有可导出的结果');
        return;
    }

    const data = appState.batchResults;
    let content = '';
    let filename = '';

    if (format === 'csv') {
        content = convertToCSV(data);
        filename = 'drugclip_results.csv';
    } else if (format === 'json') {
        content = JSON.stringify(data, null, 2);
        filename = 'drugclip_results.json';
    }

    downloadFile(content, filename, format === 'csv' ? 'text/csv' : 'application/json');
}

function convertToCSV(data) {
    const headers = ['排名', '分子名称', 'SMILES', '结合分数', '结合概率', '置信度'];
    const rows = data.map((result, idx) => [
        idx + 1,
        result.name,
        result.smi,
        result.score ? result.score.toFixed(3) : 'N/A',
        result.binding_probability ? (result.binding_probability * 100).toFixed(1) : 'N/A',
        result.confidence ? (result.confidence * 100).toFixed(1) : 'N/A'
    ]);

    let csv = headers.join(',') + '\n';
    rows.forEach(row => {
        csv += row.map(cell => {
            if (typeof cell === 'string' && cell.includes(',')) {
                return `"${cell}"`;
            }
            return cell;
        }).join(',') + '\n';
    });

    return csv;
}

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

// ============================================
// Utilities
// ============================================

async function loadExamples() {
    try {
        const response = await axios.get(endpoints.examples);
        console.log('Examples loaded:', response.data);
    } catch (error) {
        console.error('Failed to load examples:', error);
    }
}

function showLoading(message = '处理中...') {
    const modal = document.getElementById('loadingModal');
    document.getElementById('loadingText').textContent = message;
    modal.classList.remove('hidden');
    appState.isLoading = true;
}

function hideLoading() {
    const modal = document.getElementById('loadingModal');
    modal.classList.add('hidden');
    appState.isLoading = false;
}

function showError(message) {
    const modal = document.getElementById('errorModal');
    document.getElementById('errorText').textContent = message;
    modal.classList.remove('hidden');
}

// Close loading modal when escape pressed
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.getElementById('errorModal').classList.add('hidden');
    }
});
