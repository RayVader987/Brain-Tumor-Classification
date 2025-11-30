let selectedFile = null;
let currentChart = null;

const colorMap = {
    'glioma': '#ef4444',
    'meningioma': '#f59e0b',
    'pituitary': '#8b5cf6',
    'notumor': '#10b981'
};

document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ App loaded');
    
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const removeBtn = document.getElementById('removeBtn');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('spinner');
    const resultsSection = document.getElementById('resultsSection');
    
    uploadArea.addEventListener('click', () => {
        if (imagePreview.style.display === 'none') {
            fileInput.click();
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#e2e8f0';
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });
    
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        reset();
    });
    
    predictBtn.addEventListener('click', predict);
    
    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please upload an image');
            return;
        }
        if (file.size > 5 * 1024 * 1024) {
            alert('File too large (max 5MB)');
            return;
        }
        
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadPlaceholder.style.display = 'none';
            imagePreview.style.display = 'block';
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
    
    function reset() {
        selectedFile = null;
        fileInput.value = '';
        uploadPlaceholder.style.display = 'block';
        imagePreview.style.display = 'none';
        predictBtn.disabled = true;
        resultsSection.style.display = 'none';
    }
    
    async function predict() {
        if (!selectedFile) return;
        
        btnText.textContent = 'Analyzing...';
        spinner.style.display = 'block';
        predictBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model', document.getElementById('modelSelect').value);
        
        try {
            const res = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Prediction failed');
            }
            
            const result = await res.json();
            console.log('Result:', result);
            showResults(result);
            
        } catch (error) {
            console.error(error);
            alert('Error: ' + error.message);
        } finally {
            btnText.textContent = 'Analyze Scan';
            spinner.style.display = 'none';
            predictBtn.disabled = false;
        }
    }
    
    function showResults(result) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        document.getElementById('modelBadge').textContent = result.model_used;
        
        const badge = document.getElementById('predictionBadge');
        badge.className = 'prediction-badge ' + result.prediction;
        document.getElementById('predictionLabel').textContent = formatName(result.prediction);
        document.getElementById('confidenceText').textContent = result.confidence.toFixed(1) + '% Confidence';
        
        const fill = document.getElementById('confidenceFill');
        fill.style.width = '0%';
        setTimeout(() => fill.style.width = result.confidence + '%', 100);
        
        const names = Object.keys(result.probabilities);
        const probs = Object.values(result.probabilities);
        updateChart(names, probs);
        
        document.getElementById('originalImg').src = result.image_path;
        
        document.getElementById('resultDisplay').innerHTML = `
            <svg viewBox="0 0 100 100" fill="none" stroke="${colorMap[result.prediction]}" stroke-width="3">
                <circle cx="50" cy="50" r="40"/>
                <path d="M30 50 L45 65 L70 35" stroke-width="5"/>
            </svg>
            <p style="color:${colorMap[result.prediction]}">${formatName(result.prediction)}</p>
            <span style="color:#64748b">${result.confidence.toFixed(1)}%</span>
        `;
    }
    
    function updateChart(names, probs) {
        const ctx = document.getElementById('probabilityChart').getContext('2d');
        if (currentChart) currentChart.destroy();
        
        const colors = names.map(n => colorMap[n] || '#667eea');
        
        currentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: names.map(formatName),
                datasets: [{
                    data: probs,
                    backgroundColor: colors.map(c => c + '40'),
                    borderColor: colors,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: v => v + '%' }
                    }
                }
            }
        });
    }
    
    function formatName(name) {
        return name === 'notumor' ? 'No Tumor' : 
               name.charAt(0).toUpperCase() + name.slice(1);
    }
});