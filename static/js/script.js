document.addEventListener('DOMContentLoaded', function() {
    const fileUpload = document.getElementById('file-upload');
    const fileName = document.getElementById('file-name');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultLabel = document.getElementById('result-label');
    const probabilityFill = document.querySelector('.probability-fill');
    const probabilityValue = document.getElementById('probability-value');
    const inferenceTime = document.getElementById('inference-time'); // Add this line
    
    fileUpload.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                
                results.classList.add('hidden');
                
                uploadAndPredict();
            };
            reader.readAsDataURL(this.files[0]);
        }
    });
    
    function uploadAndPredict() {
        const file = fileUpload.files[0];
        if (!file) return;
        
        loading.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            loading.classList.add('hidden');
            
            resultLabel.textContent = data.prediction;
            resultLabel.style.color = data.prediction === 'Fake' ? '#e74c3c' : '#2ecc71';
            
            const displayedProb = data.prediction === 'Fake' ? 
                data.fake_probability : data.real_probability;
            const probPercent = Math.round(displayedProb * 100);
            
            probabilityFill.style.width = `${probPercent}%`;
            probabilityFill.style.backgroundColor = data.prediction === 'Fake' ? '#e74c3c' : '#2ecc71';
            probabilityValue.textContent = probPercent;
            
            if (data.inference_time_ms !== undefined) {
                inferenceTime.textContent = `Inference time: ${data.inference_time_ms.toFixed(2)} ms`;
                inferenceTime.classList.remove('hidden');
            } else {
                inferenceTime.classList.add('hidden');
            }
            
            results.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            loading.classList.add('hidden');
            alert('Error processing image. Please try again.');
        });
    }
});