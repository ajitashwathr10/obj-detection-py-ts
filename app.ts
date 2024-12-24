document.getElementById('uploadForm')?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    const file = fileInput.files?.[0];
    if (!file) {
        alert('Please select a file!');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:5000/detect', {
            method: 'POST',
            body: formData,
        });
        const result = await response.json();
        document.getElementById('result')!.innerText = JSON.stringify(result.detections);
    } catch (error) {
        console.error('Error:', error);
    }
});
