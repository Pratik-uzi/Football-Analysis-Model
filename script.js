document.getElementById('data-form').addEventListener('submit', function (e) {
    e.preventDefault();
    let fileInput = document.getElementById('data-input');
    let formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = 'Analysis Result: ' + data.result;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
