//dashboard.js

fetch('/sales-data')
    .then(response => response.json())
    .then(data => {
        const ctx = document.getElementById('myChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Sales',
                    data: data.values,
                    borderColor: 'blue',
                    fill: false
                }]
            }
        });
    })
    .catch(error => console.error(error));
