//dashboard.js

document.addEventListener("DOMContentLoaded", function () {

    fetch('/sales-data')
        .then(res => res.json())
        .then(data => {

            const ctx = document.getElementById('salesChart').getContext('2d');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.dates,
                    datasets: [
                        {
                            label: 'Actual Sales',
                            data: data.sales,
                            borderColor: '#0d6efd',
                            fill: false,
                            tension: 0.3
                        },
                        {
                            label: 'Forecast',
                            data: data.forecast,
                            borderColor: '#dc3545',
                            borderDash: [5,5],
                            fill: false,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        });
});
