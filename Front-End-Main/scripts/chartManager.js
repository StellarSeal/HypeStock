window.ChartManager = {
    mainChart: null,
    indicatorChart: null,

    _formatDateTime(val) {
        if (!val) return '';
        let dateObj = new Date(val);
        
        // Handle timestamps directly
        if (isNaN(dateObj.getTime())) {
            if (!isNaN(Number(val))) {
                const numVal = Number(val);
                dateObj = new Date(numVal < 10000000000 ? numVal * 1000 : numVal);
            } else {
                return String(val);
            }
        }
        
        const dd = String(dateObj.getDate()).padStart(2, '0');
        const MM = String(dateObj.getMonth() + 1).padStart(2, '0');
        const yyyy = dateObj.getFullYear();
        
        return `${dd}-${MM}-${yyyy}`;
    },

    // Decimates visual chart loading size to fix stutter/lag without compromising trend view
    _decimateData(data, maxPoints = 150) {
        if (!data || data.length <= maxPoints) return data;
        const step = Math.ceil(data.length / maxPoints);
        return data.filter((_, index) => index % step === 0);
    },

    renderMainChart(canvasId, priceDataRaw) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        if (this.mainChart) {
            this.mainChart.destroy();
            this.mainChart = null;
        }

        if (!priceDataRaw || priceDataRaw.length === 0) {
            this._showError(canvasId, 'No price data available');
            return;
        }

        const priceData = this._decimateData(priceDataRaw, 150);

        const ctx = canvas.getContext('2d');
        const labels = priceData.map(d => this._formatDateTime(d.date || d.timestamp || d.time));
        const prices = priceData.map(d => d.close || 0);
        const ma20 = priceData.map(d => d.ma20 || null);
        const ma50 = priceData.map(d => d.ma50 || null);

        const validPrices = prices.filter(p => p !== null && p > 0);
        const minPrice = validPrices.length ? Math.min(...validPrices) : 0;
        const maxPrice = validPrices.length ? Math.max(...validPrices) : 100;
        
        const priceRange = maxPrice - minPrice;
        
        // Tweak: 5% bottom padding ensures bezier tension doesn't dip below the axis line
        const bottomPadding = priceRange * 0.05; 
        const topPadding = priceRange * 0.15;    

        this.mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { 
                        label: 'Close Price', 
                        data: prices, 
                        borderColor: '#38bdf8', 
                        borderWidth: 2, 
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#0f172a',
                        pointBorderColor: '#38bdf8',
                        pointBorderWidth: 2
                    },
                    { 
                        label: 'MA20', 
                        data: ma20, 
                        borderColor: '#fbbf24', 
                        borderWidth: 1.5, 
                        borderDash: [5, 5], 
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    },
                    { 
                        label: 'MA50', 
                        data: ma50, 
                        borderColor: '#f87171', 
                        borderWidth: 1.5, 
                        borderDash: [2, 2], 
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                normalized: true, // Optimizes internal parsing
                
                // Tweak: "Pop up" straight line animation starting from the bottom of the graph
                animation: {
                    duration: 1200,
                    easing: 'easeOutQuart'
                },
                animations: {
                    y: {
                        from: (ctx) => {
                            if (ctx.type === 'data' && ctx.chart.scales.y) {
                                return ctx.chart.scales.y.getPixelForValue(ctx.chart.scales.y.min);
                            }
                        }
                    }
                },
                
                layout: {
                    padding: { bottom: 0 } 
                },
                interaction: { mode: 'index', intersect: false },
                plugins: { 
                    legend: { display: true, labels: { color: '#94a3b8', usePointStyle: true, boxWidth: 8, font: { weight: 300 } } },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleColor: '#e2e8f0', bodyColor: '#cbd5e1', borderColor: 'rgba(56, 189, 248, 0.3)',
                        borderWidth: 1, padding: 12, displayColors: true,
                        callbacks: {
                            title: (context) => context[0].label,
                            label: (context) => {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) {
                                    // Tweak: Formatted to 1000 VNĐ
                                    label += context.parsed.y.toLocaleString('vi-VN', {maximumFractionDigits: 2}) + 'k VNĐ';
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: '#64748b', maxTicksLimit: 4, font: { weight: 300 } }, 
                        grid: { color: '#1e293b' } 
                    },
                    y: { 
                        ticks: { color: '#64748b', maxTicksLimit: 4, font: { weight: 300 } }, 
                        grid: { color: '#1e293b' },
                        min: Math.max(0, minPrice - bottomPadding), 
                        max: maxPrice + topPadding
                    }
                }
            }
        });
    },

    renderIndicator(canvasId, type, dataRaw) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        if (this.indicatorChart) {
            this.indicatorChart.destroy();
            this.indicatorChart = null;
        }

        if (!dataRaw || dataRaw.length === 0) {
            this._showError(canvasId, 'Indicator data unavailable');
            return;
        }

        const data = this._decimateData(dataRaw, 150);

        const ctx = canvas.getContext('2d');
        const labels = data.map(d => this._formatDateTime(d.date || d.timestamp || d.time));
        const isVolume = type === 'Volume';
        
        const values = data.map(d => {
            let val = d.value || 0;
            if (isVolume && val <= 0) val = 1; 
            return val;
        });
        
        let chartType = 'line';
        let bgColor = 'rgba(56, 189, 248, 0.2)';
        let borderColor = '#38bdf8';

        if (isVolume) {
            chartType = 'bar';
            bgColor = '#38bdf8';
        }

        this.indicatorChart = new Chart(ctx, {
            type: chartType,
            data: {
                labels: labels,
                datasets: [{
                    label: type, data: values, backgroundColor: bgColor, 
                    borderColor: borderColor, borderWidth: chartType === 'line' ? 2 : 1, 
                    tension: 0.4, pointRadius: 0, pointHoverRadius: 5, fill: chartType === 'line'
                }]
            },
            options: {
                responsive: true, 
                maintainAspectRatio: false, 
                normalized: true,
                
                // Tweak: Line pop-up from bottom animation
                animation: {
                    duration: 1200,
                    easing: 'easeOutQuart'
                },
                animations: {
                    y: {
                        from: (ctx) => {
                            if (ctx.type === 'data' && ctx.chart.scales.y) {
                                return ctx.chart.scales.y.getPixelForValue(ctx.chart.scales.y.min);
                            }
                        }
                    }
                },
                
                interaction: { mode: 'index', intersect: false },
                plugins: { 
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)', titleColor: '#e2e8f0', bodyColor: '#cbd5e1',
                        borderColor: 'rgba(56, 189, 248, 0.3)', borderWidth: 1, padding: 10, displayColors: false,
                        callbacks: {
                            title: (context) => context[0].label,
                            label: (context) => {
                                let val = context.parsed.y;
                                if(isVolume && val === 1) val = 0; 
                                return `${context.dataset.label}: ${val.toLocaleString('vi-VN')}`;
                            }
                        }
                    }
                },
                scales: {
                    x: { ticks: { display: false, maxTicksLimit: 4 }, grid: { color: '#1e293b' } },
                    y: { 
                        type: isVolume ? 'logarithmic' : 'linear', 
                        ticks: { color: '#64748b', maxTicksLimit: 4, font: { weight: 300 } }, 
                        grid: { color: '#1e293b' },
                        grace: '10%'
                    }
                }
            }
        });
    },

    _showError(canvasId, message) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#64748b';
        ctx.font = '300 14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(message, canvas.width / 2, canvas.height / 2);
    }
};