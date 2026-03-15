window.ChartManager = {
    mainChart: null,
    indicatorChart: null,
    modalChart: null,

    // OKLCH-balanced functional colors mapped to standard Hexes
    // L ~ 70%, C ~ 0.15 for consistent perceived brightness across chart lines
    COLORS: {
        sky: '#38bdf8',    // Primary Data
        amber: '#fbbf24',  // MA20
        rose: '#fb7185',   // MA50
        purple: '#c084fc', // Forecast
        slateGrid: '#1e293b', 
        slateTooltip: '#0f172a'
    },

    _formatDateTime(val) {
        if (!val) return '';
        let dateObj = new Date(val);
        
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

    _decimateData(data, maxPoints = 150) {
        if (!data || data.length <= maxPoints) return data;
        const step = Math.ceil(data.length / maxPoints);
        return data.filter((_, index) => index % step === 0);
    },

    renderMainChart(canvasId, priceDataRaw, predictionRaw = null) {
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

        let predPrices = new Array(prices.length).fill(null);

        if (predictionRaw && predictionRaw.predictions) {
            predPrices[prices.length - 1] = prices[prices.length - 1]; 
            
            predictionRaw.predictions.forEach(p => {
                labels.push(this._formatDateTime(p.date));
                prices.push(null);
                ma20.push(null);
                ma50.push(null);
                predPrices.push(p.close);
            });
        }

        const validPrices = prices.filter(p => p !== null && p > 0).concat(predPrices.filter(p => p !== null && p > 0));
        const minPrice = validPrices.length ? Math.min(...validPrices) : 0;
        const maxPrice = validPrices.length ? Math.max(...validPrices) : 100;
        
        const priceRange = maxPrice - minPrice;
        const bottomPadding = priceRange * 0.05; 
        const topPadding = priceRange * 0.15;    

        const datasets = [
            { 
                label: 'Close Price', 
                data: prices, 
                borderColor: this.COLORS.sky, 
                borderWidth: 2, 
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointBackgroundColor: this.COLORS.slateTooltip,
                pointBorderColor: this.COLORS.sky,
                pointBorderWidth: 2
            },
            { 
                label: 'MA20', 
                data: ma20, 
                borderColor: this.COLORS.amber, 
                borderWidth: 1.5, 
                borderDash: [5, 5], 
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4
            },
            { 
                label: 'MA50', 
                data: ma50, 
                borderColor: this.COLORS.rose, 
                borderWidth: 1.5, 
                borderDash: [2, 2], 
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4
            }
        ];

        if (predictionRaw && predictionRaw.predictions) {
            const predLength = predictionRaw.predictions.length;
            const predLabel = predictionRaw.model_used === 'RandomForestRegressor' 
                ? `Hypo ${predLength}D Forecast (RF)` 
                : `Hypo ${predLength}D Forecast (PyTorch)`;

            datasets.push({
                label: predLabel,
                data: predPrices,
                borderColor: this.COLORS.purple, 
                borderWidth: 2,
                borderDash: [5, 5], 
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointBackgroundColor: this.COLORS.slateTooltip,
                pointBorderColor: this.COLORS.purple,
                pointBorderWidth: 2
            });
        }

        this.mainChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                normalized: true, 
                
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
                    legend: { 
                        display: true, 
                        labels: { 
                            color: '#94a3b8', 
                            usePointStyle: true, 
                            boxWidth: 8, 
                            font: { weight: 300 },
                            padding: 25 // Adds gap between legends per UI request
                        } 
                    },
                    tooltip: {
                        backgroundColor: this.COLORS.slateTooltip,
                        titleColor: '#f8fafc', bodyColor: '#cbd5e1', borderColor: '#1e293b',
                        borderWidth: 1, padding: 12, displayColors: true,
                        callbacks: {
                            title: (context) => context[0].label,
                            label: (context) => {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) {
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
                        grid: { color: this.COLORS.slateGrid } 
                    },
                    y: { 
                        ticks: { color: '#64748b', maxTicksLimit: 4, font: { weight: 300 } }, 
                        grid: { color: this.COLORS.slateGrid },
                        min: Math.max(0, minPrice - bottomPadding), 
                        max: maxPrice + topPadding
                    }
                }
            }
        });
    },
    
    // Dedicated isolated chart variant for the prediction modal (Only Historic & Predict lines)
    renderPredictModalChart(canvasId, priceDataRaw, predictionRaw = null) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        if (this.modalChart) {
            this.modalChart.destroy();
            this.modalChart = null;
        }

        if (!priceDataRaw || priceDataRaw.length === 0) {
            this._showError(canvasId, 'No price data available');
            return;
        }

        // Trim to ~30 calendar days from the final data point so predictions bind in proportionally
        const lastEntry = priceDataRaw[priceDataRaw.length - 1];
        const lastDate = new Date(lastEntry.date || lastEntry.timestamp || lastEntry.time);
        const cutoff = new Date(lastDate);
        cutoff.setDate(cutoff.getDate() - 30);
        const trimmed = priceDataRaw.filter(d => {
            const dt = new Date(d.date || d.timestamp || d.time);
            return dt >= cutoff;
        });
        const priceData = trimmed.length > 0 ? trimmed : priceDataRaw.slice(-30);

        const ctx = canvas.getContext('2d');
        
        const labels = priceData.map(d => this._formatDateTime(d.date || d.timestamp || d.time));
        const prices = priceData.map(d => d.close || 0);
        let predPrices = new Array(prices.length).fill(null);

        if (predictionRaw && predictionRaw.predictions) {
            predPrices[prices.length - 1] = prices[prices.length - 1]; 
            predictionRaw.predictions.forEach(p => {
                labels.push(this._formatDateTime(p.date));
                prices.push(null);
                predPrices.push(p.close);
            });
        }

        const validPrices = prices.filter(p => p !== null && p > 0).concat(predPrices.filter(p => p !== null && p > 0));
        const minPrice = validPrices.length ? Math.min(...validPrices) : 0;
        const maxPrice = validPrices.length ? Math.max(...validPrices) : 100;
        const padding = (maxPrice - minPrice) * 0.15;

        const datasets = [
            { 
                label: 'Historical Close', 
                data: prices, 
                borderColor: this.COLORS.sky, 
                borderWidth: 2, 
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointBackgroundColor: this.COLORS.slateTooltip,
                pointBorderColor: this.COLORS.sky,
                pointBorderWidth: 2
            }
        ];

        if (predictionRaw && predictionRaw.predictions) {
            datasets.push({
                label: `Forecast (${predictionRaw.predictions.length}D)`,
                data: predPrices,
                borderColor: this.COLORS.purple, 
                borderWidth: 2,
                borderDash: [5, 5], 
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointBackgroundColor: this.COLORS.slateTooltip,
                pointBorderColor: this.COLORS.purple,
                pointBorderWidth: 2
            });
        }

        this.modalChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { 
                    legend: { 
                        display: true, 
                        labels: { color: '#94a3b8', usePointStyle: true, boxWidth: 8, padding: 25 } 
                    },
                    tooltip: {
                        backgroundColor: this.COLORS.slateTooltip,
                        titleColor: '#f8fafc', bodyColor: '#cbd5e1', borderColor: '#1e293b',
                        borderWidth: 1, padding: 12
                    }
                },
                scales: {
                    x: { ticks: { color: '#64748b', maxTicksLimit: 6 }, grid: { color: this.COLORS.slateGrid } },
                    y: { 
                        ticks: { color: '#64748b' }, 
                        grid: { color: this.COLORS.slateGrid },
                        min: Math.max(0, minPrice - padding), 
                        max: maxPrice + padding
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
        // Ensure neutral/functional color continuity
        let bgColor = isVolume ? `${this.COLORS.sky}33` : 'transparent'; 
        let borderColor = this.COLORS.sky;

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
                
                animation: {
                    duration: 1200,
                    easing: 'easeOutQuart'
                },
                interaction: { mode: 'index', intersect: false },
                plugins: { 
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: this.COLORS.slateTooltip, titleColor: '#f8fafc', bodyColor: '#cbd5e1',
                        borderColor: '#1e293b', borderWidth: 1, padding: 10, displayColors: false,
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
                    x: { ticks: { display: false, maxTicksLimit: 4 }, grid: { color: this.COLORS.slateGrid } },
                    y: { 
                        type: isVolume ? 'logarithmic' : 'linear', 
                        ticks: { color: '#64748b', maxTicksLimit: 4, font: { weight: 300 } }, 
                        grid: { color: this.COLORS.slateGrid },
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
        ctx.fillStyle = '#64748b'; // slate-500
        ctx.font = '300 14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(message, canvas.width / 2, canvas.height / 2);
    }
};