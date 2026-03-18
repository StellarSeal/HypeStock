window.ChartManager = {
    mainChart: null,
    indicatorChart: null,
    modalChart: null,

    // OKLCH-balanced functional colors mapped to standard Hexes
    COLORS: {
        sky: '#38bdf8',    // Primary Data
        amber: '#fbbf24',  // MA20
        rose: '#fb7185',   // MA50
        purple: '#c084fc', // Forecast
        slateGrid: '#1e293b', 
        slateTooltip: '#0f172a'
    },

    // 1. FIXED: Bulletproof date formatting to prevent raw ISO strings on the axis
    _formatDateTime(val) {
        if (!val) return '';
        
        let valStr = String(val).trim();
        valStr = valStr.replace(/^["']|["']$/g, ''); // Strip quotes if any
        
        let dateObj = new Date(valStr);
        
        if (isNaN(dateObj.getTime())) {
            if (!isNaN(Number(valStr))) {
                const numVal = Number(valStr);
                dateObj = new Date(numVal < 10000000000 ? numVal * 1000 : numVal);
            } else {
                return valStr; // Fallback only if completely unparseable
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

    _normalizeCandlePoint(point, fallbackClose = 0, fallbackOpen = null) {
        const closeRaw = Number(point?.close);
        const safeClose = Number.isFinite(closeRaw) && closeRaw > 0
            ? closeRaw
            : (Number(fallbackClose) > 0 ? Number(fallbackClose) : 0.01);

        const openRaw = Number(point?.open);
        const safeOpen = Number.isFinite(openRaw) && openRaw > 0
            ? openRaw
            : (Number(fallbackOpen) > 0 ? Number(fallbackOpen) : safeClose);

        const highRaw = Number(point?.high);
        const lowRaw = Number(point?.low);

        const safeHigh = Number.isFinite(highRaw) && highRaw > 0
            ? Math.max(highRaw, safeOpen, safeClose)
            : Math.max(safeOpen, safeClose);
        const safeLow = Number.isFinite(lowRaw) && lowRaw > 0
            ? Math.min(lowRaw, safeOpen, safeClose)
            : Math.min(safeOpen, safeClose);

        return { open: safeOpen, high: safeHigh, low: safeLow, close: safeClose };
    },

    _collectValidOHLCValues(points) {
        const vals = [];
        points.forEach(p => {
            if (!p) return;
            [p.open, p.high, p.low, p.close].forEach(v => {
                const n = Number(v);
                if (Number.isFinite(n) && n > 0) vals.push(n);
            });
        });
        return vals;
    },

    renderMainChart(canvasId, priceDataRaw, predictionRaw = null, chartStyle = 'line') {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        if (this.mainChart) {
            this.mainChart.destroy();
            this.mainChart = null;
        }

        if (!priceDataRaw || priceDataRaw.length === 0) {
            this._showError(canvasId, 'Awaiting price data for trajectory synthesis');
            return;
        }

        const priceData = this._decimateData(priceDataRaw, 150);
        const hasPrediction = Boolean(
            predictionRaw
            && Array.isArray(predictionRaw.predictions)
            && predictionRaw.predictions.length > 0
        );

        const ctx = canvas.getContext('2d');
        const labels = priceData.map(d => this._formatDateTime(d.date || d.timestamp || d.time));
        const candleSourceData = priceData.map(d => this._normalizeCandlePoint(d));
        const prices = candleSourceData.map(d => d.close);
        
        let predPrices = new Array(prices.length).fill(null);

        // Process Prediction Headers
        if (hasPrediction) {
            predPrices[prices.length - 1] = prices[prices.length - 1]; // Anchor line to last historical close
            let prevClose = prices[prices.length - 1];
            predictionRaw.predictions.forEach(p => {
                const predPoint = this._normalizeCandlePoint(p, prevClose, prevClose);
                labels.push(this._formatDateTime(p.date || p.timestamp || p.time));
                prices.push(null);
                predPrices.push(predPoint.close);
                candleSourceData.push(predPoint);
                prevClose = predPoint.close;
            });
        }

        const validPrices = this._collectValidOHLCValues(candleSourceData).concat(
            predPrices.filter(p => p !== null && p > 0)
        );
        const minPrice = validPrices.length ? Math.min(...validPrices) : 0;
        const maxPrice = validPrices.length ? Math.max(...validPrices) : 100;
        
        const priceRange = Math.max(maxPrice - minPrice, maxPrice * 0.01, 1);
        const bottomPadding = priceRange * 0.05; 
        const topPadding = priceRange * 0.15;    

        const datasets = [];

        if (chartStyle === 'candle') {
            const wickData = [];
            const bodyData = [];
            const candleColors = [];

            // Render full candle stream: historical + projected OHLC values.
            candleSourceData.forEach((d, idx) => {
                const o = d.open;
                const h = d.high;
                const l = d.low;
                const c = d.close;
                const isBull = c >= o;
                const isPrediction = idx >= priceData.length;
                const color = isBull
                    ? (isPrediction ? '#22d3ee' : '#34d399')
                    : (isPrediction ? '#f97316' : '#fb7185');
                
                wickData.push([l, h]);
                bodyData.push([o, c]);
                candleColors.push(color);
            });

            // 2. FIXED: Added maxBarThickness to prevent blocky oversized candles
            datasets.push(
                { type: 'bar', label: 'Wick', data: wickData, backgroundColor: candleColors, barThickness: 2, grouped: false, order: 3 },
                { type: 'bar', label: 'OHLC', data: bodyData, backgroundColor: candleColors, barPercentage: 0.8, categoryPercentage: 0.8, maxBarThickness: 16, grouped: false, order: 2 }
            );

            // 3. FIXED: Render Prediction as a forward-projecting dashed line overlay instead of fake candles
            if (hasPrediction) {
                const predLength = predictionRaw.predictions.length;
                const predLabel = predictionRaw.model_used === 'RandomForestRegressor' ? `Hypo ${predLength}D Forecast (RF)` : `Hypo ${predLength}D Forecast (PyTorch)`;

                datasets.push({
                    type: 'line',
                    label: predLabel,
                    data: predPrices,
                    borderColor: this.COLORS.purple, 
                    borderWidth: 2,
                    borderDash: [5, 5], 
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 6,
                    pointBackgroundColor: this.COLORS.slateTooltip,
                    pointBorderColor: this.COLORS.purple,
                    pointBorderWidth: 2,
                    order: 1
                });
            }
            
        } else {
            const ma20 = priceData.map(d => d.ma20 || null);
            const ma50 = priceData.map(d => d.ma50 || null);
            
            // Pad the MAs so array length matches total labels count
            if (hasPrediction) {
                predictionRaw.predictions.forEach(() => {
                    ma20.push(null);
                    ma50.push(null);
                });
            }

            datasets.push(
                { label: 'Close Price', data: prices, borderColor: this.COLORS.sky, borderWidth: 2, tension: 0.4, pointRadius: 0, pointHoverRadius: 6, pointBackgroundColor: this.COLORS.slateTooltip, pointBorderColor: this.COLORS.sky, pointBorderWidth: 2, order: 4 },
                { label: 'MA20', data: ma20, borderColor: this.COLORS.amber, borderWidth: 1.5, borderDash: [5, 5], tension: 0.4, pointRadius: 0, pointHoverRadius: 4, order: 3 },
                { label: 'MA50', data: ma50, borderColor: this.COLORS.rose, borderWidth: 1.5, borderDash: [2, 2], tension: 0.4, pointRadius: 0, pointHoverRadius: 4, order: 2 }
            );

            // Appending Inference Output as Line
            if (hasPrediction) {
                const predLength = predictionRaw.predictions.length;
                const predLabel = predictionRaw.model_used === 'RandomForestRegressor' ? `Hypo ${predLength}D Forecast (RF)` : `Hypo ${predLength}D Forecast (PyTorch)`;

                datasets.push({
                    type: 'line',
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
                    pointBorderWidth: 2,
                    order: 1
                });
            }
        }

        this._initializeChart(ctx, canvasId, chartStyle, labels, datasets, minPrice - bottomPadding, maxPrice + topPadding, candleSourceData, 'main');
    },
    
    // Dedicated isolated chart variant for the prediction modal
    renderPredictModalChart(canvasId, priceDataRaw, predictionRaw = null, chartStyle = 'candle') {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        if (this.modalChart) {
            this.modalChart.destroy();
            this.modalChart = null;
        }

        if (!priceDataRaw || priceDataRaw.length === 0) {
            this._showError(canvasId, 'Awaiting price data for trajectory synthesis');
            return;
        }

        const lastEntry = priceDataRaw[priceDataRaw.length - 1];
        const lastDate = new Date(lastEntry.date || lastEntry.timestamp || lastEntry.time);
        const cutoff = new Date(lastDate);
        cutoff.setDate(cutoff.getDate() - 30);
        const trimmed = priceDataRaw.filter(d => new Date(d.date || d.timestamp || d.time) >= cutoff);
        const priceData = trimmed.length > 0 ? trimmed : priceDataRaw.slice(-30);
        const hasPrediction = Boolean(
            predictionRaw
            && Array.isArray(predictionRaw.predictions)
            && predictionRaw.predictions.length > 0
        );

        const ctx = canvas.getContext('2d');
        const labels = priceData.map(d => this._formatDateTime(d.date || d.timestamp || d.time));
        const candleSourceData = priceData.map(d => this._normalizeCandlePoint(d));
        const prices = candleSourceData.map(d => d.close);
        
        let predPrices = new Array(prices.length).fill(null);
        predPrices[prices.length - 1] = prices[prices.length - 1]; // Anchor

        if (hasPrediction) {
            let prevClose = prices[prices.length - 1];
            predictionRaw.predictions.forEach(p => {
                const predPoint = this._normalizeCandlePoint(p, prevClose, prevClose);
                labels.push(this._formatDateTime(p.date || p.timestamp || p.time));
                prices.push(null);
                predPrices.push(predPoint.close);
                candleSourceData.push(predPoint);
                prevClose = predPoint.close;
            });
        }

        const validPrices = this._collectValidOHLCValues(candleSourceData).concat(
            predPrices.filter(p => p !== null && p > 0)
        );
        const minPrice = validPrices.length ? Math.min(...validPrices) : 0;
        const maxPrice = validPrices.length ? Math.max(...validPrices) : 100;
        const priceRange = Math.max(maxPrice - minPrice, maxPrice * 0.01, 1);
        const padding = priceRange * 0.15;

        const datasets = [];

        if (chartStyle === 'candle') {
            const wickData = [];
            const bodyData = [];
            const candleColors = [];

            candleSourceData.forEach((d, idx) => {
                const o = d.open;
                const h = d.high;
                const l = d.low;
                const c = d.close;
                const isPrediction = idx >= priceData.length;
                wickData.push([l, h]);
                bodyData.push([o, c]);
                if (c >= o) {
                    candleColors.push(isPrediction ? '#22d3ee' : '#34d399');
                } else {
                    candleColors.push(isPrediction ? '#f97316' : '#fb7185');
                }
            });

            datasets.push(
                { type: 'bar', label: 'Wick', data: wickData, backgroundColor: candleColors, barThickness: 2, grouped: false, order: 3 },
                { type: 'bar', label: 'OHLC', data: bodyData, backgroundColor: candleColors, barPercentage: 0.8, categoryPercentage: 0.8, maxBarThickness: 24, grouped: false, order: 2 }
            );

            // Overlay predictions as a beautiful dashed trajectory line
            if (hasPrediction) {
                datasets.push({
                    type: 'line', label: `Forecast (${predictionRaw.predictions.length}D)`, data: predPrices, 
                    borderColor: this.COLORS.purple, borderWidth: 2, borderDash: [5, 5], tension: 0.4, 
                    pointRadius: 2, pointHoverRadius: 6, pointBackgroundColor: this.COLORS.slateTooltip, 
                    pointBorderColor: this.COLORS.purple, pointBorderWidth: 2, order: 1
                });
            }
        } else {
            datasets.push({ 
                type: 'line', label: 'Historical Close', data: prices, borderColor: this.COLORS.sky, borderWidth: 2, tension: 0.4, pointRadius: 0, pointHoverRadius: 6, pointBackgroundColor: this.COLORS.slateTooltip, pointBorderColor: this.COLORS.sky, pointBorderWidth: 2, order: 2 
            });

            if (hasPrediction) {
                datasets.push({
                    type: 'line', label: `Forecast (${predictionRaw.predictions.length}D)`, data: predPrices, borderColor: this.COLORS.purple, borderWidth: 2, borderDash: [5, 5], tension: 0.4, pointRadius: 0, pointHoverRadius: 6, pointBackgroundColor: this.COLORS.slateTooltip, pointBorderColor: this.COLORS.purple, pointBorderWidth: 2, order: 1
                });
            }
        }

        this._initializeChart(ctx, canvasId, chartStyle, labels, datasets, Math.max(0, minPrice - padding), maxPrice + padding, candleSourceData, 'modal');
    },

    // Extracted Initialization logic to DRY up the tooltip/options configuration
    _initializeChart(ctx, canvasId, chartStyle, labels, datasets, min, max, priceData, target) {
        const config = {
            type: chartStyle === 'candle' ? 'bar' : 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                normalized: true, 
                animation: { duration: 1200, easing: 'easeOutQuart' },
                layout: { padding: { bottom: 0 } },
                interaction: { mode: 'index', intersect: false },
                plugins: { 
                    legend: { 
                        display: true, 
                        labels: { 
                            color: '#94a3b8', usePointStyle: true, boxWidth: 8, padding: 25,
                            filter: function(item, chart) { return item.text !== 'Wick'; }
                        } 
                    },
                    tooltip: {
                        backgroundColor: this.COLORS.slateTooltip,
                        titleColor: '#f8fafc', bodyColor: '#cbd5e1', borderColor: '#1e293b',
                        borderWidth: 1, padding: 12, displayColors: true,
                        callbacks: {
                            title: (context) => context[0].label,
                            label: (context) => {
                                if (context.dataset.label === 'Wick') return null;
                                
                                // Enhanced safe-tooltip handling specifically for historical OHLC
                                if (context.dataset.label === 'OHLC') {
                                    const sourceData = priceData[context.dataIndex];
                                    if (!sourceData) return null; // Safe guard for padded null spaces
                                    return [
                                        `O: ${Number(sourceData.open || sourceData.close).toLocaleString('vi-VN', {maximumFractionDigits: 2})}`,
                                        `H: ${Number(sourceData.high || sourceData.close).toLocaleString('vi-VN', {maximumFractionDigits: 2})}`,
                                        `L: ${Number(sourceData.low || sourceData.close).toLocaleString('vi-VN', {maximumFractionDigits: 2})}`,
                                        `C: ${Number(sourceData.close).toLocaleString('vi-VN', {maximumFractionDigits: 2})}`
                                    ];
                                }

                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null && context.parsed.y !== undefined) {
                                    let val = context.parsed.y;
                                    if (Array.isArray(val)) val = val[1]; 
                                    label += Number(val).toLocaleString('vi-VN', {maximumFractionDigits: 2});
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: { ticks: { color: '#64748b', maxTicksLimit: target === 'modal' ? 6 : 4, font: { weight: 300 } }, grid: { color: this.COLORS.slateGrid } },
                    y: { ticks: { color: '#64748b', maxTicksLimit: 5, font: { weight: 300 } }, grid: { color: this.COLORS.slateGrid }, min, max }
                }
            }
        };

        if (target === 'modal') {
            this.modalChart = new Chart(ctx, config);
        } else {
            this.mainChart = new Chart(ctx, config);
        }
    },

    renderIndicator(canvasId, type, dataRaw) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        if (this.indicatorChart) {
            this.indicatorChart.destroy();
            this.indicatorChart = null;
        }

        if (!dataRaw || dataRaw.length === 0) {
            this._showError(canvasId, 'Awaiting indicator data for matrix rendering');
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
                responsive: true, maintainAspectRatio: false, normalized: true, animation: { duration: 1200, easing: 'easeOutQuart' },
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
                        grid: { color: this.COLORS.slateGrid }, grace: '10%'
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