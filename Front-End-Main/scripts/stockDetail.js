window.StockDetail = {
    symbol: null,
    range: 'ALL',
    indicator: 'RSI',
    debounceTimer: null,

    init() {
        this.bindEvents();
    },

    bindEvents() {
        document.querySelectorAll('.sd-range-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.updateRange(e.target.dataset.range));
        });

        document.querySelectorAll('.sd-ind-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchIndicator(e.target.dataset.ind));
        });
        
        const predHeader = document.getElementById('sd-pred-header');
        if (predHeader) {
            predHeader.addEventListener('click', () => {
                const content = document.getElementById('sd-pred-content');
                const icon = document.getElementById('sd-pred-icon');
                content.classList.toggle('hidden');
                icon.style.transform = content.classList.contains('hidden') ? 'rotate(0deg)' : 'rotate(180deg)';
            });
        }
    },

    loadSymbol(symbol) {
        this.symbol = symbol;
        StockCache.clear(symbol);
        
        const overlay = document.getElementById('detail-blur-overlay');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => overlay.style.display = 'none', 300);
        }

        this.range = 'ALL';
        this.indicator = 'RSI';
        this.updateActiveButtons();

        document.getElementById('sd-symbol').textContent = symbol;
        document.getElementById('sd-name').textContent = "Loading...";

        this.fetchAndRenderAll();
    },

    updateRange(newRange) {
        if (this.range === newRange) return;
        this.range = newRange;
        this.updateActiveButtons();

        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
            this.fetchAndRenderTimeSeriesData();
        }, 300);
    },

    switchIndicator(newInd) {
        if (this.indicator === newInd) return;
        this.indicator = newInd;
        
        document.querySelectorAll('.sd-ind-tab').forEach(tab => {
            if (tab.dataset.ind === newInd) {
                tab.classList.add('text-sky-400', 'border-sky-400');
                tab.classList.remove('text-slate-400', 'border-transparent');
            } else {
                tab.classList.remove('text-sky-400', 'border-sky-400');
                tab.classList.add('text-slate-400', 'border-transparent');
            }
        });

        this.fetchAndRenderIndicator();
    },

    updateActiveButtons() {
        document.querySelectorAll('.sd-range-btn').forEach(btn => {
            if (btn.dataset.range === this.range) {
                btn.classList.add('bg-sky-500', 'text-white', 'border-sky-400');
                btn.classList.remove('bg-slate-800', 'text-slate-400', 'border-slate-700');
            } else {
                btn.classList.remove('bg-sky-500', 'text-white', 'border-sky-400');
                btn.classList.add('bg-slate-800', 'text-slate-400', 'border-slate-700');
            }
        });
    },

    async fetchAndRenderAll() {
        this.setLoadingState(true);

        if (!StockCache.summary) {
            StockCache.summary = await ApiClient.getSummary(this.symbol);
        }
        this.renderSidebarSummary(StockCache.summary);

        await this.fetchAndRenderTimeSeriesData();
        this.setLoadingState(false);
    },

    async fetchAndRenderTimeSeriesData() {
        if (!StockCache.prices[this.range]) {
            StockCache.prices[this.range] = await ApiClient.getPrice(this.symbol, this.range);
        }
        ChartManager.renderMainChart('sd-main-chart', StockCache.prices[this.range]);

        if (!StockCache.prediction[this.range]) {
            StockCache.prediction[this.range] = await ApiClient.getPrediction(this.symbol, this.range);
        }
        this.renderPrediction(StockCache.prediction[this.range]);

        await this.fetchAndRenderIndicator();
    },

    async fetchAndRenderIndicator() {
        const key = StockCache.getIndicatorKey(this.indicator, this.range);
        if (!StockCache.indicators[key]) {
            StockCache.indicators[key] = await ApiClient.getIndicator(this.symbol, this.indicator, this.range);
        }
        ChartManager.renderIndicator('sd-ind-chart', this.indicator, StockCache.indicators[key]);
    },

    renderSidebarSummary(summary) {
        if (!summary) {
            document.getElementById('sd-name').textContent = "Data Unavailable";
            return;
        }
        
        const metrics = summary.metrics || {};
        document.getElementById('sd-name').textContent = summary.company_name || this.symbol;
        
        const startDate = ChartManager._formatDateTime(summary.start_date) || '--';
        const endDate = ChartManager._formatDateTime(summary.end_date) || '--';
        document.getElementById('sd-dates').textContent = `${startDate} to ${endDate}`;

        const safeVal = (val, suffix = '') => val !== undefined && val !== null ? `${val}${suffix}` : '--';
        
        // Formatted directly inline to use vi-VN locale representing 1000 VNĐ
        const formatPrice = (val) => val != null ? Number(val).toLocaleString('vi-VN', {maximumFractionDigits: 2}) + 'k ₫' : '--';

        document.getElementById('sd-high').textContent = formatPrice(metrics.highest_close);
        document.getElementById('sd-low').textContent = formatPrice(metrics.lowest_close);
        
        document.getElementById('sd-vol').textContent = safeVal(metrics.average_volume?.toFixed(0));
        document.getElementById('sd-volat').textContent = safeVal(metrics.volatility?.toFixed(2));
        document.getElementById('sd-return').textContent = safeVal(metrics.cumulative_return?.toFixed(2), '%');
        document.getElementById('sd-days').textContent = safeVal(metrics.trading_days);
    },

    renderPrediction(pred) {
        const container = document.getElementById('sd-pred-wrapper');
        const emptyState = document.getElementById('sd-pred-empty');

        if (!pred || pred.available === false) {
            container.classList.add('hidden');
            emptyState.classList.remove('hidden');
            return;
        }

        container.classList.remove('hidden');
        emptyState.classList.add('hidden');

        document.getElementById('sd-pred-trend').textContent = pred.trend || 'Neutral';
        document.getElementById('sd-pred-conf').textContent = `${pred.confidence || 0}%`;

        const featuresList = document.getElementById('sd-pred-features');
        featuresList.innerHTML = '';
        if (pred.top_features && Array.isArray(pred.top_features)) {
            pred.top_features.slice(0, 3).forEach(f => {
                featuresList.innerHTML += `
                    <div class="mb-2">
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-slate-300">${f.name}</span>
                            <span class="text-sky-400 font-mono">${f.importance}%</span>
                        </div>
                        <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                            <div class="bg-sky-500 h-full" style="width: ${f.importance}%"></div>
                        </div>
                    </div>
                `;
            });
        }
    },

    setLoadingState(isLoading) {
        const nameEl = document.getElementById('sd-name');
        if (isLoading) {
            nameEl.classList.add('animate-pulse');
        } else {
            nameEl.classList.remove('animate-pulse');
        }
    }
};