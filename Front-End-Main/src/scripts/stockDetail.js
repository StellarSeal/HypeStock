window.StockDetail = {
    symbol: null,
    range: 'ALL',
    indicator: 'RSI',
    predAlgo: 'ml', // Default model
    predTime: '7D', // Fixed prediction timeline
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
        
        // Predict Button Launch
        const predictBtn = document.getElementById('predict-price-btn');
        if (predictBtn) {
            predictBtn.addEventListener('click', () => this.openPredictModal());
        }

        // Modal Specific Binds
        const closePredictBtn = document.getElementById('predict-close-btn');
        if (closePredictBtn) {
            closePredictBtn.addEventListener('click', () => {
                const modal = document.getElementById('predict-modal');
                modal.classList.add('opacity-0');
                setTimeout(() => modal.classList.add('hidden'), 300);
            });
        }

        document.querySelectorAll('.pred-algo-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const target = e.target.closest('.pred-algo-btn');
                const selectedAlgo = target.dataset.algo;

                this.predAlgo = selectedAlgo;
                this.updatePredictionUI();
                this.triggerPrediction();
            });
        });
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

    updatePredictionUI() {
        // Toggle Algo Buttons
        document.querySelectorAll('.pred-algo-btn').forEach(btn => {
            if(btn.dataset.algo === this.predAlgo) {
                btn.classList.add('bg-sky-500/20', 'text-sky-400', 'font-medium', 'border-sky-500/30');
                btn.classList.remove('bg-slate-800', 'text-slate-400', 'border-slate-700', 'hover:bg-slate-700');
            } else {
                btn.classList.remove('bg-sky-500/20', 'text-sky-400', 'font-medium', 'border-sky-500/30');
                btn.classList.add('bg-slate-800', 'text-slate-400', 'border-slate-700', 'hover:bg-slate-700');
            }
        });

        const note = document.getElementById('predict-algo-note');
        if(note) {
            note.textContent = `Using: ${this.predAlgo === 'ml' ? 'PyTorch Transformer Model' : 'Person A Model Ensemble'}`;
        }
    },

    openPredictModal() {
        const modal = document.getElementById('predict-modal');
        modal.classList.remove('hidden');
        setTimeout(() => modal.classList.remove('opacity-0'), 10);
        
        this.updatePredictionUI();
        this.triggerPrediction();
    },

    triggerPrediction() {
        const loading = document.getElementById('predict-loading');
        const results = document.getElementById('predict-results-wrapper');
        
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        results.classList.remove('flex'); 
        
        const fetchHistoric = StockCache.prices[this.range] ? Promise.resolve() : ApiClient.getPrice(this.symbol, this.range).then(res => { StockCache.prices[this.range] = res; });
        
        // Using Fixed `7D` timeline to enforce Model Prediction capabilities
        const fetchPrediction = ApiClient.getPrediction(this.symbol, this.predTime, this.predAlgo);

        Promise.all([fetchHistoric, fetchPrediction]).then(([_, pred]) => {
            const cacheKey = `${this.predTime}_${this.predAlgo}`;
            StockCache.prediction[cacheKey] = pred;
            
            loading.classList.add('hidden');
            results.classList.remove('hidden');
            results.classList.add('flex');
            
            if (pred && pred.predictions && pred.predictions.length > 0) {
                ChartManager.renderMainChart('sd-main-chart', StockCache.prices[this.range], pred);
                ChartManager.renderPredictModalChart('modal-predict-chart', StockCache.prices[this.range], pred);

                document.getElementById('predict-desc').innerHTML = pred.message || `Model successfully projected the trajectory. Graph buffered and updated natively using ${pred.model_used}.`;
            } else {
                document.getElementById('predict-desc').innerHTML = pred.message || "Prediction failed or model unavailable.";
                ChartManager.renderPredictModalChart('modal-predict-chart', [], null); 
                ChartManager.renderMainChart('sd-main-chart', StockCache.prices[this.range]); 
            }
        }).catch(e => {
            console.error("Prediction API err:", e);
            loading.classList.add('hidden');
            results.classList.remove('hidden');
            results.classList.add('flex');
            document.getElementById('predict-desc').textContent = "Error communicating with backend.";
        });
    },

    async fetchAndRenderAll() {
        this.setLoadingState(true);

        const summaryPromise = (async () => {
            try {
                if (!StockCache.summary) {
                    StockCache.summary = await ApiClient.getSummary(this.symbol);
                }
                this.renderSidebarSummary(StockCache.summary);
            } catch (err) {
                console.error("Failed fetching summary:", err);
                this.renderSidebarSummary(null);
            }
        })();

        const timeSeriesPromise = this.fetchAndRenderTimeSeriesData();

        await Promise.allSettled([summaryPromise, timeSeriesPromise]);
        this.setLoadingState(false);
    },

    async fetchAndRenderTimeSeriesData() {
        const p1 = (async () => {
            try {
                if (!StockCache.prices[this.range]) {
                    StockCache.prices[this.range] = await ApiClient.getPrice(this.symbol, this.range);
                }
                ChartManager.renderMainChart('sd-main-chart', StockCache.prices[this.range]);
            } catch (err) {
                console.error("Failed fetching chart prices:", err);
            }
        })();

        const p3 = (async () => {
            try {
                await this.fetchAndRenderIndicator();
            } catch (err) {
                console.error("Failed fetching indicators:", err);
            }
        })();

        await Promise.allSettled([p1, p3]);
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

        const formatPrice = (val) => val != null ? Number(val).toLocaleString('vi-VN', {maximumFractionDigits: 2}) + 'k ₫' : '--';
        const formatNum = (val, maxDec = 0) => val != null ? Number(val).toLocaleString('en-US', {maximumFractionDigits: maxDec}) : '--';
        
        document.getElementById('sd-high').textContent = formatPrice(metrics.highest_close);
        document.getElementById('sd-low').textContent = formatPrice(metrics.lowest_close);
        document.getElementById('sd-vol').textContent = formatNum(metrics.average_volume, 0);
        document.getElementById('sd-volat').textContent = formatNum(metrics.volatility, 2);
        document.getElementById('sd-return').textContent = metrics.cumulative_return != null ? `${formatNum(metrics.cumulative_return, 2)}%` : '--';
        document.getElementById('sd-days').textContent = formatNum(metrics.trading_days, 0);
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