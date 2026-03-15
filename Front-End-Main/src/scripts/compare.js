/**
 * Handles the "Compare" tab logic with Smooth Grid Transitions & Active Highlight Tracking
 */

window.CompareStore = {
    symbols: [],
    dataCache: null, 
    activeMetric: 'close',
    activeTimeRange: '1Y',
    highlightedSymbol: null,
    // Semantic data visualization palette built via OKLCH approximations
    // L ~ 70%, C ~ 0.15 across different hues
    colors: ['#38bdf8', '#c084fc', '#34d399'] // sky-400, purple-400, emerald-400
};

window.CompareView = {
    chartInstance: null,
    searchDebounce: null,
    modalState: { tempSymbols: [], searchResults: [] },

    init() {
        this.bindModalEvents();
        this.bindControlEvents();
    },

    onTabEnter() {
        if (CompareStore.symbols.length === 0) {
            this.openModal();
        }
    },

    /* --- MODAL LOGIC --- */
    openModal() {
        document.getElementById('compare-modal').classList.remove('hidden');
        document.getElementById('compare-modal').style.display = 'flex';
        setTimeout(() => { document.getElementById('compare-modal').style.opacity = '1'; }, 10);
        
        this.modalState.tempSymbols = [...CompareStore.symbols];
        document.getElementById('compare-search-input').value = '';
        document.getElementById('compare-search-results').innerHTML = '';
        document.getElementById('compare-search-results').classList.add('hidden');
        this.renderModalChips();
    },

    closeModal() {
        document.getElementById('compare-modal').style.opacity = '0';
        setTimeout(() => { 
            document.getElementById('compare-modal').style.display = 'none'; 
            document.getElementById('compare-modal').classList.add('hidden');
        }, 300);
    },

    bindModalEvents() {
        document.getElementById('compare-reselect-btn').addEventListener('click', () => this.openModal());
        document.getElementById('compare-cancel-btn').addEventListener('click', () => {
            this.closeModal();
            if (CompareStore.symbols.length === 0) window.navigateTo('overview');
        });

        document.getElementById('compare-confirm-btn').addEventListener('click', () => {
            CompareStore.symbols = [...this.modalState.tempSymbols];
            this.closeModal();
            this.fetchAndRenderComparison();
        });

        const searchInput = document.getElementById('compare-search-input');
        searchInput.addEventListener('input', (e) => {
            clearTimeout(this.searchDebounce);
            this.searchDebounce = setTimeout(() => this.executeSearch(e.target.value), 300);
        });
    },

    async executeSearch(query) {
        const resultsContainer = document.getElementById('compare-search-results');
        if (!query.trim()) {
            resultsContainer.classList.add('hidden');
            return;
        }

        const res = await window.ApiClient.searchStocks(query);
        if (!res || !res.items || res.items.length === 0) {
            resultsContainer.innerHTML = `<div class="p-4 text-sm text-slate-500 text-center">No results found.</div>`;
            resultsContainer.classList.remove('hidden');
            return;
        }

        this.modalState.searchResults = res.items;
        
        resultsContainer.innerHTML = res.items.map(item => `
            <div class="search-result-item p-3 hover:bg-slate-800 cursor-pointer flex justify-between items-center transition-colors border-b border-slate-800 last:border-0" data-sym="${item.stock_code}">
                <div>
                    <span class="font-bold text-slate-50 mr-2">${item.stock_code}</span>
                    <span class="text-xs text-slate-500">${item.company_name}</span>
                </div>
                <button class="add-sym-btn w-6 h-6 rounded-full bg-slate-800 border border-slate-700 text-slate-300 flex items-center justify-center transition-colors pointer-events-none">+</button>
            </div>
        `).join('');
        
        resultsContainer.classList.remove('hidden');

        resultsContainer.querySelectorAll('.search-result-item').forEach(row => {
            row.addEventListener('click', (e) => {
                const sym = e.currentTarget.dataset.sym;
                this.addSymbolToModal(sym);
            });
        });
    },

    addSymbolToModal(sym) {
        if (this.modalState.tempSymbols.includes(sym)) return;
        if (this.modalState.tempSymbols.length >= 3) {
            window.showToast("Maximum 3 stocks allowed for comparison.", "error");
            return;
        }
        this.modalState.tempSymbols.push(sym);
        this.renderModalChips();
    },

    removeSymbolFromModal(sym) {
        this.modalState.tempSymbols = this.modalState.tempSymbols.filter(s => s !== sym);
        this.renderModalChips();
    },

    renderModalChips() {
        const container = document.getElementById('compare-selected-chips');
        document.getElementById('compare-selected-count').textContent = this.modalState.tempSymbols.length;
        
        container.innerHTML = this.modalState.tempSymbols.map((sym, i) => `
            <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700">
                <span class="w-2 h-2 rounded-full" style="background-color: ${CompareStore.colors[i]}"></span>
                <span class="text-sm font-medium text-slate-100">${sym}</span>
                <button class="ml-1 text-slate-500 hover:text-slate-300 transition-colors" onclick="CompareView.removeSymbolFromModal('${sym}')">✕</button>
            </div>
        `).join('');

        document.getElementById('compare-confirm-btn').disabled = this.modalState.tempSymbols.length < 1;
    },

    /* --- DATA & CONTROL ENGINE --- */
    bindControlEvents() {
        const dropdownBtn = document.getElementById('metric-dropdown-btn');
        const dropdownMenu = document.getElementById('metric-dropdown-menu');

        if(dropdownBtn && dropdownMenu) {
            dropdownBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const isExpanded = dropdownMenu.classList.contains('opacity-100');
                if (isExpanded) {
                    this.closeMetricDropdown();
                } else {
                    this.openMetricDropdown();
                }
            });

            document.addEventListener('click', (e) => {
                if (!dropdownBtn.contains(e.target) && !dropdownMenu.contains(e.target)) {
                    this.closeMetricDropdown();
                }
            });
        }
    },

    openMetricDropdown() {
        const menu = document.getElementById('metric-dropdown-menu');
        const icon = document.getElementById('metric-dropdown-icon');
        if(menu && icon) {
            menu.classList.remove('opacity-0', 'grid-rows-[0fr]');
            menu.classList.add('opacity-100', 'grid-rows-[1fr]');
            icon.style.transform = 'rotate(180deg)';
        }
    },

    closeMetricDropdown() {
        const menu = document.getElementById('metric-dropdown-menu');
        const icon = document.getElementById('metric-dropdown-icon');
        if(menu && icon) {
            menu.classList.remove('opacity-100', 'grid-rows-[1fr]');
            menu.classList.add('opacity-0', 'grid-rows-[0fr]');
            icon.style.transform = 'rotate(0deg)';
        }
    },

    selectMetric(m) {
        CompareStore.activeMetric = m;
        document.getElementById('metric-dropdown-label').textContent = m.toUpperCase().replace(/_/g, ' ');
        this.closeMetricDropdown();
        this.renderMetricOptions(CompareStore.dataCache.available_metrics);
        this.updateChartData();
    },

    async fetchAndRenderComparison() {
        if (CompareStore.symbols.length === 0) return;
        
        document.getElementById('compare-loading').classList.remove('hidden');
        document.getElementById('compare-chart-subtitle').textContent = CompareStore.symbols.join(' vs ');

        const payload = await window.ApiClient.getComparison(CompareStore.symbols, CompareStore.activeTimeRange);
        document.getElementById('compare-loading').classList.add('hidden');

        if (!payload || !payload.data) {
            window.showToast("Failed to fetch comparison data.", "error");
            return;
        }

        CompareStore.dataCache = payload;
        
        this.populateControls(payload);
        this.renderLegend();
        this.updateHighlightSlider();
        this.updateChartData();
    },

    changeTimeRange(range) {
        if (CompareStore.activeTimeRange === range) return;
        CompareStore.activeTimeRange = range;
        this.fetchAndRenderComparison();
    },

    renderMetricOptions(metrics) {
        const menu = document.getElementById('metric-dropdown-menu');
        if (!menu) return;
        
        menu.innerHTML = metrics.map(m => {
            const isActive = m === CompareStore.activeMetric;
            const activeClasses = isActive ? 'bg-slate-700 text-sky-400 font-medium' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200';
            return `<div class="px-4 py-3 text-sm cursor-pointer transition-colors ${activeClasses}" onclick="CompareView.selectMetric('${m}')">${m.toUpperCase().replace(/_/g, ' ')}</div>`;
        }).join('');
    },

    populateControls(payload) {
        const trContainer = document.getElementById('compare-time-ranges');
        trContainer.innerHTML = payload.available_time_ranges.map(tr => `
            <button class="px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${CompareStore.activeTimeRange === tr ? 'bg-sky-500 text-slate-950 border-sky-400' : 'bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800 hover:text-slate-200'}" onclick="CompareView.changeTimeRange('${tr}')">${tr}</button>
        `).join('');

        if (!payload.available_metrics.includes(CompareStore.activeMetric) && payload.available_metrics.length > 0) {
            CompareStore.activeMetric = payload.available_metrics[0];
        }
        
        const labelEl = document.getElementById('metric-dropdown-label');
        if (labelEl) labelEl.textContent = CompareStore.activeMetric.toUpperCase().replace(/_/g, ' ');
        
        this.renderMetricOptions(payload.available_metrics);
    },

    renderLegend() {
        const container = document.getElementById('compare-legend');
        const itemsHtml = CompareStore.symbols.map((sym, i) => {
            const isHighlighted = CompareStore.highlightedSymbol === sym;
            const opacity = (!CompareStore.highlightedSymbol || isHighlighted) ? '1' : '0.4';
            const textClass = isHighlighted ? 'text-contrast-fix' : 'text-slate-300';
            const shadowClass = isHighlighted ? '' : 'shadow-sm';
            
            return `
            <div id="legend-item-${sym}" class="compare-legend-item flex items-center justify-between p-3 rounded-xl border border-slate-800 cursor-pointer transition-all ${shadowClass} relative z-10 hover:bg-slate-800" style="opacity: ${opacity};" onclick="CompareView.toggleHighlight('${sym}')">
                <div class="flex items-center gap-3">
                    <div class="w-3 h-3 rounded-full shadow-[0_0_8px_rgba(0,0,0,0.5)] transition-shadow" style="background-color: ${CompareStore.colors[i]}; box-shadow: 0 0 ${isHighlighted ? '15px' : '8px'} ${CompareStore.colors[i]}"></div>
                    <span class="font-bold tracking-wide transition-colors ${textClass}">${sym}</span>
                </div>
                ${isHighlighted ? `<span class="text-[10px] bg-slate-950/40 text-slate-50 px-2 py-0.5 rounded uppercase tracking-wider shadow-inner font-medium">Locked</span>` : ''}
            </div>
            `;
        }).join('');
        
        const sliderHtml = `<div id="highlight-slider" class="absolute left-0 w-full rounded-xl transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] animated-gradient opacity-0 pointer-events-none z-0"></div>`;
        container.innerHTML = sliderHtml + itemsHtml;
    },

    updateHighlightSlider() {
        const slider = document.getElementById('highlight-slider');
        if(!slider) return;

        if (CompareStore.highlightedSymbol) {
            const activeItem = document.getElementById(`legend-item-${CompareStore.highlightedSymbol}`);
            if(activeItem) {
                slider.style.top = activeItem.offsetTop + 'px';
                slider.style.height = activeItem.offsetHeight + 'px';
                slider.classList.remove('opacity-0');
                slider.classList.add('opacity-100');
            }
        } else {
            slider.classList.remove('opacity-100');
            slider.classList.add('opacity-0');
        }
    },

    toggleHighlight(sym) {
        CompareStore.highlightedSymbol = CompareStore.highlightedSymbol === sym ? null : sym;
        this.renderLegend();
        this.updateHighlightSlider();
        this.applyChartHighlight();
    },

    applyChartHighlight() {
        if (!this.chartInstance) return;
        const focus = CompareStore.highlightedSymbol;
        
        this.chartInstance.data.datasets.forEach((ds, i) => {
            const isFocused = !focus || ds.label === focus;
            const baseColor = CompareStore.colors[i];
            ds.borderColor = isFocused ? baseColor : baseColor + '40'; 
            ds.borderWidth = isFocused ? (focus ? 3 : 2) : 1;
        });
        this.chartInstance.update('none'); 
    },

    updateChartData() {
        if (!CompareStore.dataCache) return;
        const metricData = CompareStore.dataCache.data[CompareStore.activeMetric] || [];
        
        const labels = metricData.map(d => window.ChartManager._formatDateTime(d.time));
        const datasets = CompareStore.symbols.map((sym, i) => {
            const isFocused = !CompareStore.highlightedSymbol || CompareStore.highlightedSymbol === sym;
            const baseColor = CompareStore.colors[i];
            
            return {
                label: sym,
                data: metricData.map(d => d[sym] !== null ? d[sym] : null),
                borderColor: isFocused ? baseColor : baseColor + '40',
                borderWidth: isFocused ? (CompareStore.highlightedSymbol ? 3 : 2) : 1,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointBackgroundColor: '#0f172a',
                pointBorderColor: baseColor,
                pointBorderWidth: 2,
                spanGaps: true
            };
        });

        const canvas = document.getElementById('compare-chart');
        
        if (this.chartInstance) {
            this.chartInstance.destroy();
        }

        const ctx = canvas.getContext('2d');
        this.chartInstance = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                normalized: true,
                animation: { duration: 800, easing: 'easeOutQuart' },
                interaction: { mode: 'index', intersect: false },
                onHover: (e, elements) => {
                    if (elements.length > 0) this.updateHoverPanel(elements[0].index);
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                scales: {
                    x: { ticks: { color: '#64748b', maxTicksLimit: 6 }, grid: { color: '#1e293b' } },
                    y: { ticks: { color: '#64748b', maxTicksLimit: 6 }, grid: { color: '#1e293b' }, grace: '5%' }
                }
            }
        });

        if (metricData.length > 0) this.updateHoverPanel(metricData.length - 1);
    },

    updateHoverPanel(index) {
        if (index === null || !CompareStore.dataCache) return;
        const metricData = CompareStore.dataCache.data[CompareStore.activeMetric];
        if (!metricData) return;
        const dataPoint = metricData[index];
        if (!dataPoint) return;

        document.getElementById('compare-hover-date').textContent = window.ChartManager._formatDateTime(dataPoint.time);

        const container = document.getElementById('compare-hover-values');
        container.innerHTML = CompareStore.symbols.map((sym, i) => {
            const val = dataPoint[sym];
            const color = CompareStore.colors[i];
            const isHighlighted = CompareStore.highlightedSymbol === sym;
            const bg = isHighlighted ? 'bg-slate-800 shadow-md' : 'bg-slate-900';
            const border = isHighlighted ? color : '#1e293b'; // slate-800
            const formattedVal = val !== null && val !== undefined ? Number(val).toLocaleString('en-US', {maximumFractionDigits: 3}) : '--';

            return `
                <div class="px-4 py-2 rounded-xl border flex items-center gap-3 transition-all ${bg}" style="border-color: ${border}">
                    <div class="w-2 h-2 rounded-full shadow-[0_0_5px_rgba(0,0,0,0.5)]" style="background-color: ${color}"></div>
                    <span class="text-xs font-medium text-slate-500 uppercase tracking-widest">${sym}</span>
                    <span class="text-base font-mono text-slate-50 ml-2">${formattedVal}</span>
                </div>
            `;
        }).join('');
    }
};

window.addEventListener('load', () => window.CompareView.init());