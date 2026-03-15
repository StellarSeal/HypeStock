/**
 * Independent Controller for the Gemini AI Analysis Popup logic and UI injections
 */
window.AIAssistant = {
    lastDetailSymbol: null,
    lastCompareHash: null,

    init() {
        // Poll loosely to intercept fully loaded datasets dynamically without risking race conditions
        setInterval(() => this.checkDetailState(), 1000);
        setInterval(() => this.checkCompareState(), 1000);
    },

    getDetailCacheKey(symbol, latest_close) {
        const date = new Date().toISOString().split('T')[0];
        return `${symbol}_${latest_close}_${date}`;
    },

    getCompareCacheKey(symbols) {
        const date = new Date().toISOString().split('T')[0];
        return `compare_${symbols.slice().sort().join('_')}_${date}`;
    },

    checkDetailState() {
        if (window.currentTabIndex !== 2) return; 
        
        const cache = window.StockCache;
        if (!cache || !cache.currentSymbol || !cache.summary) return;

        // Ensure price arrays are successfully hydrated
        if (cache.prices && cache.prices['1M'] && cache.prices['1M'].length > 0) {
            if (this.lastDetailSymbol !== cache.currentSymbol) {
                this.lastDetailSymbol = cache.currentSymbol;
                const latestClose = cache.prices['1M'][cache.prices['1M'].length - 1].close;
                
                const cacheKey = this.getDetailCacheKey(cache.currentSymbol, latestClose);

                // Always trigger the popup prompt first, regardless of cache existence
                this.promptUser('detail', {
                    symbol: cache.currentSymbol,
                    name: cache.summary.company_name,
                    close: latestClose,
                    cacheKey: cacheKey
                });
            }
        }
    },

    checkCompareState() {
        if (window.currentTabIndex !== 3) return; 

        const store = window.CompareStore;
        if (!store || !store.symbols || store.symbols.length < 2) return;
        if (!store.dataCache || !store.dataCache.data || !store.dataCache.data['close']) return;

        const currentHash = store.symbols.slice().sort().join(',');
        if (this.lastCompareHash !== currentHash) {
            this.lastCompareHash = currentHash;

            const cacheKey = this.getCompareCacheKey(store.symbols);

            // Always trigger the popup prompt first, regardless of cache existence
            this.promptUser('compare', {
                symbols: store.symbols,
                cacheKey: cacheKey
            });
        }
    },

    promptUser(mode, data) {
        const popup = document.getElementById('ai-assistant-popup');
        if (!popup) return;

        // Inject the bot.jpg asset dynamically
        const avatarContainer = popup.querySelector('.w-10.h-10');
        if (avatarContainer && !avatarContainer.querySelector('img')) {
            avatarContainer.innerHTML = '<img src="../assets/bot.jpg" class="w-full h-full object-cover rounded-full shadow-[0_0_10px_rgba(168,85,247,0.4)]" alt="Hypo AI">';
            avatarContainer.classList.remove('bg-purple-500/20', 'border');
        }

        const textElem = popup.querySelector('p');
        const btnYes = document.getElementById('ai-popup-yes');
        const btnNo = document.getElementById('ai-popup-no');
        
        // Clear UI boxes initially
        if (mode === 'detail') document.getElementById('sd-ai-analysis-container').classList.add('hidden');
        if (mode === 'compare') document.getElementById('compare-ai-analysis-container').classList.add('hidden');

        textElem.innerText = mode === 'detail' 
            ? `Would you like AI assistance analyzing this stock?`
            : `Would you like AI to rank these compared stocks?`;

        popup.classList.remove('translate-y-[150%]');

        // Clear existing event listeners to prevent cross-contamination
        const newBtnYes = btnYes.cloneNode(true);
        const newBtnNo = btnNo.cloneNode(true);
        btnYes.replaceWith(newBtnYes);
        btnNo.replaceWith(newBtnNo);

        newBtnNo.addEventListener('click', () => {
            popup.classList.add('translate-y-[150%]');
        });

        newBtnYes.addEventListener('click', () => {
            popup.classList.add('translate-y-[150%]');
            if (mode === 'detail') this.runDetailAnalysis(data);
            if (mode === 'compare') this.runCompareAnalysis(data);
        });
    },

    async runDetailAnalysis(data) {
        const contId = 'sd-ai-analysis-container';
        const txtId = 'sd-ai-analysis-content';
        
        // Check cache AFTER user confirms they want the analysis
        const cachedResult = localStorage.getItem(data.cacheKey);
        if (cachedResult) {
            this.renderAnalysis(contId, txtId, cachedResult);
            return;
        }

        this.renderLoading(contId, txtId);

        const embedding = await window.ApiClient.getLatentEmbedding(data.symbol);
        if (!embedding) {
            this.renderAnalysis(contId, txtId, "Failed to compile latent embedding vector.");
            return;
        }

        const prompt = `
Analyze this stock based on the following compressed market state embedding vector.
Company: ${data.name}
Symbol: ${data.symbol}

Embedding Vector Indices:
[close, dist_from_ma50, daily_return_1d, daily_return_5d, lagged_return_t1, lagged_return_t3, lagged_return_t5, rsi, macd, obv_slope_5d, adx, rolling_vol_20d_std, atr, daily_range, ma20, ma50, ema20, volume, volume_ma20, volume_change_pct, bb_width, vol_close_corr_20d, cumulative_return, dist_from_ma50]

Embedding Vector:
${JSON.stringify(embedding)}

Please provide a structured, concise analysis detailing EXACTLY these 5 points based on the indicators above:
1. Trend direction (Using MAs)
2. Volatility (Using ATR/BB_Width)
3. Momentum (Using RSI/MACD)
4. Forecast interpretation (What the data suggests next)
5. Bullish/Bearish scenarios

Format using clean markdown.`;

        const result = await window.ApiClient.callGeminiAPI(prompt);
        if (result && !result.startsWith("Generation Error")) {
            localStorage.setItem(data.cacheKey, result);
        }
        this.renderAnalysis(contId, txtId, result);
    },

    async runCompareAnalysis(data) {
        const contId = 'compare-ai-analysis-container';
        const txtId = 'compare-ai-analysis-content';

        // Check cache AFTER user confirms they want the analysis
        const cachedResult = localStorage.getItem(data.cacheKey);
        if (cachedResult) {
            this.renderAnalysis(contId, txtId, cachedResult);
            return;
        }

        this.renderLoading(contId, txtId);

        const embeddingsArray = [];
        for (const sym of data.symbols) {
            const vector = await window.ApiClient.getLatentEmbedding(sym);
            if (vector) {
                embeddingsArray.push({ symbol: sym, embedding: vector });
            }
        }

        if (embeddingsArray.length < 2) {
            this.renderAnalysis(contId, txtId, "Failed to compile necessary embedding vectors.");
            return;
        }

        const prompt = `
Rank the following stocks based on their compiled market state embedding vectors.

Embedding Vector Indices:
[close, dist_from_ma50, daily_return_1d, daily_return_5d, lagged_return_t1, lagged_return_t3, lagged_return_t5, rsi, macd, obv_slope_5d, adx, rolling_vol_20d_std, atr, daily_range, ma20, ma50, ema20, volume, volume_ma20, volume_change_pct, bb_width, vol_close_corr_20d, cumulative_return, dist_from_ma50]

Stocks Data:
${JSON.stringify(embeddingsArray, null, 2)}

Please rank these stocks and provide a brief justification based on:
1. Trend strength
2. Momentum
3. Risk level

Format the output clearly in markdown.`;

        const result = await window.ApiClient.callGeminiAPI(prompt);
        if (result && !result.startsWith("Generation Error")) {
            localStorage.setItem(data.cacheKey, result);
        }
        this.renderAnalysis(contId, txtId, result);
    },

    renderLoading(containerId, textId) {
        const container = document.getElementById(containerId);
        const content = document.getElementById(textId);
        container.classList.remove('hidden');
        content.innerHTML = `<div class="flex items-center gap-2 text-purple-400">
            <div class="w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div> 
            Processing market state & interacting with LLM...
        </div>`;
    },

    renderAnalysis(containerId, textId, markdownText) {
        const container = document.getElementById(containerId);
        const content = document.getElementById(textId);
        container.classList.remove('hidden');
        if (typeof marked !== 'undefined') {
            content.innerHTML = marked.parse(markdownText);
        } else {
            content.innerText = markdownText;
        }
    }
};