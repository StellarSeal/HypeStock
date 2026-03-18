/**
 * Handles REST data fetching and Gemini API Integration
 */
window.ApiClient = {
    baseUrl: '', 

    async fetchWithFallback(endpoint, options = {}) {
        try {
            const res = await fetch(this.baseUrl + endpoint, options);
            if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
            return await res.json();
        } catch (error) {
            console.warn(`REST fetch failed for ${endpoint}:`, error);
            return null;
        }
    },

    async getSystemStatus() {
        return await this.fetchWithFallback(`/system/status`);
    },

    async getSummary(symbol) {
        return await this.fetchWithFallback(`/stock/${symbol}/summary`);
    },

    async getPrice(symbol, range) {
        return await this.fetchWithFallback(`/stock/${symbol}/price?range=${range}`);
    },

    async getIndicator(symbol, type, range) {
        return await this.fetchWithFallback(`/stock/${symbol}/indicator?type=${type}&range=${range}`);
    },

    async getPrediction(symbol) {
        return await this.fetchWithFallback(`/stock/${symbol}/prediction`);
    },

    async searchStocks(query) {
        return await this.fetchWithFallback(`/stock/search?query=${encodeURIComponent(query)}`);
    },

    async getComparison(symbols, range) {
        return await this.fetchWithFallback(`/stock/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols, default_time_range: range })
        });
    },

    // --- GEMINI AI ASSISTANT FEATURES ---

    /**
     * Extracts and constructs the 24-feature Latent Embedding
     * Prevents raw OHLC streams from being sent to the LLM.
     */
    async getLatentEmbedding(symbol) {
        const prices = await this.getPrice(symbol, '1M');
        if (!prices || prices.length === 0) return null;
        const latestPrice = prices[prices.length - 1];

        // Fetching indicators dynamically to match the exact 24-feature requirement
        const metricsToFetch = [
            'dist_from_ma50', 'daily_return_1d', 'daily_return_5d', 
            'lagged_return_t1', 'lagged_return_t3', 'lagged_return_t5', 
            'rsi', 'macd', 'obv_slope_5d', 'adx', 'volatility', 
            'atr', 'daily_range', 'ema20', 'volume_ma20', 
            'volume_change_pct', 'bb_width', 'vol_close_corr_20d', 
            'cumulative_return'
        ];

        // Execute concurrent fast-fetches
        const results = await Promise.all(
            metricsToFetch.map(type => this.getIndicator(symbol, type, '1M'))
        );

        const latest = {};
        metricsToFetch.forEach((type, index) => {
            const data = results[index];
            latest[type] = (data && data.length > 0) ? data[data.length - 1].value : 0;
        });

        // Round numeric states to 3 decimals to maintain compact context limits
        const round = (val) => val != null ? Number(Number(val).toFixed(3)) : 0;

        return [
            round(latestPrice.close),
            round(latest['dist_from_ma50']),
            round(latest['daily_return_1d']),
            round(latest['daily_return_5d']),
            round(latest['lagged_return_t1']),
            round(latest['lagged_return_t3']),
            round(latest['lagged_return_t5']),
            round(latest['rsi']),
            round(latest['macd']),
            round(latest['obv_slope_5d']),
            round(latest['adx']),
            round(latest['volatility']),
            round(latest['atr']),
            round(latest['daily_range']),
            round(latestPrice.ma20),
            round(latestPrice.ma50),
            round(latest['ema20']),
            round(latestPrice.volume),
            round(latest['volume_ma20']),
            round(latest['volume_change_pct']),
            round(latest['bb_width']),
            round(latest['vol_close_corr_20d']),
            round(latest['cumulative_return']),
            round(latest['dist_from_ma50']) // Passed twice per explicit vector constraints
        ];
    },

    /**
     * Executes the Gemini request utilizing Exponential Backoff
     */
    async callGeminiAPI(prompt) {
        const apiKey = ""; // API Key provided automatically by the canvas environment
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;

        const payload = {
            contents: [{ parts: [{ text: prompt }] }],
            systemInstruction: { parts: [{ text: "You are an elite quantitative financial analyst." }] }
        };

        const maxRetries = 5;
        const delays = [1000, 2000, 4000, 8000, 16000];

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) {
                    if (res.status === 429 && attempt < maxRetries) {
                        await new Promise(r => setTimeout(r, delays[attempt]));
                        continue;
                    }
                    throw new Error(`Gemini API Error: ${res.status}`);
                }

                const data = await res.json();
                return data.candidates?.[0]?.content?.parts?.[0]?.text || "No analysis provided.";
            } catch (err) {
                if (attempt === maxRetries) {
                    return `Generation Error: Failed to generate analysis due to API limits or network instability.`;
                }
                await new Promise(r => setTimeout(r, delays[attempt]));
            }
        }
    }
};