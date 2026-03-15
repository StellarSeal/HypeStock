/**
 * In-memory cache for the Stock Detail view.
 * Caches summary, prices, indicators, and predictions per symbol and range.
 */
window.StockCache = {
    currentSymbol: null,
    summary: null,
    prices: {},      // Format: { '1M': data, '3M': data }
    indicators: {},  // Format: { 'RSI_1M': data, 'MACD_1M': data }
    prediction: {},  // Format: { '1M': data }

    clear(symbol) {
        if (this.currentSymbol !== symbol) {
            this.currentSymbol = symbol;
            this.summary = null;
            this.prices = {};
            this.indicators = {};
            this.prediction = {};
            console.log(`Cache cleared for new symbol: ${symbol}`);
        }
    },

    getIndicatorKey(type, range) {
        return `${type}_${range}`;
    }
};