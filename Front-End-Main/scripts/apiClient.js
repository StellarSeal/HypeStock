/**
 * Handles REST data fetching for the Stock Detail view.
 */
window.ApiClient = {
    baseUrl: 'http://localhost:8000',

    async fetchWithFallback(endpoint) {
        try {
            const res = await fetch(this.baseUrl + endpoint);
            if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
            return await res.json();
        } catch (error) {
            console.warn(`REST fetch failed for ${endpoint}:`, error);
            return null; // Signals to the UI to handle the error state
        }
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

    async getPrediction(symbol, range) {
        return await this.fetchWithFallback(`/stock/${symbol}/prediction?range=${range}`);
    }
};