/**
 * Independent Controller for the Gemini AI Analysis Popup logic and UI injections
 */
window.AIAssistant = {
    lastDetailSymbol: null,
    lastCompareHash: null,
    popupTimeout: null,

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
        // Check active tab via DOM since window.currentTabIndex is isolated in the IIFE bundle
        const detailBtn = document.querySelector('.tab-btn[data-target="detail"]');
        if (!detailBtn || !detailBtn.classList.contains('text-white')) return; 
        
        const cache = window.StockCache;
        if (!cache || !cache.currentSymbol || !cache.summary) return;

        // Ensure price arrays are successfully hydrated
        const currentRange = window.StockDetail && window.StockDetail.range ? window.StockDetail.range : 'ALL';

        if (cache.prices && cache.prices[currentRange] && cache.prices[currentRange].length > 0) {
            if (this.lastDetailSymbol !== cache.currentSymbol) {
                this.lastDetailSymbol = cache.currentSymbol;
                const latestClose = cache.prices[currentRange][cache.prices[currentRange].length - 1].close;
                
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
        // Check active tab via DOM since window.currentTabIndex is isolated in the IIFE bundle
        const compareBtn = document.querySelector('.tab-btn[data-target="compare"]');
        if (!compareBtn || !compareBtn.classList.contains('text-white')) return;

        const store = window.CompareStore;
        if (!store || !store.symbols || store.symbols.length < 2) return;
        if (!store.dataCache || !store.dataCache.data) return;

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

        console.log(`[AIAssistant] Queuing popup for ${mode} analysis. Waiting 3 seconds...`);

        // Clear any existing pending popups to prevent overlap if user switches fast
        if (this.popupTimeout) {
            clearTimeout(this.popupTimeout);
        }

        this.popupTimeout = setTimeout(() => {
            console.log(`[AIAssistant] 3 seconds passed. Displaying popup for ${mode} analysis now.`);

            // Inject the bot.jpg asset dynamically
            const avatarContainer = popup.querySelector('.w-10.h-10');
            if (avatarContainer && !avatarContainer.querySelector('img')) {
                avatarContainer.innerHTML = '<img src="../assets/bot.jpg" class="w-full h-full object-cover rounded-full shadow-[0_0_10px_rgba(168,85,247,0.4)]" alt="Hypo AI">';
                avatarContainer.classList.remove('bg-purple-500/20', 'border');
            }

            const textElem = popup.querySelector('p');
            const btnYes = document.getElementById('ai-popup-yes');
            const btnNo = document.getElementById('ai-popup-no');

            textElem.innerText = mode === 'detail' 
                ? `Would you like AI assistance analyzing this stock?`
                : `Would you like AI to rank these compared stocks?`;

            // Explicitly trigger translation classes for Tailwind to pick up the transition
            popup.classList.remove('translate-y-[150%]');
            popup.classList.add('translate-y-0');

            // Clear existing event listeners to prevent cross-contamination
            const newBtnYes = btnYes.cloneNode(true);
            const newBtnNo = btnNo.cloneNode(true);
            btnYes.replaceWith(newBtnYes);
            btnNo.replaceWith(newBtnNo);

            newBtnNo.addEventListener('click', () => {
                popup.classList.remove('translate-y-0');
                popup.classList.add('translate-y-[150%]');
            });

            newBtnYes.addEventListener('click', () => {
                popup.classList.remove('translate-y-0');
                popup.classList.add('translate-y-[150%]');
                if (mode === 'detail') this.runDetailAnalysis(data);
                if (mode === 'compare') this.runCompareAnalysis(data);
            });
        }, 3000);
    },

    async runDetailAnalysis(data) {
        const userMsg = `Analyze details of stock '${data.symbol}'`;
        
        // Dispatch the message directly through the centralized socket pipeline
        // This ensures the backend `ai_agent.py` receives and processes the contextual request
        if (window.ChatUI && window.ChatUI.sendAIMessage) {
            window.ChatUI.sendAIMessage(userMsg);
        } else if (window.ChatUI) {
            // Fallback
            window.ChatUI.addMessage(userMsg, true);
        }

        // Switch to the chat view smoothly after dispatching so context states are captured cleanly
        if (window.navigateTo) window.navigateTo('chat');
    },

    async runCompareAnalysis(data) {
        const symbolsStr = data.symbols.join(', ');
        const userMsg = `Compare stocks '${symbolsStr}'`;
        
        // Dispatch the message directly through the centralized socket pipeline
        // This ensures the backend `ai_agent.py` receives and processes the contextual request
        if (window.ChatUI && window.ChatUI.sendAIMessage) {
            window.ChatUI.sendAIMessage(userMsg);
        } else if (window.ChatUI) {
            // Fallback
            window.ChatUI.addMessage(userMsg, true);
        }

        // Switch to the chat view smoothly after dispatching so context states are captured cleanly
        if (window.navigateTo) window.navigateTo('chat');
    }
};