import './scripts/cacheManager.js';
import './scripts/apiClient.js';
import './scripts/chartManager.js';
import './scripts/stockDetail.js';
import './scripts/compare.js';
import './scripts/aiAssistant.js';
import './script.js';

// 2. Fix for Broken Lenis Scroll & UI Initialization
// ES modules are deferred by default. If the browser already finished loading the page 
// before script.js was evaluated, the 'load' event was missed. We manually re-trigger it.
if (document.readyState === 'complete') {
    window.dispatchEvent(new Event('load'));
}

// 3. Initiate the AI Assistant Observer
if (window.AIAssistant) {
    window.AIAssistant.init();
}