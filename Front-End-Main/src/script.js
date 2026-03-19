let currentTabIndex = 0;
let isFocusMode = false;
const tabs = ['overview', 'stocks', 'detail', 'compare', 'chat'];
const pageTrack = document.getElementById('page-track');

let stockPage = 0;
let stockLimit = 24; 
let isLoadingStocks = false;
let hasMoreStocks = true;
let stockSearchQuery = "";
let stockListInited = false;
let stockLoadTimeout = null;

let lenisInstances = [];
let isAnimationDone = false;
let isBackendReady = false;

function generateReqId() {
    return 'req_' + Math.random().toString(36).substring(2, 10);
}

if (typeof marked !== 'undefined') {
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true
    });
}

const SOCKET_URL = window.location.origin; 
let socket = null;
const pendingAIRequests = new Map();

function initSocket() {
    console.log(`Attempting connection to Gateway at ${SOCKET_URL}...`);
    try {
        socket = io(SOCKET_URL, {
            transports: ['websocket'],
            reconnectionAttempts: 3,
            timeout: 3000
        });

        socket.on('connect', () => {
            console.log("✅ Socket Connected");
            updateConnectionStatus(true);
            
            socket.emit('startup', { 
                timestamp: Date.now(),
                request_id: generateReqId()
            });
        });

        socket.on('disconnect', () => {
            console.warn("❌ Socket Disconnected");
            updateConnectionStatus(false);
        });

        socket.on('error', (envelope) => {
            const data = envelope.payload || envelope;
            console.error("❌ Backend Error:", data);
            showToast(`Error: ${data.message || "Request failed"}`, "error");
            isLoadingStocks = false;
        });

        socket.on('startup_response', (data) => {
            console.log("🚀 Backend Ready:", data);
            isBackendReady = true;
            tryDismissIntro();
        });

        socket.on('ai_response', (data) => handleServerMessage(data));
        socket.on('stock_data', (data) => handleStockData(data));

    } catch (e) {
        console.error("Socket initialization failed", e);
    }
}

function updateConnectionStatus(isConnected) {
    const indicators = document.querySelectorAll('#socket-indicator');
    const texts = document.querySelectorAll('#connection-status');
    
    indicators.forEach(ind => {
        ind.className = isConnected ? "w-2 h-2 rounded-full bg-emerald-400 animate-pulse" : "w-2 h-2 rounded-full bg-rose-500";
    });
    
    texts.forEach(text => {
        if(text) {
            text.textContent = isConnected ? "System Online" : "Disconnected";
            text.className = isConnected ? "text-emerald-400 font-medium" : "text-rose-500 font-medium";
        }
    });
}

function initLenis() {
    const containers = document.querySelectorAll('.content-scroll-wrapper');
    if (containers.length > 0) {
        containers.forEach(container => {
            const content = container.firstElementChild;
            if (!content) return;
            
            const lenis = new Lenis({
                wrapper: container,
                content: content,
                duration: 1.2,
                easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
                orientation: 'vertical',
                gestureOrientation: 'vertical',
                smoothWheel: true,
                wheelMultiplier: 1,
                touchMultiplier: 2,
            });
            lenisInstances.push(lenis);

            if (container.parentElement.id === 'overview') {
                lenis.on('scroll', (e) => {
                    const blurOverlay = document.getElementById('overview-blur-overlay');
                    if (blurOverlay) {
                        const maxScroll = window.innerHeight * 0.7;
                        const progress = Math.min(e.scroll / maxScroll, 1);
                        
                        const blurValue = progress * 12; 
                        const opacityValue = progress * 0.36; 
                        
                        blurOverlay.style.backdropFilter = `blur(${blurValue}px)`;
                        blurOverlay.style.webkitBackdropFilter = `blur(${blurValue}px)`;
                        blurOverlay.style.backgroundColor = `rgba(2, 6, 23, ${opacityValue})`;
                    }
                });
            }

            if (container.id === 'stock-scroll-container') {
                lenis.on('scroll', (e) => {
                    const searchContainer = document.getElementById('search-container');
                    if (searchContainer) {
                        if (e.scroll > 20) {
                            searchContainer.classList.add('bg-slate-950/80', 'backdrop-blur-md', 'border-b', 'border-slate-800', 'py-4', 'top-0');
                            searchContainer.classList.remove('top-6');
                        } else {
                            searchContainer.classList.remove('bg-slate-950/80', 'backdrop-blur-md', 'border-b', 'border-slate-800', 'py-4', 'top-0');
                            searchContainer.classList.add('top-6');
                        }
                    }
                });
            }
        });

        function raf(time) {
            lenisInstances.forEach(l => l.raf(time));
            requestAnimationFrame(raf);
        }
        requestAnimationFrame(raf);
    }
}

function formatHighestPower(num) {
    if (num < 10) return num;
    const power = Math.floor(Math.log10(num));
    const factor = Math.pow(10, power);
    return Math.floor(num / factor) * factor;
}

function runRollingNumbers(contextSelector) {
    const elements = document.querySelectorAll(contextSelector + ' .rolling-number');
    elements.forEach(el => {
        if (el.dataset.animated === 'true') return;
        el.dataset.animated = 'true';
        
        const rawTarget = parseInt(el.getAttribute('data-target') || "0", 10);
        const target = el.hasAttribute('data-round') ? formatHighestPower(rawTarget) : rawTarget;
        
        if(target === 0) {
            el.textContent = "0";
            return;
        }

        const duration = 2500; 
        const startTime = performance.now();
        
        function update(time) {
            const elapsed = time - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const ease = 1 - Math.pow(1 - progress, 4);
            
            const current = Math.floor(target * ease);
            el.textContent = current.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                el.textContent = target.toLocaleString();
            }
        }
        requestAnimationFrame(update);
    });
}

function fetchStocks(reset = false) {
    if (reset) {
        stockPage = 0;
        hasMoreStocks = true;
    }

    if (isLoadingStocks) return;

    if (socket && socket.connected) {
        isLoadingStocks = true;
        stockListInited = true;

        const loader = document.getElementById('stock-loader');
        const list = document.getElementById('stock-list');
        const pagControls = document.getElementById('pagination-controls');
        
        if (loader) loader.classList.remove('hidden');
        if (pagControls) pagControls.classList.add('hidden');
        if (list) list.innerHTML = '';

        clearTimeout(stockLoadTimeout);
        stockLoadTimeout = setTimeout(() => {
            if (isLoadingStocks) {
                console.warn("Stock fetch timed out - resetting lock");
                isLoadingStocks = false;
                if(loader) loader.classList.add('hidden');
                updatePaginationUI();
            }
        }, 8000);

        socket.emit('request_stocks', {
            request_id: generateReqId(),
            page: stockPage,
            limit: stockLimit,
            query: stockSearchQuery
        });
    }
}

function handleStockData(envelope) {
    clearTimeout(stockLoadTimeout);
    const loader = document.getElementById('stock-loader');
    const list = document.getElementById('stock-list');
    
    isLoadingStocks = false;
    if(loader) loader.classList.add('hidden');
    if(list) list.innerHTML = '';

    const data = envelope.payload || envelope;
    const itemsArray = data.items || (Array.isArray(data) ? data : []);

    if (itemsArray.length === 0) {
        list.innerHTML = `<div id="empty-msg" class="col-span-full text-center text-slate-500 py-10">No stocks found matching "${stockSearchQuery}".</div>`;
        hasMoreStocks = false;
        updatePaginationUI();
        return;
    }

    hasMoreStocks = data.hasMore !== undefined ? data.hasMore : itemsArray.length >= stockLimit;

    const fragment = document.createDocumentFragment();
    itemsArray.forEach((stock, index) => {
        const card = createStockCard(stock);
        card.style.animation = `fadeInUp 0.3s ease forwards ${index * 0.03}s`;
        card.style.opacity = '0';
        fragment.appendChild(card);
    });
    
    list.appendChild(fragment);
    updatePaginationUI();
    
    const container = document.getElementById('stock-scroll-container');
    const instance = lenisInstances.find(l => l.options.wrapper === container);
    if (instance) {
        instance.scrollTo(0, { immediate: true });
    } else if (container) {
        container.scrollTo({ top: 0, behavior: 'auto' });
    }
}

function updatePaginationUI() {
    const pagControls = document.getElementById('pagination-controls');
    const prevBtn = document.getElementById('prev-page-btn');
    const nextBtn = document.getElementById('next-page-btn');
    const numbersContainer = document.getElementById('page-numbers-container');
    
    if (!pagControls || !prevBtn || !nextBtn || !numbersContainer) return;
    
    const list = document.getElementById('stock-list');
    if (stockPage === 0 && !hasMoreStocks && (!list || list.children.length === 0 || list.querySelector('#empty-msg'))) {
        pagControls.classList.add('hidden');
        return;
    }
    
    pagControls.classList.remove('hidden');
    prevBtn.disabled = stockPage === 0;
    nextBtn.disabled = !hasMoreStocks;
    
    let startPage = Math.max(0, stockPage - 2);
    let endPage = startPage + 4;
    
    if (!hasMoreStocks) {
        endPage = stockPage;
        startPage = Math.max(0, endPage - 4);
    } else if (stockPage < 2) {
        endPage = 4;
    }

    let html = '';
    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="page-num-btn ${i === stockPage ? 'active' : ''}" data-page="${i}">${i + 1}</button>`;
    }
    numbersContainer.innerHTML = html;

    numbersContainer.querySelectorAll('.page-num-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetPage = parseInt(e.target.getAttribute('data-page'));
            if (targetPage !== stockPage) {
                stockPage = targetPage;
                fetchStocks();
            }
        });
    });

    setTimeout(updatePaginationPill, 10);
}

function updatePaginationPill() {
    const container = document.getElementById('page-numbers-container');
    const pill = document.getElementById('page-pill');
    const activeBtn = container?.querySelector('.page-num-btn.active');

    if (pill && activeBtn && container) {
        const containerRect = container.getBoundingClientRect();
        const btnRect = activeBtn.getBoundingClientRect();

        pill.style.opacity = '1';
        pill.style.width = `${btnRect.width}px`;
        pill.style.transform = `translateX(${btnRect.left - containerRect.left}px)`;
    } else if (pill) {
        pill.style.opacity = '0';
    }
}

function createStockCard(stock) {
    const div = document.createElement('div');
    div.className = 'stock-card h-[160px] flex flex-col justify-between group cursor-pointer';
    
    const code = stock.stock_code || stock.symbol || "N/A";
    const name = stock.company_name || stock.name || "Unknown Co.";
    const start = stock.start_date || "--"; 
    const end = stock.end_date || "--";
    const count = stock.trading_days ? stock.trading_days.toLocaleString() : "0"; 

    div.onclick = () => {
        showToast(`Loading data for ${code}...`, "info");
        navigateTo('detail');
        if (window.StockDetail) {
            window.StockDetail.loadSymbol(code);
        } else {
            console.warn("StockDetail module not initialized.");
        }
    };

    div.innerHTML = `
        <div>
            <div class="flex justify-between items-start mb-2">
                <span class="text-xl font-bold text-white tracking-tight">${code}</span>
                <span class="text-xs text-slate-500 font-mono bg-slate-900 px-2 py-1 rounded border border-slate-800">EQ</span>
            </div>
            <h3 class="text-sm text-slate-300 font-medium leading-tight mb-4 line-clamp-2">${name}</h3>
        </div>
        
        <div class="space-y-2 border-t border-slate-700/50 pt-3">
            <div class="flex justify-between text-xs">
                <span class="text-slate-500">Range</span>
                <span class="text-slate-300 font-mono">${start} → ${end}</span>
            </div>
            <div class="flex justify-between text-xs">
                <span class="text-slate-500">Trading Days</span>
                <span class="text-sky-400 font-mono font-bold">${count}</span>
            </div>
        </div>
    `;
    return div;
}

function sendAIMessage(content) {
    if (!socket || !socket.connected) {
        addMessage(content, true);
        setTimeout(() => addMessage("Offline Mode: Connect backend for AI.", false), 1000);
        return;
    }

    // Obfuscated Payload Construction - Passed as formatted JSON string to Backend
    let contextObj = {
        details: "",
        compare: "",
        graph_data: ""
    };

    if (currentTabIndex === 2 && window.StockDetail && window.StockDetail.symbol) {
        contextObj.details = window.StockDetail.symbol;
        if (window.StockCache && window.StockCache.summary && window.StockCache.summary.metrics) {
            const m = window.StockCache.summary.metrics;
            contextObj.details += ` (High: ${m.highest_close}, Low: ${m.lowest_close}, Vol: ${m.average_volume})`;
        }
        
        // COMPRESSION: Add dense graph/metrics data representation to consume least tokens
        const range = window.StockDetail.range || '1M';
        if (window.StockCache && window.StockCache.prices && window.StockCache.prices[range]) {
            const prices = window.StockCache.prices[range];
            // Get last 30 data points, map into tight string
            const recent = prices.slice(-30);
            const c = recent.map(p => Number(p.close).toFixed(2)).join(',');
            const v = recent.map(p => {
                const vol = Number(p.volume);
                return vol >= 1000000 ? (vol/1000000).toFixed(1)+'M' : (vol/1000).toFixed(0)+'k';
            }).join(',');
            
            contextObj.graph_data = `T[30d] C[${c}] V[${v}]`;
        }
    } else if (currentTabIndex === 3 && window.CompareStore && window.CompareStore.symbols.length > 0) {
        contextObj.compare = window.CompareStore.symbols.join(', ');
    } else if (currentTabIndex === 1) {
        contextObj.details = "Browsing stock list";
    }

    const seed = Math.floor(Math.random() * 2147483647);
    const payload = { 
        request_id: generateReqId(),
        type: "ai", 
        content: content, 
        seed: seed,
        context: JSON.stringify(contextObj)
    };
    
    // UI strictly only displays the concise user query
    addMessage(content, true);
    showTypingIndicator();
    
    const timeoutId = setTimeout(() => {
        if (pendingAIRequests.has(seed)) {
            pendingAIRequests.delete(seed);
            removeTypingIndicator();
            addMessage("No response from server.", false);
        }
    }, 300000); 
    
    pendingAIRequests.set(seed, { timeoutId });
    socket.emit('ai', payload);
}

function handleServerMessage(envelope) {
    const data = envelope.payload || envelope;
    if (data.type === 'ai' && data.seed !== undefined) {
        if (pendingAIRequests.has(data.seed)) {
            const req = pendingAIRequests.get(data.seed);
            clearTimeout(req.timeoutId);
            pendingAIRequests.delete(data.seed);
            removeTypingIndicator();
            
            if (req.onResponse) {
                req.onResponse(data.response);
            } else if (data.response) {
                addMessage(data.response, false);
            }
        }
    }
}

const chatInput = document.getElementById('chat-input');
const chatBtn = document.getElementById('chat-btn');
const chatMessages = document.getElementById('chat-messages');
let typingElement = null;

const avatarHtml = `<div class="w-8 h-8 rounded-full border border-sky-500/30 flex items-center justify-center flex-shrink-0 overflow-hidden bg-slate-800"><img src="../assets/bot.jpg" alt="Hypo" class="w-full h-full object-cover"></div>`;

function addMessage(text, isUser) {
    if (!chatMessages) return;
    const div = document.createElement('div');
    div.className = 'flex gap-3 ' + (isUser ? 'justify-end' : '');
    if (isUser) {
        div.innerHTML = `<div class="bg-sky-600 text-white text-sm p-3 rounded-l-xl rounded-br-xl max-w-[80%] shadow-lg whitespace-pre-wrap">${text}</div>`;
    } else {
        let contentHtml = text;
        if (typeof marked !== 'undefined') {
            try { contentHtml = marked.parse(text); } catch (err) { console.error(err); }
        }
        div.innerHTML = `
            ${avatarHtml}
            <div class="bg-slate-800 text-slate-200 text-sm p-3 rounded-r-xl rounded-bl-xl max-w-[80%] border border-slate-700 markdown-content">${contentHtml}</div>`;
    }
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    if (typingElement) return;
    typingElement = document.createElement('div');
    typingElement.className = 'flex gap-3';
    
    typingElement.innerHTML = `
        ${avatarHtml}
        <div class="bg-slate-800 rounded-r-xl rounded-bl-xl border border-slate-700 flex items-center gap-2 px-4 h-10 w-fit">
            <span class="text-slate-400 text-xs font-medium animate-pulse tracking-wide">Typing</span>
            <div class="flex gap-1 pt-1"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>
        </div>`;
        
    chatMessages.appendChild(typingElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    if (typingElement) { typingElement.remove(); typingElement = null; }
}

if (chatBtn && chatInput) {
    const triggerChat = () => {
        const text = chatInput.value.trim();
        if(!text) return;
        chatInput.value = '';
        sendAIMessage(text);
    };
    chatBtn.addEventListener('click', triggerChat);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); triggerChat(); }
    });
}

window.addEventListener('load', () => {
    setTimeout(() => {
        if (!isBackendReady) {
            console.warn("Backend connection timed out. Forcing UI unlock.");
            isBackendReady = true; 
            tryDismissIntro();
        }
    }, 5000);
    initSocket();
    initLenis();
    
    if (window.StockDetail) window.StockDetail.init();

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                runRollingNumbers('#about-project');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    
    const aboutProjectSection = document.getElementById('about-project');
    if (aboutProjectSection) {
        observer.observe(aboutProjectSection);
    }

    const introProgress = document.getElementById('intro-progress');
    const dots = document.querySelectorAll('.intro-dot');

    if (introProgress) {
        introProgress.style.transition = 'width 1500ms linear';
        introProgress.offsetHeight;
        introProgress.style.width = '100%';
        dots.forEach((dot, index) => {
            setTimeout(() => {
                dot.style.opacity = '1';
                dot.style.transform = 'scale(1)';
            }, index * 400);
        });
        setTimeout(() => {
            isAnimationDone = true;
            tryDismissIntro();
        }, 1600);
    }
    updateNavPill();
    
    window.addEventListener('resize', updatePaginationPill);
    
    document.getElementById('prev-page-btn')?.addEventListener('click', () => {
        if (stockPage > 0) { stockPage--; fetchStocks(); }
    });
    
    document.getElementById('next-page-btn')?.addEventListener('click', () => {
        if (hasMoreStocks) { stockPage++; fetchStocks(); }
    });
});

function tryDismissIntro() {
    const introOverlay = document.getElementById('intro-overlay');
    if (introOverlay && isAnimationDone && isBackendReady) {
        introOverlay.style.opacity = '0';
        setTimeout(() => {
            introOverlay.remove();
        }, 700);
    }
}

function scrollToAbout() {
    const container = document.getElementById('overview').querySelector('.content-scroll-wrapper');
    const instance = lenisInstances.find(l => l.options.wrapper === container);
    if (instance) {
        instance.scrollTo('#about-project', { offset: -64 }); 
    } else {
        document.getElementById('about-project').scrollIntoView({ behavior: 'smooth' });
    }
}

function hideDisplayedModalsOnTabSwitch() {
    const predictModal = document.getElementById('predict-modal');
    if (predictModal && !predictModal.classList.contains('hidden')) {
        if (window.StockDetail && typeof window.StockDetail.closePredictModal === 'function') {
            window.StockDetail.closePredictModal(true);
        } else {
            predictModal.classList.add('opacity-0');
            predictModal.classList.add('hidden');
        }
    }

    const compareModal = document.getElementById('compare-modal');
    if (compareModal && !compareModal.classList.contains('hidden')) {
        compareModal.style.opacity = '0';
        compareModal.style.display = 'none';
        compareModal.classList.add('hidden');
    }

    const assistantPopup = document.getElementById('ai-assistant-popup');
    if (assistantPopup) {
        assistantPopup.classList.remove('translate-y-0');
        assistantPopup.classList.add('translate-y-[150%]');
    }
}

function navigateTo(targetId) {
    if (isFocusMode) return;
    const index = tabs.indexOf(targetId);
    if (index === -1) return;

    if (currentTabIndex !== index) {
        hideDisplayedModalsOnTabSwitch();
    }

    currentTabIndex = index;
    if (pageTrack) pageTrack.style.transform = `translateX(-${currentTabIndex * 100}vw)`;
    
    const searchContainer = document.getElementById('search-container');
    if (targetId === 'stocks') {
        searchContainer.classList.add('swipe-in');
        if (!stockListInited) fetchStocks(true);
    } else {
        searchContainer.classList.remove('swipe-in');
    }

    if (targetId === 'compare' && window.CompareView) {
        window.CompareView.onTabEnter();
    }

    const container = document.querySelector(`#${targetId} .content-scroll-wrapper`);
    if (container) {
        const instance = lenisInstances.find(l => l.options.wrapper === container);
        if (instance) instance.scrollTo(0, { immediate: true });
        else container.scrollTop = 0;
    }

    updateNavUI();
}

function updateNavUI() {
    const targetId = tabs[currentTabIndex];
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.dataset.target === targetId) {
            btn.classList.remove('text-slate-400');
            btn.classList.add('text-white');
            updateNavPill(btn);
        } else {
            btn.classList.remove('text-white');
            btn.classList.add('text-slate-400');
        }
    });
}

function updateNavPill(activeBtn) {
    const pill = document.getElementById('tab-pill');
    if (!pill) return;
    if (!activeBtn) {
        const targetId = tabs[currentTabIndex];
        activeBtn = document.querySelector(`.tab-btn[data-target="${targetId}"]`);
    }
    if (activeBtn) {
        const parentRect = activeBtn.parentElement.getBoundingClientRect();
        const elRect = activeBtn.getBoundingClientRect();
        pill.style.opacity = '1';
        pill.style.width = `${elRect.width}px`;
        pill.style.transform = `translateX(${elRect.left - parentRect.left}px)`;
    }
}

const searchInput = document.getElementById('stock-search');
if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            if (query.length > 0 || (query.length === 0 && stockSearchQuery.length > 0)) {
                stockSearchQuery = query;
                fetchStocks(true);
            }
        }, 300);
    });
}

function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }

    const toasts = container.querySelectorAll('.toast-notification');
    if (toasts.length >= 3) {
        toasts[0].remove();
    }

    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    const color = type === 'error' ? 'bg-rose-500' : 'bg-sky-400';
    toast.innerHTML = `<div class="w-2 h-2 rounded-full ${color} animate-pulse shrink-0"></div><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-fading-out');
        toast.addEventListener('animationend', () => toast.remove());
    }, 3500);
}

// Global Exports
window.navigateTo = navigateTo;
window.scrollToAbout = scrollToAbout;
window.showToast = showToast;
window.ChatUI = {
    addMessage,
    showTypingIndicator,
    removeTypingIndicator,
    sendAIMessage, // Exposed for AIAssistant integration
    sendHiddenQuery: (prompt, seed, onResponse) => {
        // Exposes a method for AIAssistant to funnel giant context prompts through the backend.
        if (!socket || !socket.connected) {
            removeTypingIndicator();
            addMessage("Offline Mode: Connect backend for AI.", false);
            return;
        }

        const payload = { 
            request_id: generateReqId(),
            type: "ai", 
            content: prompt, 
            seed: seed,
            context: "{}" // Empty to let backend directly digest the giant prompt.
        };
        
        const timeoutId = setTimeout(() => {
            if (pendingAIRequests.has(seed)) {
                pendingAIRequests.delete(seed);
                removeTypingIndicator();
                addMessage("No response from server.", false);
            }
        }, 300000); 
        
        pendingAIRequests.set(seed, { timeoutId, onResponse });
        socket.emit('ai', payload);
    }
};