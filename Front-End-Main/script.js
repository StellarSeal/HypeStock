// --- 1. STATE & CONFIG ---
let currentTabIndex = 0;
let isFocusMode = false;
const tabs = ['overview', 'stocks', 'chat'];
const pageTrack = document.getElementById('page-track');

// Stock Pagination State
let stockPage = 0;
let stockLimit = 24; // Multiples of 1, 2, 3, 4 columns for a neat grid
let isLoadingStocks = false;
let hasMoreStocks = true;
let stockSearchQuery = "";
let stockListInited = false;
let stockLoadTimeout = null;

// Lenis smooth-scroll instance
let lenis = null;

// Startup State
let isAnimationDone = false;
let isBackendReady = false;

// --- SETUP MARKED.JS ---
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

// --- 2. SOCKET & ORCHESTRATION ---
const SOCKET_URL = 'http://localhost:8000';
let socket = null;
const pendingAIRequests = new Map();

function initSocket() {
    console.log(`Attempting connection to ${SOCKET_URL}...`);
    
    try {
        socket = io(SOCKET_URL, {
            transports: ['websocket'],
            reconnectionAttempts: 3,
            timeout: 3000
        });

        socket.on('connect', () => {
            console.log("✅ Socket Connected");
            updateConnectionStatus(true);
            showToast("Connected to Backend");
            socket.emit('startup', { timestamp: Date.now() });
        });

        socket.on('disconnect', () => {
            console.warn("❌ Socket Disconnected");
            updateConnectionStatus(false);
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
    const indicator = document.getElementById('socket-indicator');
    const text = document.getElementById('connection-status');
    if (isConnected) {
        if(indicator) indicator.className = "w-2 h-2 rounded-full bg-emerald-400 animate-pulse";
        if(text) { text.textContent = "Connected (Port 8000)"; text.className = "text-lg font-bold text-emerald-400"; }
    } else {
        if(indicator) indicator.className = "w-2 h-2 rounded-full bg-rose-500";
        if(text) { text.textContent = "Disconnected"; text.className = "text-lg font-bold text-rose-500"; }
    }
}

// --- 3. SCROLLING SETUP ---
function initLenis() {
    const container = document.getElementById('stock-scroll-container');
    const content   = document.getElementById('content-wrapper');
    if (!container || !content || lenis) return;

    // Dynamic padding to allow scrolling past the footer (25% of resolution height)
    const updatePadding = () => {
        // Fix: Apply padding to the inner content wrapper so Lenis registers the height increase
        content.style.paddingBottom = `${window.innerHeight * 0.25}px`;
        if (lenis) lenis.resize();
    };
    updatePadding();
    window.addEventListener('resize', updatePadding);

    lenis = new Lenis({
        wrapper:            container,
        content:            content,
        duration:           1.2,
        easing:             (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        orientation:        'vertical',
        gestureOrientation: 'vertical',
        smoothWheel:        true,
        wheelMultiplier:    1,
        touchMultiplier:    2,
    });

    // Handle scroll events for dynamic UI (Search bar blur effect)
    lenis.on('scroll', (e) => {
        const searchContainer = document.getElementById('search-container');
        if (searchContainer) {
            if (e.scroll > 20) {
                // Scrolled down: stick to top and apply glassmorphism
                searchContainer.classList.add('bg-slate-950/80', 'backdrop-blur-md', 'border-b', 'border-slate-800', 'py-4', 'top-0');
                searchContainer.classList.remove('top-6');
            } else {
                // Top of page: return to floating appearance
                searchContainer.classList.remove('bg-slate-950/80', 'backdrop-blur-md', 'border-b', 'border-slate-800', 'py-4', 'top-0');
                searchContainer.classList.add('top-6');
            }
        }
    });

    function raf(time) {
        lenis.raf(time);
        requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);

    // Auto-resize Lenis when content changes to prevent hidden/cut-off footers
    const resizeObserver = new ResizeObserver(() => {
        if (lenis) lenis.resize();
    });
    resizeObserver.observe(content);
}

// --- 4. PAGINATED DATA FETCHING ---
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
        
        // Setup UI for Loading State
        if (loader) loader.classList.remove('hidden');
        if (pagControls) pagControls.classList.add('hidden');
        if (list) list.innerHTML = ''; // Clear layout for fresh items

        // Safety Timeout
        clearTimeout(stockLoadTimeout);
        stockLoadTimeout = setTimeout(() => {
            if (isLoadingStocks) {
                console.warn("Stock fetch timed out - resetting lock");
                isLoadingStocks = false;
                if(loader) loader.classList.add('hidden');
                updatePaginationUI();
            }
        }, 8000);

        // Fetch request
        socket.emit('request_stocks', {
            page: stockPage,
            limit: stockLimit,
            query: stockSearchQuery
        });
    } else {
        console.warn("Socket disconnected. Cannot fetch stocks.");
    }
}

function handleStockData(data) {
    clearTimeout(stockLoadTimeout);
    const loader = document.getElementById('stock-loader');
    const list = document.getElementById('stock-list');
    
    isLoadingStocks = false;
    if(loader) loader.classList.add('hidden');
    if(list) list.innerHTML = ''; 

    if (data.items) {
        if (data.items.length === 0) {
            list.innerHTML = `<div id="empty-msg" class="col-span-full text-center text-slate-500 py-10">No stocks found matching "${stockSearchQuery}".</div>`;
            hasMoreStocks = false;
            updatePaginationUI();
            return;
        }

        // Determine if we have a next page
        hasMoreStocks = data.items.length >= stockLimit;

        // Render Page Items
        const fragment = document.createDocumentFragment();
        data.items.forEach((stock, index) => {
            const card = createStockCard(stock);
            // Staggered fade in
            card.style.animation = `fadeInUp 0.3s ease forwards ${index * 0.03}s`;
            card.style.opacity = '0';
            fragment.appendChild(card);
        });
        
        list.appendChild(fragment);
        updatePaginationUI();
        
        // Reset scroll position gracefully using Lenis to prevent fighting native scroll
        if (lenis) {
            lenis.scrollTo(0, { immediate: true });
        } else {
            const container = document.getElementById('stock-scroll-container');
            if (container) container.scrollTo({ top: 0, behavior: 'auto' });
        }
    }
}

function updatePaginationUI() {
    const pagControls = document.getElementById('pagination-controls');
    const prevBtn = document.getElementById('prev-page-btn');
    const nextBtn = document.getElementById('next-page-btn');
    const numbersContainer = document.getElementById('page-numbers-container');
    
    if (!pagControls || !prevBtn || !nextBtn || !numbersContainer) return;
    
    // Hide controls if absolutely no items and on first page
    const list = document.getElementById('stock-list');
    if (stockPage === 0 && !hasMoreStocks && (!list || list.children.length === 0 || list.querySelector('#empty-msg'))) {
        pagControls.classList.add('hidden');
        return;
    }
    
    pagControls.classList.remove('hidden');
    prevBtn.disabled = stockPage === 0;
    nextBtn.disabled = !hasMoreStocks;
    
    // Render Numbered Page Buttons dynamically (show a window of up to 5 pages)
    let startPage = Math.max(0, stockPage - 2);
    let endPage = startPage + 4;
    
    // Clamp the end based on whether there's more data to show
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

    // Attach click events for direct page skipping
    numbersContainer.querySelectorAll('.page-num-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetPage = parseInt(e.target.getAttribute('data-page'));
            if (targetPage !== stockPage) {
                stockPage = targetPage;
                fetchStocks();
            }
        });
    });

    // Update pill position slightly after rendering to let DOM calculate sizes
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
    div.className = 'stock-card h-[160px] flex flex-col justify-between group';
    div.onclick = () => showToast(`Selected ${stock.stock_code}`, "info");

    div.innerHTML = `
        <div>
            <div class="flex justify-between items-start mb-2">
                <span class="text-xl font-bold text-white tracking-tight">${stock.stock_code}</span>
                <span class="text-xs text-slate-500 font-mono bg-slate-900 px-2 py-1 rounded border border-slate-800">EQ</span>
            </div>
            <h3 class="text-sm text-slate-300 font-medium leading-tight mb-4 line-clamp-2">${stock.company_name}</h3>
        </div>
        
        <div class="space-y-2 border-t border-slate-700/50 pt-3">
            <div class="flex justify-between text-xs">
                <span class="text-slate-500">Range</span>
                <span class="text-slate-300 font-mono">${stock.start_date.substring(0,4)} → ${stock.end_date.substring(0,4)}</span>
            </div>
            <div class="flex justify-between text-xs">
                <span class="text-slate-500">Entries</span>
                <span class="text-sky-400 font-mono font-bold">${stock.entry_count.toLocaleString()}</span>
            </div>
        </div>
    `;
    return div;
}

// --- 5. AI CHAT LOGIC ---
function sendAIMessage(content) {
    if (!socket || !socket.connected) {
        addMessage(content, true);
        setTimeout(() => addMessage("Offline Mode: Connect backend for AI.", false), 1000);
        return;
    }
    const seed = Math.floor(Math.random() * 2147483647);
    const modelSelect = document.getElementById('model-select');
    const payload = { type: "ai", content: content, seed: seed, model: modelSelect ? modelSelect.value : 'cloud' };
    addMessage(content, true);
    showTypingIndicator();
    const timeoutId = setTimeout(() => {
        if (pendingAIRequests.has(seed)) {
            pendingAIRequests.delete(seed);
            removeTypingIndicator();
            addMessage("No response from server.", false);
        }
    }, 15000);
    pendingAIRequests.set(seed, { timeoutId });
    socket.emit('ai', payload);
}

function handleServerMessage(data) {
    if (data.type === 'ai' && data.seed !== undefined) {
        if (pendingAIRequests.has(data.seed)) {
            const req = pendingAIRequests.get(data.seed);
            clearTimeout(req.timeoutId);
            pendingAIRequests.delete(data.seed);
            removeTypingIndicator();
            if (data.response) addMessage(data.response, false);
        }
    }
}

// Chat UI Setup
const chatInput = document.getElementById('chat-input');
const chatBtn = document.getElementById('chat-btn');
const chatMessages = document.getElementById('chat-messages');
let typingElement = null;

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
            <div class="w-8 h-8 rounded-full bg-sky-900 flex items-center justify-center text-xs text-white flex-shrink-0">AI</div>
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
        <div class="w-8 h-8 rounded-full bg-sky-900 flex items-center justify-center text-xs text-white flex-shrink-0">AI</div>
        <div class="bg-slate-800 p-4 rounded-r-xl rounded-bl-xl border border-slate-700 flex items-center gap-1.5 h-12 w-16 justify-center">
            <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
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


// --- 6. EVENT LISTENERS & NAVIGATION ---
window.addEventListener('load', () => {
    initSocket();
    initLenis();

    // Intro Overlay Animation
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
    
    // Re-adjust pill width and position if the screen is resized
    window.addEventListener('resize', () => {
        updatePaginationPill();
    });
    
    // Pagination Listeners
    document.getElementById('prev-page-btn')?.addEventListener('click', () => {
        if (stockPage > 0) {
            stockPage--;
            fetchStocks();
        }
    });
    
    document.getElementById('next-page-btn')?.addEventListener('click', () => {
        if (hasMoreStocks) {
            stockPage++;
            fetchStocks();
        }
    });
});

function tryDismissIntro() {
    const introOverlay = document.getElementById('intro-overlay');
    if (introOverlay && isAnimationDone && isBackendReady) {
        introOverlay.style.opacity = '0';
        setTimeout(() => introOverlay.remove(), 700);
    }
}

function navigateTo(targetId) {
    if (isFocusMode) return;
    const index = tabs.indexOf(targetId);
    if (index === -1) return;
    currentTabIndex = index;
    if (pageTrack) pageTrack.style.transform = `translateX(-${currentTabIndex * 100}vw)`;
    
    const searchContainer = document.getElementById('search-container');
    if (targetId === 'stocks') {
        searchContainer.classList.add('swipe-in');
        if (!stockListInited) {
            fetchStocks(true);
        }
    } else {
        searchContainer.classList.remove('swipe-in');
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

// Search Listener
const searchInput = document.getElementById('stock-search');
if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            if (query.length > 0 || (query.length === 0 && stockSearchQuery.length > 0)) {
                stockSearchQuery = query;
                fetchStocks(true); // Reset to page 0 on new search
            }
        }, 300);
    });
}

// --- 7. UTILS ---
function showToast(message, type = 'info') {
    const existing = document.querySelector('.toast-notification');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    const color = type === 'error' ? 'bg-rose-500' : 'bg-sky-400';
    toast.innerHTML = `<div class="w-2 h-2 rounded-full ${color} animate-pulse"></div><span>${message}</span>`;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('toast-fading-out');
        toast.addEventListener('animationend', () => toast.remove());
    }, 3500);
}