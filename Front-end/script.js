// --- 1. STATE MANAGEMENT ---
let currentTabIndex = 0;
let isFocusMode = false;
const tabs = ['overview', 'top', 'analysis', 'prediction', 'chat'];
const pageTrack = document.getElementById('page-track');

// --- 2. GLOBAL STYLE INJECTION (CRITICAL UI FIXES) ---
// We inject styles via JS to ensure they exist regardless of external CSS state
const styleSheet = document.createElement('style');
styleSheet.textContent = `
    /* 1. Scrollbar Hiding */
    .content-scroll-wrapper::-webkit-scrollbar { display: none !important; }
    .content-scroll-wrapper { -ms-overflow-style: none !important; scrollbar-width: none !important; }
    
    /* 2. Critical Focus Mode Styles (Ensures Layout Visibility) */
    .fullscreen-mode {
        position: fixed !important;
        top: 0; left: 0; right: 0; bottom: 0;
        width: 100vw !important; height: 100vh !important;
        z-index: 2147483647 !important;
        background: #020617 !important; /* Ensure background is opaque */
        margin: 0 !important;
        border-radius: 0 !important;
        padding: 1rem 1.5rem !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* --- OVERVIEW SPECIFIC FOCUS LAYOUT (SIDEBAR + CHART) --- */
    /* Target the Market Snapshot specifically using :has(.overview-plane) */
    .fullscreen-mode:has(.overview-plane) {
        display: grid !important;
        grid-template-columns: 280px 1fr !important; /* Left Sidebar | Main Chart */
        grid-template-rows: auto 1fr auto !important; /* Header | Content | Footer */
        gap: 24px !important;
        align-items: start !important;
    }

    /* 1. Header: Span across top */
    .fullscreen-mode:has(.overview-plane) > div:first-child {
        grid-column: 1 / -1 !important;
        grid-row: 1 !important;
        margin-bottom: 0 !important;
        padding-bottom: 1rem !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important;
    }

    /* 2. The Stats Grid (Becomes Left Sidebar) */
    .fullscreen-mode:has(.overview-plane) > .grid {
        grid-column: 1 !important;
        grid-row: 2 !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 16px !important;
        margin-top: 0 !important;
        height: 100% !important;
        overflow-y: auto !important;
    }
    
    /* Style the sidebar boxes */
    .fullscreen-mode:has(.overview-plane) > .grid > div {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(56, 189, 248, 0.1) !important;
        padding: 16px !important;
    }

    /* 3. The Chart (Becomes Main Content) */
    .fullscreen-mode:has(.overview-plane) > .group {
        grid-column: 2 !important;
        grid-row: 2 !important;
        height: 100% !important;
        width: 100% !important;
        min-height: 0 !important;
    }

    /* 4. Footer: Span across bottom */
    .fullscreen-mode:has(.overview-plane) > footer {
        grid-column: 1 / -1 !important;
        grid-row: 3 !important;
        margin-top: 0 !important;
        padding-top: 1rem !important;
        border-top: 1px solid rgba(148, 163, 184, 0.1) !important;
    }
    
    /* --- ANALYSIS SPECIFIC STYLES --- */
    /* Force Grid to become Flex row in Focus Mode */
    .fullscreen-mode #analysis-grid {
        display: flex !important;
        flex-direction: row !important;
        flex-grow: 1;
        gap: 24px;
        overflow: hidden;
    }
    
    /* Fix Sidebar Dimensions */
    .fullscreen-mode #analysis-sidebar {
        width: 280px !important;
        min-width: 280px !important;
        height: 100%;
        overflow-y: auto;
        display: flex; flex-direction: column;
        background: rgba(15, 23, 42, 0.5); /* Semi-transparent background */
    }
    
    /* Fix Chart Expansion */
    .fullscreen-mode #chart-wrapper {
        flex-grow: 1 !important;
        width: auto !important;
        height: 100% !important;
    }

    .fullscreen-mode #drag-hint { opacity: 1; }

    /* 3. Toast Notifications (Ensures Visibility) */
    .toast-notification {
        position: fixed;
        bottom: 30px; right: 30px;
        z-index: 2147483647; /* Match max safe integer to be safe */
        background: rgba(15, 23, 42, 0.95);
        color: #e2e8f0;
        padding: 12px 24px;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.8);
        display: flex; align-items: center; gap: 12px;
        animation: toastSlideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        pointer-events: none;
    }
    
    @keyframes toastSlideIn {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .toast-fading-out {
        animation: toastFadeOut 0.3s ease-in forwards;
    }
    
    @keyframes toastFadeOut {
        to { opacity: 0; transform: translateX(20px); }
    }

    /* 4. Chat Typing Animation */
    .typing-dot {
        width: 5px; height: 5px;
        background: #94a3b8;
        border-radius: 50%;
        animation: typingBounce 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typingBounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
`;
document.head.appendChild(styleSheet);

// --- 3. INTRO SEQUENCE ---
window.addEventListener('load', () => {
    const introOverlay = document.getElementById('intro-overlay');
    const introProgress = document.getElementById('intro-progress');
    const dots = document.querySelectorAll('.intro-dot');

    if (introOverlay && introProgress) {
        const totalDuration = 1500;
        const dotInterval = 400;
        const endDelay = 400;

        introProgress.style.transition = `width ${totalDuration}ms linear`;
        introProgress.offsetHeight; 
        introProgress.style.width = '100%';

        dots.forEach((dot, index) => {
            setTimeout(() => {
                dot.style.opacity = '1';
                dot.style.transform = 'scale(1)';
            }, index * dotInterval);
        });

        setTimeout(() => {
            introOverlay.style.opacity = '0';
            setTimeout(() => introOverlay.remove(), 700);
        }, totalDuration + endDelay);
    }
    
    updateNavPill();
    injectOverviewFocusButton();
});

// --- 4. NAVIGATION LOGIC ---
function navigateTo(targetId) {
    // [STATE GUARD] Option B Implementation
    // Prevent navigation while Focus Mode is active to prevent section hiding/clipping
    if (isFocusMode) return;

    const index = tabs.indexOf(targetId);
    if (index === -1) return;
    
    currentTabIndex = index;
    updatePageTrack();
    updateNavUI();
}

function updatePageTrack() {
    if (!pageTrack) return;
    pageTrack.style.transform = `translateX(-${currentTabIndex * 100}vw)`;
}

function updateNavUI() {
    const targetId = tabs[currentTabIndex];
    const buttons = document.querySelectorAll('.tab-btn');
    
    buttons.forEach(btn => {
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
        const left = elRect.left - parentRect.left;
        
        pill.style.opacity = '1';
        pill.style.width = `${elRect.width}px`;
        pill.style.transform = `translateX(${left}px)`;
    }
}

// --- 5. SWIPE GESTURE HANDLING ---
let touchStartX = 0;
let touchStartY = 0;
let touchEndX = 0;
let touchEndY = 0;
const minSwipeDistance = 50;

document.addEventListener('touchstart', (e) => {
    if (isFocusMode) return;
    touchStartX = e.changedTouches[0].screenX;
    touchStartY = e.changedTouches[0].screenY;
}, { passive: true });

document.addEventListener('touchmove', (e) => {
    // Prevent swipe nav only if in focus mode
    if (isFocusMode) {
        // Allow internal scrolling in focus mode elements if needed
    }
}, { passive: true });

document.addEventListener('touchend', (e) => {
    if (isFocusMode) return;
    touchEndX = e.changedTouches[0].screenX;
    touchEndY = e.changedTouches[0].screenY;
    handleSwipe();
}, { passive: true });

function handleSwipe() {
    const deltaX = touchEndX - touchStartX;
    const deltaY = touchEndY - touchStartY;

    if (Math.abs(deltaX) < Math.abs(deltaY)) return;

    if (Math.abs(deltaX) > minSwipeDistance) {
        if (deltaX < 0) {
            if (currentTabIndex < tabs.length - 1) {
                currentTabIndex++;
                navigateTo(tabs[currentTabIndex]);
            }
        } else {
            if (currentTabIndex > 0) {
                currentTabIndex--;
                navigateTo(tabs[currentTabIndex]);
            }
        }
    }
}

// --- 6. FOCUS MODE LOGIC (AUDITED: Option A + B Implemented) ---
// Fix Strategy: "Teleportation"
// We explicitly move the node to document.body to escape any parent hiding logic.

const placeholders = new WeakMap();

function toggleContainerFocus(container, button) {
    const isEntering = !container.classList.contains('fullscreen-mode');
    
    if (isEntering) {
        // --- ENTER FOCUS MODE ---
        
        // 1. Create Placeholder to remember DOM position
        const placeholder = document.createComment("focus-placeholder");
        container.parentNode.insertBefore(placeholder, container);
        placeholders.set(container, placeholder);

        // 2. Add Class FIRST to prepare styles (Injected styles now catch this)
        container.classList.add('fullscreen-mode');

        // 3. Teleport to Body (Escapes the overflow/transform of #page-track)
        // This is the critical fix for "Parent is Hidden" bugs.
        document.body.appendChild(container);
        
        // 4. Update State
        isFocusMode = true;
        
        // 5. Update UI Text
        const span = button.querySelector('span');
        if (span) span.textContent = "Exit Focus";

    } else {
        // --- EXIT FOCUS MODE ---
        
        // 1. Remove Styles
        container.classList.remove('fullscreen-mode');
        
        // 2. Return to Original DOM Position
        const placeholder = placeholders.get(container);
        if (placeholder && placeholder.parentNode) {
            placeholder.parentNode.insertBefore(container, placeholder);
            placeholder.remove();
            placeholders.delete(container);
        }
        
        // 3. Update State
        isFocusMode = false;
        
        // 4. Update UI Text
        const span = button.querySelector('span');
        if (span) span.textContent = "Focus Mode";

        // 5. Reset any dragged transforms
        const plane = container.querySelector('#free-plane') || container.querySelector('.overview-plane');
        if (plane) plane.style.transform = 'translate(0, 0) scale(1)';
    }
}

const techAnalysisContainer = document.getElementById('analysis-container');
const techBtn = document.getElementById('fullscreen-btn');
if (techBtn && techAnalysisContainer) {
    techBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleContainerFocus(techAnalysisContainer, techBtn);
    });
}

function injectOverviewFocusButton() {
    const overviewTab = document.getElementById('overview');
    if (!overviewTab) return;
    
    const card = overviewTab.querySelector('.glass-card');
    if (!card) return;
    if (card.querySelector('.focus-btn-dynamic')) return;

    const header = card.querySelector('h2');
    if (!header) return;
    
    const headerRow = header.parentElement;

    const newBtn = document.createElement('button');
    newBtn.className = 'focus-btn-dynamic ml-auto flex items-center gap-2 text-xs font-medium text-sky-400 hover:text-white border border-sky-500/30 hover:bg-sky-500/20 px-3 py-1.5 rounded-lg transition-all';
    newBtn.innerHTML = `
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5m11 5l-5-5m5 5v-4m0 4h-4"></path></svg>
        <span>Focus Mode</span>
    `;

    headerRow.appendChild(newBtn);

    newBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleContainerFocus(card, newBtn);
    });
}

// --- 7. DRAG LOGIC ---
let isDragging = false;
let startX, startY, translateX = 0, translateY = 0;
let currentTarget = null;

document.addEventListener('mousedown', (e) => {
    const fsContainer = e.target.closest('.fullscreen-mode');
    if (!fsContainer) return;

    // Only allow drag on specific areas
    const wrapper = e.target.closest('#chart-wrapper') || e.target.closest('.group');
    if (wrapper) {
        // Prevent dragging if clicking controls
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;

        isDragging = true;
        currentTarget = wrapper.querySelector('#free-plane') || wrapper.querySelector('.overview-plane');
        
        if (currentTarget) {
            const style = window.getComputedStyle(currentTarget);
            const matrix = new WebKitCSSMatrix(style.transform);
            translateX = matrix.m41;
            translateY = matrix.m42;
            startX = e.clientX - translateX;
            startY = e.clientY - translateY;
            wrapper.style.cursor = 'grabbing';
            e.preventDefault();
        }
    }
});

window.addEventListener('mouseup', () => {
    isDragging = false;
    currentTarget = null;
    document.querySelectorAll('#chart-wrapper, .group').forEach(w => w.style.cursor = 'crosshair');
});

window.addEventListener('mousemove', (e) => {
    if (!isDragging || !currentTarget) return;
    e.preventDefault();
    const x = e.clientX - startX;
    const y = e.clientY - startY;
    currentTarget.style.transform = `translate(${x}px, ${y}px)`;
});

// --- 8. TOAST NOTIFICATION ---

function showToast(message) {
    // 1. Remove existing to prevent stacking
    const existing = document.querySelector('.toast-notification');
    if (existing) existing.remove();

    // 2. Create Element
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    // Added 'bg-emerald-500' dot for visual polish
    toast.innerHTML = `
        <div class="w-2 h-2 rounded-full bg-sky-400 animate-pulse"></div>
        <span>${message}</span>
    `;

    // 3. Append to body (ensures z-index works relative to viewport)
    document.body.appendChild(toast);

    // 4. Auto-dismiss logic
    setTimeout(() => {
        // Add class to trigger CSS fade out
        toast.classList.add('toast-fading-out');
        
        // Remove from DOM after animation completes
        toast.addEventListener('animationend', () => {
            if (toast.parentNode) toast.remove();
        });
    }, 3500);
}

// Global Click Listener for Demos
document.addEventListener('click', (e) => {
    // Targeted selectors
    const target = e.target.closest('button, input, label, .demo-trigger');
    
    if (target) {
        // --- EXCLUSION LIST (Real Functionality) ---
        
        // 1. Navigation Tabs
        if (target.closest('.tab-btn')) return;
        
        // 2. Focus Mode Toggles
        if (target.closest('#fullscreen-btn') || target.closest('.focus-btn-dynamic')) return;
        
        // 3. Chat Interaction
        if (target.closest('#chat-btn') || target.closest('#chat-input')) return;
        
        // 4. Sidebar Checkboxes
        if (target.tagName === 'INPUT' && target.type === 'checkbox') {
            // Allow
        }

        // --- SHOW TOAST ---
        showToast("This is a demo website, so this functionality is yet to be implemented");
    }
});

// --- 9. CHAT DEMO ---
const chatInput = document.getElementById('chat-input');
const chatBtn = document.getElementById('chat-btn');
const chatMessages = document.getElementById('chat-messages');

function addMessage(text, isUser) {
    if (!chatMessages) return;
    const div = document.createElement('div');
    div.className = 'flex gap-3 ' + (isUser ? 'justify-end' : '');
    div.innerHTML = isUser ? 
        `<div class="bg-sky-600 text-white text-sm p-3 rounded-l-xl rounded-br-xl max-w-[80%] shadow-lg">${text}</div>` :
        `<div class="w-8 h-8 rounded-full bg-sky-900 flex items-center justify-center text-xs text-white flex-shrink-0">AI</div>
         <div class="bg-slate-800 text-slate-200 text-sm p-3 rounded-r-xl rounded-bl-xl max-w-[80%] border border-slate-700">${text}</div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    if (!chatMessages) return null;
    const div = document.createElement('div');
    div.id = 'typing-indicator';
    div.className = 'flex gap-3';
    div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-sky-900 flex items-center justify-center text-xs text-white flex-shrink-0">AI</div>
        <div class="bg-slate-800 p-4 rounded-r-xl rounded-bl-xl border border-slate-700 flex items-center gap-1.5 h-12 w-16 justify-center">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

if (chatBtn && chatInput) {
    chatBtn.addEventListener('click', () => {
        const text = chatInput.value.trim();
        if(!text) return;
        
        // 1. User Message
        addMessage(text, true);
        chatInput.value = '';

        // 2. Typing Indicator (simulating network delay)
        setTimeout(() => {
            const typingEl = showTypingIndicator();
            
            // 3. AI Response
            setTimeout(() => {
                if(typingEl) typingEl.remove();
                addMessage("This is a UI demo. No real AI backend is connected yet.", false);
            }, 1500); // 1.5s typing duration
        }, 500); // 0.5s delay before typing starts
    });
    chatInput.addEventListener('keypress', (e) => {
        if(e.key === 'Enter') chatBtn.click();
    });
}