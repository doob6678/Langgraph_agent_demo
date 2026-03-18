
/**
 * Memory UI Manager
 * Handles the display and interaction of the Agent Memory Panel.
 */

const memoryPanel = document.getElementById('memory-panel');
const memoryContentShortTerm = document.getElementById('memory-content-short-term');
const memoryContentLongTerm = document.getElementById('memory-content-long-term');
const memoryContentImages = document.getElementById('memory-content-images');
const memoryUiState = {
    loadingImages: false,
    loadingFacts: false,
    loadedImagesIdentity: '',
    loadedFactsIdentity: '',
    factEvents: []
};

function renderImagesGrid(images) {
    if (!memoryContentImages) return;
    if (Array.isArray(images) && images.length > 0) {
        memoryContentImages.innerHTML = images.map(img => {
            let uri = img.uri || img.image_uri || '';
            if (uri.startsWith('oss://')) {
                uri = uri.replace('oss://', '/assets/');
            }
            if (uri && !uri.startsWith('http://') && !uri.startsWith('https://') && !uri.startsWith('/assets/')) {
                uri = `/assets/${uri.replace(/^\/+/, '')}`;
            }
            const createdAt = String(img.created_at || '');
            const visibility = String(img.visibility || 'private');
            const owner = `${String(img.dept_id || '-')}/${String(img.user_id || '-')}`;
            return `
                <div class="relative group cursor-pointer border rounded overflow-hidden shadow-sm hover:shadow-md transition-shadow bg-gray-100 h-40">
                    <img src="${uri}" class="w-full h-full object-cover" alt="${escapeHtml(img.description || '')}" 
                         onerror="this.src='https://placehold.co/100x100?text=Error';this.onerror=null;">
                    <div class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-60 transition-all flex items-center justify-center p-2">
                        <span class="text-white opacity-0 group-hover:opacity-100 text-xs text-center line-clamp-3 select-none pointer-events-none">
                            ${escapeHtml(img.description || img.content || 'Image')}
                        </span>
                    </div>
                    <div class="absolute left-0 right-0 bottom-0 bg-black bg-opacity-60 text-white text-[10px] px-2 py-1 leading-4">
                        <div>${escapeHtml(owner)} · ${escapeHtml(visibility)}</div>
                        <div>${escapeHtml(createdAt)}</div>
                    </div>
                </div>
            `;
        }).join('');
        if (!memoryContentImages.classList.contains('hidden')) {
            memoryContentImages.classList.add('grid');
        }
        return;
    }
    memoryContentImages.innerHTML = '<div class="text-sm text-gray-400 italic col-span-2 text-center mt-4">No images found.</div>';
}

async function loadPersonalImages(force = false) {
    if (!memoryContentImages || memoryUiState.loadingImages) return;
    const userId = (document.getElementById('user-id')?.value || '').trim() || 'default_user';
    const deptId = (document.getElementById('dept-id')?.value || '').trim() || 'default_dept';
    const identity = `${deptId}::${userId}`;
    if (!force && memoryUiState.loadedImagesIdentity === identity) return;
    memoryUiState.loadingImages = true;
    memoryContentImages.innerHTML = '<div class="text-sm text-gray-400 italic col-span-2 text-center mt-4">Loading images...</div>';
    try {
        const params = new URLSearchParams({
            user_id: userId,
            dept_id: deptId,
            limit: '100'
        });
        const response = await fetch(`/api/memory/images/query?${params.toString()}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        const items = Array.isArray(payload?.items) ? payload.items : [];
        renderImagesGrid(items);
        memoryUiState.loadedImagesIdentity = identity;
    } catch (e) {
        memoryContentImages.innerHTML = `<div class="text-sm text-red-500 italic col-span-2 text-center mt-4">Load images failed: ${escapeHtml(String(e?.message || e || 'unknown error'))}</div>`;
    } finally {
        memoryUiState.loadingImages = false;
    }
}

function formatUnixTime(ts) {
    const num = Number(ts || 0);
    if (!num) return '';
    const ms = num > 1000000000000 ? num : num * 1000;
    const d = new Date(ms);
    if (Number.isNaN(d.getTime())) return '';
    return d.toLocaleString('zh-CN');
}

function renderFactsList(facts = [], events = []) {
    if (!memoryContentLongTerm) return;
    const safeFacts = (Array.isArray(facts) ? facts : []).filter(item => {
        const t = String(item?.metadata?.type || 'fact').trim().toLowerCase();
        return t !== 'image_summary';
    });
    const safeEvents = (Array.isArray(events) ? events : []).filter(evt => {
        const eventType = String(evt?.type || '').trim().toLowerCase();
        const memoryType = String(evt?.memory_type || 'fact').trim().toLowerCase();
        if (eventType && eventType !== 'long_term_saved') return false;
        return memoryType !== 'image_summary';
    });
    if (!safeFacts.length && !safeEvents.length) {
        memoryContentLongTerm.innerHTML = '<div class="text-sm text-gray-400 italic text-center mt-4">No long-term facts found.</div>';
        return;
    }
    const eventCards = safeEvents.map(evt => {
        const timeText = formatUnixTime(evt.created_at || evt.timestamp || 0);
        return `
            <div class="p-2 rounded bg-green-50 text-xs mb-2 border border-green-100">
                <div class="text-green-700 font-semibold">Saved</div>
                <div class="text-gray-800 mt-1">${escapeHtml(evt.fact || evt.content || '')}</div>
                <div class="text-gray-400 text-[10px] mt-1">${escapeHtml(timeText || '')}</div>
            </div>
        `;
    }).join('');
    const factCards = safeFacts.map(item => `
        <div class="p-2 rounded bg-yellow-50 text-xs mb-2 border border-yellow-100">
            <div class="text-gray-800">${escapeHtml(item.content || '')}</div>
            <div class="text-gray-400 text-[10px] mt-1">
                ${escapeHtml(String(item.visibility || 'private'))}
                ${item.created_at ? ` · ${escapeHtml(formatUnixTime(item.created_at))}` : ''}
            </div>
        </div>
    `).join('');
    memoryContentLongTerm.innerHTML = eventCards + factCards;
}

async function loadLongTermFacts(force = false) {
    if (!memoryContentLongTerm || memoryUiState.loadingFacts) return;
    const userId = (document.getElementById('user-id')?.value || '').trim() || 'default_user';
    const deptId = (document.getElementById('dept-id')?.value || '').trim() || 'default_dept';
    const identity = `${deptId}::${userId}`;
    if (!force && memoryUiState.loadedFactsIdentity === identity) return;
    memoryUiState.loadingFacts = true;
    memoryContentLongTerm.innerHTML = '<div class="text-sm text-gray-400 italic text-center mt-4">Loading facts...</div>';
    try {
        const params = new URLSearchParams({
            user_id: userId,
            dept_id: deptId,
            limit: '100',
            include_image_summary: 'false'
        });
        const response = await fetch(`/api/memory/facts?${params.toString()}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        const items = Array.isArray(payload?.items) ? payload.items : [];
        renderFactsList(items, memoryUiState.factEvents);
        memoryUiState.loadedFactsIdentity = identity;
    } catch (e) {
        memoryContentLongTerm.innerHTML = `<div class="text-sm text-red-500 italic text-center mt-4">Load facts failed: ${escapeHtml(String(e?.message || e || 'unknown error'))}</div>`;
    } finally {
        memoryUiState.loadingFacts = false;
    }
}

/**
 * Toggle the visibility of the memory panel
 */
function toggleMemoryPanel() {
    if (memoryPanel) {
        memoryPanel.classList.toggle('translate-x-full');
        memoryPanel.classList.toggle('translate-x-0');
        const imagesVisible = memoryContentImages && !memoryContentImages.classList.contains('hidden');
        const factsVisible = memoryContentLongTerm && !memoryContentLongTerm.classList.contains('hidden');
        if (imagesVisible) {
            loadPersonalImages();
        }
        if (factsVisible) {
            loadLongTermFacts();
        }
    }
}

/**
 * Switch between memory tabs
 * @param {string} tabName - 'short-term', 'long-term', or 'images'
 * @param {HTMLElement} btn - The clicked button element
 */
function switchTab(tabName, btn) {
    // Hide all contents
    const contents = ['short-term', 'long-term', 'images'];
    contents.forEach(name => {
        const el = document.getElementById(`memory-content-${name}`);
        if (el) {
            el.classList.add('hidden');
            el.classList.remove('grid'); // Remove grid class just in case
        }
    });
    
    // Show selected
    const selectedEl = document.getElementById(`memory-content-${tabName}`);
    if (selectedEl) {
        selectedEl.classList.remove('hidden');
        if (tabName === 'images') {
            selectedEl.classList.add('grid');
            loadPersonalImages();
        } else if (tabName === 'long-term') {
            loadLongTermFacts();
        }
    }

    // Update buttons
    if (btn && btn.parentElement) {
        const buttons = btn.parentElement.querySelectorAll('button');
        buttons.forEach(b => {
            b.className = 'flex-1 py-2 text-sm font-medium text-gray-500 hover:text-blue-600 border-b-2 border-transparent';
        });
        btn.className = 'flex-1 py-2 text-sm font-medium text-blue-600 border-b-2 border-blue-600';
    }
}

/**
 * Render memory data received from the backend
 * @param {Object} data - The x_memory_event data object
 */
function renderMemory(data) {
    console.log("Rendering memory data:", data);
    
    // data structure from backend: { status, data: { ... }, context: "..." }
    const memData = data.data || {};
    const contextStr = data.context || "";
    
    // 1. Render Short-term (Context)
    if (memoryContentShortTerm) {
        if (contextStr) {
            memoryContentShortTerm.innerHTML = `<div class="text-sm text-gray-700 whitespace-pre-wrap font-mono bg-white p-2 rounded border border-gray-100">${escapeHtml(contextStr)}</div>`;
        } else if (memData.short_term && Array.isArray(memData.short_term) && memData.short_term.length > 0) {
            // Fallback to structured data if context string is not available or we prefer it
            memoryContentShortTerm.innerHTML = memData.short_term.map(item => `
                <div class="p-2 rounded bg-gray-50 text-xs mb-2 border border-gray-200">
                    <div class="font-bold text-gray-500 mb-1">${escapeHtml(item.role || 'user')}:</div>
                    <div class="text-gray-800">${escapeHtml(item.content)}</div>
                </div>
            `).join('');
        } else {
            memoryContentShortTerm.innerHTML = '<div class="text-sm text-gray-400 italic text-center mt-4">No recent context.</div>';
        }
    }

    // 2. Render Long-term (Facts)
    // Currently backend returns formatted context string, but if we had structured facts:
    if (memoryContentLongTerm) {
        if (memData.events && Array.isArray(memData.events) && memData.events.length > 0) {
            const incomingFactEvents = memData.events.filter(evt => {
                const eventType = String(evt?.type || '').trim().toLowerCase();
                const memoryType = String(evt?.memory_type || 'fact').trim().toLowerCase();
                if (eventType && eventType !== 'long_term_saved') return false;
                return memoryType !== 'image_summary';
            });
            memoryUiState.factEvents = incomingFactEvents.concat(memoryUiState.factEvents).slice(0, 30);
        }
        if (memData.long_term && Array.isArray(memData.long_term)) {
            renderFactsList(memData.long_term, memoryUiState.factEvents);
        } else {
            loadLongTermFacts();
        }
    }

    // 3. Render Images
    if (memoryContentImages) {
        if (Array.isArray(memData.images) && memData.images.length > 0) {
            renderImagesGrid(memData.images);
        }
    }
}

// Simple HTML escaper
function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

const userIdInput = document.getElementById('user-id');
const deptIdInput = document.getElementById('dept-id');
const onIdentityChanged = () => {
    memoryUiState.loadedImagesIdentity = '';
    memoryUiState.loadedFactsIdentity = '';
    memoryUiState.factEvents = [];
    const imagesVisible = memoryContentImages && !memoryContentImages.classList.contains('hidden');
    const factsVisible = memoryContentLongTerm && !memoryContentLongTerm.classList.contains('hidden');
    if (imagesVisible) {
        loadPersonalImages(true);
    }
    if (factsVisible) {
        loadLongTermFacts(true);
    }
};
if (userIdInput) userIdInput.addEventListener('change', onIdentityChanged);
if (deptIdInput) deptIdInput.addEventListener('change', onIdentityChanged);
