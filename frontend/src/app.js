class DiabetesRagApp {
    constructor() {
        this.isLoading = false;
        this.currentLoadingId = null;
        this.loadingTimer = null;
        this.lastResponses = [];
        this.lastResponseForDrawer = null;
        this.currentResponseIndex = -1;
        this.conversationCount = 0;
        this.lastQuestionText = '';
        this.lastQuestionIntelligence = null;
        this.currentChatId = null;
        this.chatMessages = [];
    }

    init() {
        window.chatStorage.init();
        inputComponent.init();
        sourcesModal.init();
        howItWorksComponent.init();
        drawerComponent.init();
        if (window.demoModeComponent) demoModeComponent.init();
        this.setupEventListeners();
        this.checkHealth();
        sidebarSourcesComponent.load();
        this.renderRecentChats();
        this.startNewChat();
    }

    setupEventListeners() {
        document.getElementById('clear-btn').addEventListener('click', () => this.clearConversation());
        document.getElementById('sidebar-toggle').addEventListener('click', () => this.toggleSidebar());
        document.getElementById('new-chat-btn').addEventListener('click', () => this.startNewChat());
        document.getElementById('clear-all-chats-btn').addEventListener('click', () => this.clearAllChats());

        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.dataset.question;
                if (question) {
                    inputComponent.input.value = question;
                    inputComponent.handleSubmit();
                }
            });
        });

        document.getElementById('chat-container').addEventListener('click', (e) => {
            const citationRef = e.target.closest('.citation-ref');
            if (citationRef) {
                const index = parseInt(citationRef.dataset.evidence);
                if (!isNaN(index)) {
                    citationComponent.handleClick(index);
                }
            }
        });

        document.getElementById('recent-chats').addEventListener('click', (e) => {
            const chatItem = e.target.closest('.recent-chat-item');
            const deleteBtn = e.target.closest('.recent-chat-delete');
            const renameBtn = e.target.closest('.recent-chat-rename');

            if (deleteBtn) {
                e.stopPropagation();
                const id = deleteBtn.dataset.id;
                window.chatStorage.deleteConversation(id);
                if (this.currentChatId === id) this.startNewChat();
                this.renderRecentChats();
                return;
            }
            if (renameBtn) {
                e.stopPropagation();
                const id = renameBtn.dataset.id;
                const conv = window.chatStorage.getConversation(id);
                if (conv) {
                    const newTitle = prompt('Rename conversation:', conv.title);
                    if (newTitle && newTitle.trim()) {
                        window.chatStorage.renameConversation(id, newTitle.trim());
                        this.renderRecentChats();
                    }
                }
                return;
            }
            if (chatItem) {
                const id = chatItem.dataset.id;
                this.loadConversation(id);
            }
        });

        const mobileOverlay = document.getElementById('mobile-overlay');
        if (mobileOverlay) {
            mobileOverlay.addEventListener('click', () => {
                document.querySelector('.app-layout').classList.remove('sidebar-open');
                mobileOverlay.classList.add('hidden');
            });
        }
    }

    toggleSidebar() {
        const layout = document.querySelector('.app-layout');
        const isMobile = window.innerWidth <= 1024;
        if (isMobile) {
            layout.classList.toggle('sidebar-open');
            const overlay = document.getElementById('mobile-overlay');
            if (layout.classList.contains('sidebar-open')) {
                overlay.classList.remove('hidden');
            } else {
                overlay.classList.add('hidden');
            }
        } else {
            layout.classList.toggle('sidebar-collapsed');
        }
    }

    startNewChat() {
        this.currentChatId = window.chatStorage.generateId();
        this.chatMessages = [];
        this.lastResponses = [];
        this.lastResponseForDrawer = null;
        this.currentResponseIndex = -1;
        this.conversationCount = 0;
        this.lastQuestionText = '';
        this.lastQuestionIntelligence = null;
        const messagesEl = document.getElementById('messages');
        messagesEl.innerHTML = '';
        this.showWelcome();
        this.renderRecentChats();
        drawerComponent.close();
        pipelineComponent.reset();
    }

    loadConversation(id) {
        const conv = window.chatStorage.getConversation(id);
        if (!conv) return;

        this.currentChatId = id;
        this.chatMessages = conv.messages;
        this.lastResponses = [];
        this.lastResponseForDrawer = null;
        this.currentResponseIndex = -1;
        this.conversationCount = 0;
        this.lastQuestionText = '';
        this.lastQuestionIntelligence = null;

        const messagesEl = document.getElementById('messages');
        messagesEl.innerHTML = '';
        this.hideWelcome();

        for (const msg of conv.messages) {
            if (msg.role === 'user') {
                this.appendMessage(messageComponent.createUserMessage(msg.text));
                this.conversationCount++;
            } else if (msg.role === 'assistant' && msg.data) {
                this.lastResponses.push(msg.data);
                this.currentResponseIndex = this.lastResponses.length - 1;
                this.lastResponseForDrawer = msg.data;
                this.appendMessage(messageComponent.createAssistantMessage(msg.data));
            }
        }

        this.renderRecentChats();
        this.scrollToBottom();

        const isMobile = window.innerWidth <= 1024;
        if (isMobile) {
            document.querySelector('.app-layout').classList.remove('sidebar-open');
            document.getElementById('mobile-overlay').classList.add('hidden');
        }
    }

    renderRecentChats() {
        const container = document.getElementById('recent-chats');
        const conversations = window.chatStorage.getConversations();

        if (conversations.length === 0) {
            container.innerHTML = '<div class="sidebar-loading">No conversations yet</div>';
            return;
        }

        const now = Date.now();
        const DAY = 86400000;
        const groups = { Today: [], Yesterday: [], Earlier: [] };

        for (const conv of conversations) {
            const age = now - conv.updatedAt;
            if (age < DAY) groups.Today.push(conv);
            else if (age < DAY * 2) groups.Yesterday.push(conv);
            else groups.Earlier.push(conv);
        }

        let html = '';
        for (const [label, items] of Object.entries(groups)) {
            if (items.length === 0) continue;
            html += `<div class="recent-chat-group-label">${label}</div>`;
            for (const conv of items) {
                const isActive = conv.id === this.currentChatId;
                const time = new Date(conv.updatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                html += `
                    <div class="recent-chat-item${isActive ? ' active' : ''}" data-id="${conv.id}" role="button" tabindex="0">
                        <div class="recent-chat-title">${this.escapeHtml(conv.title)}</div>
                        <div class="recent-chat-actions">
                            <button class="recent-chat-rename" data-id="${conv.id}" title="Rename" aria-label="Rename">
                                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                            </button>
                            <button class="recent-chat-delete" data-id="${conv.id}" title="Delete" aria-label="Delete">
                                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                            </button>
                        </div>
                    </div>`;
            }
        }
        container.innerHTML = html;
    }

    clearAllChats() {
        if (!confirm('Clear all conversation history? This cannot be undone.')) return;
        window.chatStorage.clearAll();
        this.startNewChat();
    }

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    detectQuestionIntelligence(text) {
        const lower = text.toLowerCase();
        if (/dosage|dose|medication|insulin|drug|treat|prescribe|how much|how often/i.test(lower)) {
            return { type: 'medication', label: 'Medication Question', icon: '\u{1F48A}' };
        }
        if (/diagnos|criteria|cutoff|threshold|classify|type\s*[12]/i.test(lower)) {
            return { type: 'diagnostic', label: 'Diagnostic Criteria', icon: '\u{1F52C}' };
        }
        if (/a1c|fpg|ogtt|hba1c|fasting|glucose test|tolerance/i.test(lower)) {
            return { type: 'lab', label: 'Lab Test', icon: '\u{1F9EA}' };
        }
        if (/difference|compar|vs|versus|better|worse/i.test(lower)) {
            return { type: 'comparison', label: 'Comparison', icon: '\u2696\uFE0F' };
        }
        if (/gestational|pregnan|pregnancy/i.test(lower)) {
            return { type: 'gestational', label: 'Gestational', icon: '\u{1F930}' };
        }
        if (/prediabetes|impaired|borderline/i.test(lower)) {
            return { type: 'prediabetes', label: 'Prediabetes', icon: '\u{1F4CB}' };
        }
        return { type: 'general', label: 'Clinical Question', icon: '\u2753' };
    }

    async checkHealth() {
        const statusEl = document.getElementById('engine-status');
        try {
            const health = await window.api.healthCheck();
            statusEl.className = 'engine-status online';
            const providers = health.providers || [];
            const mode = health.llm_provider_mode || 'auto';
            if (providers.length > 0) {
                const chainLabel = providers.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' → ');
                statusEl.querySelector('.status-text').textContent = 'Evidence Engine \u00B7 Hybrid Retrieval';
                statusEl.title = `LLM chain: ${chainLabel} (mode: ${mode})`;
            } else {
                statusEl.querySelector('.status-text').textContent = 'Evidence Engine \u00B7 No LLM';
                statusEl.title = 'No LLM provider configured';
            }
        } catch (error) {
            statusEl.className = 'engine-status offline';
            statusEl.querySelector('.status-text').textContent = 'Engine Offline';
        }
    }

    async handleUserMessage(text) {
        if (this.isLoading) return;

        this.hideWelcome();
        this.appendMessage(messageComponent.createUserMessage(text));

        this.chatMessages.push({ role: 'user', text: text });

        this.isLoading = true;
        inputComponent.setDisabled(true);

        this.lastQuestionText = text;
        this.lastQuestionIntelligence = this.detectQuestionIntelligence(text);

        let loading = null;
        let stageInterval = null;

        try {
            loading = loadingComponent.create();
            this.appendMessage(loading.html);
            this.currentLoadingId = loading.id;

            pipelineComponent.reset();

            let elapsed = 0;
            this.loadingTimer = setInterval(() => {
                elapsed += 100;
                loadingComponent.updateTime(loading.id, elapsed);
            }, 100);

            const stageSequence = ['intent', 'retrieve', 'evidence', 'conflict', 'generate', 'citation', 'verify', 'safety'];
            let stageIndex = 0;

            const advanceStage = () => {
                if (stageIndex < stageSequence.length) {
                    loadingComponent.updateStage(loading.id, stageSequence[stageIndex]);
                    stageIndex++;
                }
            };

            advanceStage();
            stageInterval = setInterval(() => advanceStage(), 800);

            const response = await window.api.chat(text);

            clearInterval(stageInterval);
            loadingComponent.updateStage(loading.id, 'answer');
            await this.delay(150);

            loadingComponent.remove(loading.id);
            pipelineComponent.resetAfterResponse();

            this.conversationCount++;
            this.currentResponseIndex = this.lastResponses.length;
            this.lastResponses.push(response);
            this.lastResponseForDrawer = response;

            this.chatMessages.push({ role: 'assistant', text: response.answer, data: response });

            this.appendMessage(messageComponent.createAssistantMessage(response));

            window.chatStorage.saveConversation(this.currentChatId, this.chatMessages);
            this.renderRecentChats();

        } catch (error) {
            console.error("[CHAT] error:", error.message || error);
            if (stageInterval) clearInterval(stageInterval);
            if (loading) loadingComponent.remove(loading.id);
            pipelineComponent.reset();
            this.appendMessage(this.createErrorMessage(error));
        } finally {
            clearInterval(this.loadingTimer);
            if (stageInterval) clearInterval(stageInterval);
            this.isLoading = false;
            this.currentLoadingId = null;
            inputComponent.setDisabled(false);
            inputComponent.focus();
        }
    }

    createErrorMessage(error) {
        let title = 'Error';
        let details = error.userMessage || error.message || 'An unexpected error occurred.';
        let showRetry = true;

        if (error.status === 429) {
            title = 'Service Temporarily Unavailable';
            details = 'The AI generation service is rate-limited. You can still explore the system using <strong>Demo Mode</strong> to see how it works, or try again shortly.';
        } else if (error.status === 500) {
            title = 'Server Error';
            details = 'The evidence engine encountered an error. Please try again.';
        } else if (error.name === 'AbortError') {
            title = 'Request Timed Out';
            details = 'The request took too long. The evidence engine may be loading. Please try again.';
        } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
            title = 'Connection Error';
            details = 'Unable to connect to the evidence engine. Please ensure the backend is running on port 8000.';
        }

        return `
            <div class="message error" role="alert">
                <div class="message-avatar" aria-hidden="true">!</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-role">System</span>
                        <span class="message-time">${helpers.formatTimestamp()}</span>
                    </div>
                    <div class="message-body">
                        <div class="error-message">
                            <div class="error-title">${title}</div>
                            <div class="error-details">${details}</div>
                            <div>
                                ${showRetry ? '<button class="btn-retry" onclick="app.retryLastMessage()" style="margin-right:8px">Try Again</button>' : ''}
                                ${error.status === 429 ? '<button class="btn-retry" onclick="demoModeComponent.start()" style="background:var(--color-primary)">Run Demo</button>' : ''}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    appendMessage(html) {
        const messagesEl = document.getElementById('messages');
        messagesEl.insertAdjacentHTML('beforeend', html);
        this.scrollToBottom();
    }

    scrollToBottom() {
        const container = document.getElementById('chat-container');
        requestAnimationFrame(() => {
            container.scrollTop = container.scrollHeight;
        });
    }

    hideWelcome() {
        const welcome = document.getElementById('welcome-screen');
        if (welcome) welcome.style.display = 'none';
    }

    showWelcome() {
        const welcome = document.getElementById('welcome-screen');
        if (welcome) welcome.style.display = '';
    }

    clearConversation() {
        this.startNewChat();
    }

    retryLastMessage() {
        const messagesEl = document.getElementById('messages');
        const lastUserMsg = messagesEl.querySelectorAll('.message.user');
        if (lastUserMsg.length > 0) {
            const lastMsg = lastUserMsg[lastUserMsg.length - 1];
            const text = lastMsg.querySelector('.message-body').textContent.trim();
            if (text) this.handleUserMessage(text);
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

window.app = new DiabetesRagApp();
document.addEventListener('DOMContentLoaded', () => {
    window.app.init();
});
