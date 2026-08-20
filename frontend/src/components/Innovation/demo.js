window.demoModeComponent = {
    isActive: false,
    currentIndex: 0,
    sequenceTimer: null,
    demoQuestions: [
        {
            question: 'What is the fasting glucose cutoff for diagnosing diabetes?',
            label: 'Diagnostic Threshold',
            description: 'Demonstrates evidence-grounded clinical threshold lookup with full citations'
        },
        {
            question: 'How is prediabetes diagnosed?',
            label: 'Classification Criteria',
            description: 'Shows multi-source evidence aggregation from ADA and NIDDK'
        },
        {
            question: 'What is the recommended dosage of insulin for type 2 diabetes?',
            label: 'Safety Boundary',
            description: 'Demonstrates clinical safety refusal with helpful redirect'
        }
    ],

    init() {
        const btn = document.getElementById('demo-mode-btn');
        if (btn) {
            btn.addEventListener('click', () => this.toggle());
        }
    },

    toggle() {
        if (this.isActive) {
            this.stop();
        } else {
            this.start();
        }
    },

    async start() {
        this.isActive = true;
        this.currentIndex = 0;
        const btn = document.getElementById('demo-mode-btn');
        if (btn) {
            btn.classList.add('active');
            btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:inline;vertical-align:-1px;margin-right:3px"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>Stop Demo';
            btn.title = 'Stop demo sequence';
        }

        this.showDemoBanner();
        await this.runNextQuestion();
    },

    stop() {
        this.isActive = false;
        if (this.sequenceTimer) {
            clearTimeout(this.sequenceTimer);
            this.sequenceTimer = null;
        }
        const btn = document.getElementById('demo-mode-btn');
        if (btn) {
            btn.classList.remove('active');
            btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:inline;vertical-align:-1px;margin-right:3px"><polygon points="5 3 19 12 5 21 5 3"/></svg>Demo';
            btn.title = 'Demo mode for judges';
        }
        this.hideDemoBanner();
    },

    async runNextQuestion() {
        if (!this.isActive) return;
        if (this.currentIndex >= this.demoQuestions.length) {
            this.sequenceTimer = setTimeout(() => this.stop(), 2000);
            return;
        }

        const item = this.demoQuestions[this.currentIndex];
        this.updateDemoBanner(this.currentIndex);

        while (window.app && window.app.isLoading) {
            await this.delay(300);
        }

        if (!this.isActive) return;

        if (window.app) {
            window.app.handleUserMessage(item.question);
        }

        const maxWait = 30000;
        let waited = 0;
        while (window.app && window.app.isLoading && waited < maxWait) {
            await this.delay(500);
            waited += 500;
        }

        await this.delay(1500);

        this.currentIndex++;
        if (this.isActive && this.currentIndex < this.demoQuestions.length) {
            this.sequenceTimer = setTimeout(() => this.runNextQuestion(), 800);
        } else if (this.isActive) {
            this.sequenceTimer = setTimeout(() => this.stop(), 2000);
        }
    },

    showDemoBanner() {
        let banner = document.getElementById('demo-banner');
        if (!banner) {
            banner = document.createElement('div');
            banner.id = 'demo-banner';
            banner.className = 'demo-banner';
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.insertBefore(banner, chatContainer.firstChild);
            }
        }
        this.updateDemoBanner(0);
    },

    updateDemoBanner(index) {
        const banner = document.getElementById('demo-banner');
        if (!banner) return;

        const total = this.demoQuestions.length;
        const item = this.demoQuestions[index];

        let dots = '';
        for (let i = 0; i < total; i++) {
            const cls = i < index ? 'done' : i === index ? 'active' : '';
            dots += `<span class="demo-dot ${cls}"></span>`;
        }

        banner.innerHTML = `
            <div class="demo-banner-content">
                <div class="demo-banner-badge">DEMO MODE</div>
                <div class="demo-banner-info">
                    <span class="demo-banner-step">Question ${index + 1} of ${total}</span>
                    <span class="demo-banner-label">${item.label}</span>
                </div>
                <div class="demo-banner-dots">${dots}</div>
                <div class="demo-banner-desc">${item.description}</div>
            </div>
        `;
    },

    hideDemoBanner() {
        const banner = document.getElementById('demo-banner');
        if (banner) banner.remove();
    },

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
};
