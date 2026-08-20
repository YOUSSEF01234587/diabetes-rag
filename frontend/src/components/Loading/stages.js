window.loadingComponent = {
    stages: [
        { id: 'intent', label: 'Understanding question' },
        { id: 'retrieve', label: 'Finding evidence' },
        { id: 'evidence', label: 'Selecting evidence' },
        { id: 'conflict', label: 'Checking clinical context' },
        { id: 'generate', label: 'Generating grounded answer' },
        { id: 'citation', label: 'Validating citations' },
        { id: 'verify', label: 'Verifying answer' },
        { id: 'safety', label: 'Checking safety' },
        { id: 'answer', label: 'Preparing answer' }
    ],

    create() {
        const loadingId = `loading-${Date.now()}`;

        const progressBarHtml = `
            <div class="loading-progress">
                <div class="loading-progress-bar" id="${loadingId}-progress-bar"></div>
            </div>
        `;

        let html = `
            <div class="message assistant loading-message" id="${loadingId}" role="status" aria-label="Loading response" aria-live="polite">
                <div class="message-avatar" aria-hidden="true">AI</div>
                <div class="message-content">
                    <div class="loading-stages">
                        ${this.stages.map((stage, index) => `
                            <div class="loading-stage ${index === 0 ? 'active' : ''}" id="${loadingId}-${stage.id}">
                                <span class="loading-stage-icon">
                                    ${index === 0 ? '<div class="spinner"></div>' : ''}
                                </span>
                                <span>${stage.label}</span>
                            </div>
                        `).join('')}
                        ${progressBarHtml}
                        <div class="loading-time" id="${loadingId}-time">Starting...</div>
                    </div>
                </div>
            </div>
        `;

        return { html, id: loadingId };
    },

    updateStage(id, completedStageId) {
        const completedIndex = this.stages.findIndex(s => s.id === completedStageId);

        this.stages.forEach((stage, index) => {
            const el = document.getElementById(`${id}-${stage.id}`);
            if (!el) return;

            const iconEl = el.querySelector('.loading-stage-icon');

            if (index <= completedIndex) {
                el.className = 'loading-stage completed';
                iconEl.innerHTML = '<span class="check">✓</span>';
            } else if (index === completedIndex + 1) {
                el.className = 'loading-stage active';
                iconEl.innerHTML = '<div class="spinner"></div>';
            } else {
                el.className = 'loading-stage';
                iconEl.innerHTML = '';
            }
        });

        const pct = Math.min(((completedIndex + 1) / this.stages.length) * 100, 100);
        const bar = document.getElementById(`${id}-progress-bar`);
        if (bar) bar.style.width = `${pct}%`;

        if (window.pipelineComponent) {
            window.pipelineComponent.animateToStage(completedStageId);
        }
    },

    updateTime(id, elapsed) {
        const timeEl = document.getElementById(`${id}-time`);
        if (timeEl) {
            timeEl.textContent = `Elapsed: ${helpers.formatTime(elapsed)}`;
        }
    },

    remove(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
};
