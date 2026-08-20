window.inputComponent = {
    init() {
        this.input = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-btn');
        this.setupEventListeners();
    },

    setupEventListeners() {
        this.sendBtn.addEventListener('click', () => {
            this.handleSubmit();
        });

        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSubmit();
            }
        });

        this.input.addEventListener('input', () => {
            this.autoResize();
        });
    },

    autoResize() {
        this.input.style.height = 'auto';
        this.input.style.height = Math.min(this.input.scrollHeight, 160) + 'px';
    },

    handleSubmit() {
        const text = this.input.value.trim();
        if (!text) return;

        this.input.value = '';
        this.autoResize();

        if (window.app && window.app.handleUserMessage) {
            window.app.handleUserMessage(text).catch(err => {
                console.error("[CHAT] handleUserMessage error:", err);
                window.app.isLoading = false;
                window.app.appendMessage(window.app.createErrorMessage(err));
                inputComponent.setDisabled(false);
                inputComponent.focus();
            });
        }
    },

    setDisabled(disabled) {
        this.input.disabled = disabled;
        this.sendBtn.disabled = disabled;
    },

    focus() {
        this.input.focus();
    }
};
