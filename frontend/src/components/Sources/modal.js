window.sourcesModal = {
    isOpen: false,

    init() {
        this.modal = document.getElementById('sources-modal');
        this.list = document.getElementById('sources-list');
        this.closeBtn = this.modal.querySelector('.modal-close');
        this.backdrop = this.modal.querySelector('.modal-backdrop');

        document.getElementById('sources-btn').addEventListener('click', () => this.open());
        this.closeBtn.addEventListener('click', () => this.close());
        this.backdrop.addEventListener('click', () => this.close());

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) this.close();
        });
    },

    async open() {
        this.isOpen = true;
        this.modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';

        this.list.innerHTML = '<p style="text-align: center; color: var(--color-text-tertiary);">Loading sources...</p>';

        try {
            const data = await window.api.getSources();
            this.renderSources(data.sources);
        } catch (error) {
            this.list.innerHTML = `<p style="color: var(--color-danger);">Failed to load sources: ${error.message}</p>`;
        }

        this.closeBtn.focus();
    },

    close() {
        this.isOpen = false;
        this.modal.classList.add('hidden');
        document.body.style.overflow = '';
        document.getElementById('sources-btn').focus();
    },

    renderSources(sources) {
        if (!sources || sources.length === 0) {
            this.list.innerHTML = '<p style="text-align: center; color: var(--color-text-tertiary);">No sources available.</p>';
            return;
        }

        this.list.innerHTML = sources.map(source => `
            <div class="source-card">
                <h3>${helpers.sanitize(source.short_title || source.title || 'Untitled')}</h3>
                <p><strong>Organization:</strong> ${helpers.sanitize(source.organization || 'Unknown')}</p>
                <p><strong>Type:</strong> ${helpers.sanitize(source.document_type || 'Document').replace(/_/g, ' ')}</p>
                ${source.year ? `<p><strong>Year:</strong> ${source.year}</p>` : ''}
                ${source.pages ? `<p><strong>Pages:</strong> ${helpers.sanitize(source.pages)}</p>` : ''}
                ${source.indexed_chunks ? `<p><strong>Indexed chunks:</strong> ${source.indexed_chunks}</p>` : ''}
                ${source.doi ? `<p class="doi">DOI: ${helpers.sanitize(source.doi)}</p>` : ''}
                ${source.official_url ? `<p><a href="${helpers.sanitize(source.official_url)}" target="_blank" rel="noopener noreferrer">View Source ↗</a></p>` : ''}
            </div>
        `).join('');
    }
};
