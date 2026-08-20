window.drawerComponent = {
    isOpen: false,

    init() {
        this.drawer = document.getElementById('evidence-drawer');
        this.body = document.getElementById('drawer-body');
        this.closeBtn = document.getElementById('drawer-close');
        this.overlay = document.getElementById('drawer-overlay');
        this.layout = document.querySelector('.app-layout');

        this.closeBtn.addEventListener('click', () => this.close());
        if (this.overlay) {
            this.overlay.addEventListener('click', () => this.close());
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) this.close();
        });

        this.updateDesktopLayout();
        window.addEventListener('resize', () => this.updateDesktopLayout());
    },

    isMobile() {
        return window.innerWidth <= 1024;
    },

    updateDesktopLayout() {
        if (this.isMobile()) {
            this.layout.classList.remove('drawer-open', 'drawer-closed');
        } else if (this.isOpen) {
            this.layout.classList.add('drawer-open');
            this.layout.classList.remove('drawer-closed');
        } else {
            this.layout.classList.remove('drawer-open');
            this.layout.classList.add('drawer-closed');
        }
    },

    open() {
        this.isOpen = true;
        this.drawer.classList.add('open');
        if (!this.isMobile() && this.overlay) {
            this.overlay.classList.add('hidden');
        } else if (this.overlay) {
            this.overlay.classList.remove('hidden');
        }
        this.updateDesktopLayout();
    },

    close() {
        this.isOpen = false;
        this.drawer.classList.remove('open');
        if (this.overlay) this.overlay.classList.add('hidden');
        this.updateDesktopLayout();
    },

    openForEvidence(evidenceIndex) {
        const response = window.app ? window.app.lastResponseForDrawer : null;
        if (!response) {
            this.open();
            this.body.innerHTML = '<p class="drawer-empty">Evidence details will appear here when you click a citation.</p>';
            return;
        }

        const evidenceItem = response.evidence ? response.evidence.find(e => {
            const sources = response.sources || [];
            const source = sources.find(s => s.source_id === e.source_id);
            return source && source.index === evidenceIndex;
        }) : null;

        const sourceItem = response.sources ? response.sources.find(s => s.index === evidenceIndex) : null;

        if (!evidenceItem && !sourceItem) {
            this.open();
            this.body.innerHTML = `<p class="drawer-empty">Evidence ${evidenceIndex} details not available in the response.</p>`;
            return;
        }

        const org = (sourceItem && sourceItem.organization) || (evidenceItem && evidenceItem.organization) || 'Unknown';
        const section = (sourceItem && sourceItem.section) || (evidenceItem && evidenceItem.section) || '';
        const page = (sourceItem && sourceItem.page) || (evidenceItem && evidenceItem.page) || '';
        const label = (sourceItem && sourceItem.source_label) || org;
        const doi = sourceItem ? sourceItem.doi : '';
        const text = evidenceItem ? (evidenceItem.text_preview || '') : '';

        let html = `
            <div class="drawer-evidence-card">
                <div class="drawer-evidence-header">
                    <h4>Evidence ${evidenceIndex}</h4>
                    <div class="drawer-meta">${helpers.sanitize(label)}</div>
                </div>
                <div class="drawer-evidence-body">
                    <ul class="drawer-meta-list">
                        <li><strong>Source:</strong> ${helpers.sanitize(org)}</li>
                        ${section ? `<li><strong>Section:</strong> ${helpers.sanitize(section)}</li>` : ''}
                        ${page ? `<li><strong>Page:</strong> ${page}</li>` : ''}
                        ${doi ? `<li><strong>DOI:</strong> ${helpers.sanitize(doi)}</li>` : ''}
                    </ul>
                    ${text ? `<div style="margin-top:12px;padding:12px;background:var(--color-gray-50);border-radius:var(--radius-md);font-size:13px;line-height:1.6;color:var(--color-text-secondary);">${helpers.sanitize(text)}</div>` : ''}
                </div>
            </div>
        `;

        this.body.innerHTML = html;
        this.open();

        const card = document.getElementById(`evidence-card-${evidenceIndex}`);
        if (card) {
            card.classList.add('highlighted');
            setTimeout(() => card.classList.remove('highlighted'), 3000);
        }
    }
};
