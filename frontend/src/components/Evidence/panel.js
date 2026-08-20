window.evidenceComponent = {
    render(evidence, sources) {
        if (!evidence || evidence.length === 0) return '';

        const evidenceId = `evidence-${Date.now()}`;

        let html = `
            <div class="evidence-section">
                <div class="evidence-header"
                     role="button"
                     tabindex="0"
                     aria-expanded="false"
                     aria-controls="${evidenceId}"
                     onclick="evidenceComponent.toggle('${evidenceId}', this)"
                     onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();evidenceComponent.toggle('${evidenceId}', this)}">
                    <span class="evidence-title">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        Evidence (${evidence.length} sources)
                    </span>
                    <svg class="evidence-toggle" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </div>
                <div class="evidence-list" id="${evidenceId}">
                    ${evidence.map((item, index) => this.renderCard(item, index + 1, sources)).join('')}
                </div>
            </div>
        `;

        return html;
    },

    renderCard(item, index, sources) {
        const text = item.text_preview || item.text || '';
        const hasTable = helpers.hasTableContent(text);
        const source = sources ? sources.find(s => s.source_id === item.source_id) : null;
        const sourceLabel = source ? source.source_label : (item.organization || 'Unknown Source');

        let metaHtml = '';
        if (item.page) metaHtml += `<span>Page ${item.page}</span>`;
        if (item.section) metaHtml += `<span>${helpers.sanitize(item.section)}</span>`;
        if (hasTable) metaHtml += '<span class="badge-table">Table</span>';

        let html = `
            <div class="evidence-card" id="evidence-card-${index}"
                 role="article"
                 aria-label="Evidence ${index}"
                 onclick="evidenceComponent.toggleText(this)"
                 onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();evidenceComponent.toggleText(this)}">
                <div class="evidence-card-header">
                    <div class="evidence-card-index">
                        <span class="evidence-number">${index}</span>
                        <span class="evidence-source">${helpers.sanitize(sourceLabel)}</span>
                    </div>
                    <div class="evidence-meta">${metaHtml}</div>
                </div>
                <div class="evidence-text">${this.renderText(text, hasTable)}</div>
                <div class="evidence-expand-hint">Click to expand</div>
            </div>
        `;

        return html;
    },

    renderText(text, hasTable) {
        if (!text) return '<em>No text available</em>';

        if (hasTable) {
            return tableRenderer.renderInEvidence(text);
        }

        return helpers.sanitize(text);
    },

    toggle(id, header) {
        const list = document.getElementById(id);
        const toggle = header.querySelector('.evidence-toggle');
        const isExpanded = list.classList.contains('open');

        list.classList.toggle('open');
        toggle.classList.toggle('open');
        header.setAttribute('aria-expanded', !isExpanded);
    },

    toggleText(card) {
        const textEl = card.querySelector('.evidence-text');
        const hintEl = card.querySelector('.evidence-expand-hint');

        textEl.classList.toggle('expanded');
        hintEl.textContent = textEl.classList.contains('expanded') ? 'Click to collapse' : 'Click to expand';
    },

    highlightCard(index) {
        document.querySelectorAll('.evidence-card.highlighted').forEach(c => c.classList.remove('highlighted'));
        const card = document.getElementById(`evidence-card-${index}`);
        if (card) {
            card.classList.add('highlighted');
            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
            setTimeout(() => card.classList.remove('highlighted'), 3000);
        }
    }
};
