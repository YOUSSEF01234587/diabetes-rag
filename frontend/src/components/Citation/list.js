window.citationComponent = {
    render(citations, evidence) {
        if (!citations || citations.length === 0) return '';

        let html = `
            <div class="citations-section">
                <div class="citations-title">Sources Cited</div>
                <div class="citations-list">
                    ${citations.map(cit => this.renderItem(cit, evidence)).join('')}
                </div>
            </div>
        `;

        return html;
    },

    renderItem(citation, evidence) {
        const evIndex = citation.evidence_index;
        const sourceTitle = citation.source_title || citation.organization || 'Unknown';
        const section = citation.section;
        const page = citation.page;

        let displayText = sourceTitle;
        if (section) displayText += `, ${section}`;
        if (page) displayText += `, p.${page}`;

        return `
            <span class="citation-item"
                  role="button"
                  tabindex="0"
                  aria-label="Citation ${evIndex}: ${displayText}"
                  onclick="citationComponent.handleClick(${evIndex})"
                  onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();citationComponent.handleClick(${evIndex})}">
                <span class="citation-index">${evIndex}</span>
                <span>${helpers.sanitize(displayText)}</span>
            </span>
        `;
    },

    handleClick(index) {
        evidenceComponent.highlightCard(index);

        if (window.drawerComponent) {
            window.drawerComponent.openForEvidence(index);
        }
    }
};
