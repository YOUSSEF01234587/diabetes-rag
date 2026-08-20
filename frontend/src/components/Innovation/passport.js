window.sourcePassportComponent = {
    render(data) {
        if (!data || data.refused) return '';

        const sources = data.sources || [];
        const uniqueSources = helpers.getUniqueSources(sources);

        if (uniqueSources.length === 0) return '';

        let html = '<div class="source-passports">';
        html += '<div class="passport-title">Source Passports</div>';

        for (const src of uniqueSources) {
            html += '<div class="passport-card">';
            html += '<div class="passport-doc">';
            html += '<div class="passport-doc-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>';
            html += `<div class="passport-doc-name">${helpers.sanitize(src.source_label || src.organization || 'Source')}</div>`;
            html += '</div>';
            html += '<div class="passport-arrow">↓</div>';
            html += '<div class="passport-section">';
            html += `<span class="passport-label">Section</span>`;
            html += `<span class="passport-value">${helpers.sanitize(src.section || 'N/A')}</span>`;
            html += '</div>';
            html += '<div class="passport-arrow">↓</div>';
            html += '<div class="passport-page">';
            html += `<span class="passport-label">Page</span>`;
            html += `<span class="passport-value">${src.page || 'N/A'}</span>`;
            html += '</div>';
            html += '<div class="passport-arrow">↓</div>';
            html += '<div class="passport-evidence">';
            html += `<span class="passport-label">Evidence Used</span>`;
            html += `<span class="passport-value">Referenced in answer</span>`;
            html += '</div>';
            if (src.doi) {
                html += `<div class="passport-doi">DOI: ${helpers.sanitize(src.doi)}</div>`;
            }
            html += '</div>';
        }

        html += '</div>';
        return html;
    }
};
