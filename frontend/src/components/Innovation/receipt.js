window.evidenceReceiptComponent = {
    render(data) {
        if (!data || data.refused) return '';

        const evidence = data.evidence || [];
        const sources = data.sources || [];
        const citations = data.citations || [];
        const verification = data.verification || {};
        const safety = data.safety || {};
        const uniqueSources = helpers.getUniqueSources(sources);

        const sections = [...new Set(evidence.map(e => e.section).filter(Boolean))];
        const pages = [...new Set(evidence.map(e => e.page).filter(Boolean))];

        const checks = verification.checks || {};
        const citationPassed = checks.citations ? checks.citations.passed : null;

        let html = '<div class="evidence-receipt">';
        html += '<div class="receipt-header">';
        html += '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>';
        html += '<span class="receipt-title">Evidence Receipt</span>';
        html += '</div>';
        html += '<div class="receipt-grid">';
        html += `<div class="receipt-row"><span class="receipt-key">Source(s)</span><span class="receipt-val">${uniqueSources.map(s => helpers.sanitize(s.source_label || s.organization || '')).join('; ') || 'N/A'}</span></div>`;
        if (sections.length > 0) html += `<div class="receipt-row"><span class="receipt-key">Section(s)</span><span class="receipt-val">${sections.slice(0, 3).map(s => helpers.sanitize(s)).join('; ')}${sections.length > 3 ? '...' : ''}</span></div>`;
        if (pages.length > 0) html += `<div class="receipt-row"><span class="receipt-key">Page(s)</span><span class="receipt-val">${pages.join(', ')}</span></div>`;
        html += `<div class="receipt-row"><span class="receipt-key">Evidence</span><span class="receipt-val">${evidence.length} chunk${evidence.length !== 1 ? 's' : ''}</span></div>`;
        html += `<div class="receipt-row"><span class="receipt-key">Citation check</span><span class="receipt-val receipt-${citationPassed === true ? 'pass' : citationPassed === false ? 'fail' : 'na'}">${citationPassed === true ? '\u2713 Passed' : citationPassed === false ? '\u2717 Issues' : 'N/A'}</span></div>`;
        html += `<div class="receipt-row"><span class="receipt-key">Verification</span><span class="receipt-val receipt-${verification.passed ? 'pass' : 'fail'}">${verification.passed ? '\u2713 Passed' : '\u2717 Issues'}</span></div>`;
        html += `<div class="receipt-row"><span class="receipt-key">Conflicts</span><span class="receipt-val">${this.getConflictStatus(data)}</span></div>`;
        html += '</div></div>';

        return html;
    },

    getConflictStatus(data) {
        const ev = data.evidence_validation || {};
        const conflicts = ev.conflict_report || {};
        if (conflicts.total_conflicts && conflicts.total_conflicts > 0) {
            return `<span class="receipt-fail">\u26A0 ${conflicts.total_conflicts} conflict(s)</span>`;
        }
        return '<span class="receipt-pass">\u2713 None</span>';
    }
};
