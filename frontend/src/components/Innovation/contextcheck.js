window.contextCheckComponent = {
    render(data) {
        if (!data || data.refused) return '';

        const evidence = data.evidence || [];
        const sources = data.sources || [];
        const conflicts = (data.evidence_validation || {}).conflict_report || {};
        const verification = data.verification || {};

        const orgs = [...new Set(evidence.map(e => e.organization).filter(Boolean))];
        const sections = [...new Set(evidence.map(e => e.section).filter(Boolean))];

        const checks = [];

        if (orgs.length <= 2) {
            checks.push({ pass: true, label: 'Population consistent' });
        } else {
            checks.push({ pass: false, label: 'Multiple source populations' });
        }

        if (conflicts.total_conflicts && conflicts.total_conflicts > 0) {
            const types = [];
            if (conflicts.has_population_conflict) types.push('population');
            if (conflicts.has_threshold_conflict) types.push('threshold');
            if (conflicts.has_source_disagreement) types.push('source');
            checks.push({ pass: false, label: `Context conflict: ${types.join(', ')}` });
        } else {
            checks.push({ pass: true, label: 'No detected context conflict' });
        }

        if (verification.passed) {
            checks.push({ pass: true, label: 'Verification passed' });
        } else if (verification.issues && verification.issues.length > 0) {
            checks.push({ pass: false, label: 'Verification requires review' });
        }

        if (checks.length === 0) return '';

        const allPass = checks.every(c => c.pass);
        let html = `<div class="context-check ${allPass ? 'clean' : 'warn'}">`;
        html += '<div class="context-check-header">';
        html += allPass
            ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
            : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
        html += `<span class="context-check-title">${allPass ? 'Clinical Context Check' : 'Context Requires Attention'}</span>`;
        html += '</div>';
        html += '<div class="context-check-list">';
        for (const check of checks) {
            const icon = check.pass
                ? '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M20 6L9 17l-5-5"/></svg>'
                : '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
            html += `<span class="context-check-item ${check.pass ? 'pass' : 'fail'}">${icon} ${check.label}</span>`;
        }
        html += '</div></div>';

        return html;
    }
};
