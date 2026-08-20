window.answerTraceComponent = {
    render(data, questionText) {
        if (!data) return '';

        const isRefused = data.refused === true;
        const evidence = data.evidence || [];
        const sources = data.sources || [];
        const uniqueSources = helpers.getUniqueSources(sources);
        const verification = data.verification || {};
        const safety = data.safety || {};

        const steps = [];

        steps.push({ icon: 'Q', label: 'Question', detail: questionText || 'Clinical question', done: true });

        if (evidence.length > 0) {
            const sections = [...new Set(evidence.map(e => e.section).filter(Boolean))];
            steps.push({ icon: 'E', label: 'Evidence Retrieved', detail: `${evidence.length} chunks from ${uniqueSources.length} source(s)`, done: true });
        }

        if (uniqueSources.length > 0) {
            steps.push({ icon: 'S', label: 'Sources Identified', detail: uniqueSources.map(s => s.source_label || s.organization || '').filter(Boolean).join(', '), done: true });
        }

        const contexts = helpers.detectContext(evidence.map(e => e.section || '').join(' '));
        if (contexts.length > 0 && contexts[0] !== 'General') {
            steps.push({ icon: 'C', label: 'Clinical Context', detail: contexts.join(', '), done: true });
        }

        if (!isRefused && verification) {
            steps.push({ icon: 'V', label: 'Verification', detail: verification.passed ? 'Passed' : 'Issues detected', done: true });
        }

        if (safety && safety.risk_level) {
            steps.push({ icon: '!', label: 'Safety Check', detail: helpers.getSafetyLabel(safety.risk_level), done: true });
        }

        steps.push({ icon: 'A', label: isRefused ? 'Refusal' : 'Final Answer', detail: isRefused ? 'Response refused for safety' : 'Evidence-grounded response', done: true });

        let html = '<div class="answer-trace">';
        html += '<div class="trace-header"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>Answer Trace</div>';
        html += '<div class="trace-timeline">';
        for (const step of steps) {
            html += `<div class="trace-step">`;
            html += `<div class="trace-dot">${step.icon}</div>`;
            html += `<div class="trace-connector"></div>`;
            html += `<div class="trace-content"><div class="trace-label">${step.label}</div><div class="trace-detail">${helpers.sanitize(step.detail)}</div></div>`;
            html += `</div>`;
        }
        html += '</div></div>';

        return html;
    }
};
