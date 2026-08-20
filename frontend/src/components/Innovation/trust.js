window.trustLayerComponent = {
    render(data) {
        if (!data) return '';

        const isRefused = data.refused === true;
        const checks = [];

        if (!isRefused && data.evidence && data.evidence.length > 0) {
            checks.push({ pass: true, label: 'Evidence retrieved' });
        } else if (isRefused) {
            checks.push({ pass: false, label: 'No evidence used' });
        }

        if (data.citations && data.citations.length > 0) {
            checks.push({ pass: true, label: 'Citations validated' });
        }

        if (data.verification) {
            checks.push({ pass: data.verification.passed, label: 'Answer verified' });
        }

        if (data.safety && data.safety.risk_level) {
            checks.push({ pass: true, label: 'Safety checked' });
        }

        if (data.grounded) {
            checks.push({ pass: true, label: 'Evidence grounded' });
        }

        if (checks.length === 0) return '';

        let html = '<div class="trust-layer">';
        html += '<div class="trust-label">Trust Layer</div>';
        html += '<div class="trust-checks">';
        for (const check of checks) {
            const icon = check.pass
                ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>'
                : '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/></svg>';
            html += `<span class="trust-check ${check.pass ? 'pass' : 'fail'}">${icon} ${check.label}</span>`;
        }
        html += '</div></div>';

        return html;
    }
};
