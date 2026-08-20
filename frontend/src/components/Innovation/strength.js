window.evidenceStrengthComponent = {
    render(data) {
        if (!data || data.refused) return '';

        const evidence = data.evidence || [];
        const verification = data.verification || {};
        const confidence = (data.confidence || '').toLowerCase();
        const grounded = data.grounded;
        const sources = data.sources || [];
        const uniqueSources = helpers.getUniqueSources(sources);

        let level = 'limited';
        let reason = '';

        if (confidence === 'high' && verification.passed && grounded) {
            level = 'strong';
            reason = 'Multiple evidence items with strong validation';
        } else if (confidence === 'moderate' && verification.passed) {
            level = 'moderate';
            reason = 'Useful evidence with some limitations';
        } else if (evidence.length >= 1 && verification.passed) {
            level = 'moderate';
            reason = 'Evidence available and verified';
        } else if (evidence.length >= 1) {
            level = 'limited';
            reason = 'Evidence available but limited validation';
        } else {
            level = 'limited';
            reason = 'No evidence available';
        }

        const labels = { strong: 'Strong', moderate: 'Moderate', limited: 'Limited' };
        const icons = {
            strong: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
            moderate: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>',
            limited: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
        };

        return `
            <div class="evidence-strength ${level}">
                <div class="evidence-strength-header">
                    ${icons[level]}
                    <span class="evidence-strength-label">Evidence Strength: <strong>${labels[level]}</strong></span>
                    <button class="evidence-strength-why" onclick="evidenceStrengthComponent.showWhy(${window.app ? window.app.currentResponseIndex : 0})" title="Why this strength?">Why?</button>
                </div>
                <div class="evidence-strength-factors">
                    <span>${evidence.length} item${evidence.length !== 1 ? 's' : ''}</span>
                    <span>\u00B7</span>
                    <span>${uniqueSources.length} source${uniqueSources.length !== 1 ? 's' : ''}</span>
                    <span>\u00B7</span>
                    <span>${verification.passed ? 'Verified' : 'Partial verification'}</span>
                </div>
            </div>
        `;
    },

    showWhy(index) {
        const data = window.app ? window.app.lastResponses[index] : null;
        if (!data) return;

        const modal = document.getElementById('why-modal');
        const content = document.getElementById('why-content');

        const evidence = data.evidence || [];
        const verification = data.verification || {};
        const sources = data.sources || [];
        const uniqueSources = helpers.getUniqueSources(sources);

        let html = '<div class="strength-explanation">';
        html += '<h4 style="margin-bottom:12px;font-size:14px;">Evidence Strength Factors</h4>';
        html += '<ul class="why-list">';

        html += `<li><span class="why-icon info"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg></span><div><span class="why-label">Evidence count:</span> ${evidence.length} chunks retrieved</div></li>`;
        html += `<li><span class="why-icon info"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg></span><div><span class="why-label">Source diversity:</span> ${uniqueSources.length} independent source(s)</div></li>`;
        html += `<li><span class="why-icon ${verification.passed ? 'pass' : 'warn'}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="${verification.passed ? 'M22 11.08V12a10 10 0 1 1-5.93-9.14' : 'M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'}"/><path d="${verification.passed ? 'M22 4L12 14.01 9 11.01' : 'M12 9v4M12 17h.01'}"/></svg></span><div><span class="why-label">Verification:</span> ${verification.passed ? 'Passed' : 'Issues detected'}</div></li>`;
        html += `<li><span class="why-icon ${data.grounded ? 'pass' : 'warn'}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="${data.grounded ? 'M22 11.08V12a10 10 0 1 1-5.93-9.14' : 'M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'}"/><path d="${data.grounded ? 'M22 4L12 14.01 9 11.01' : 'M12 9v4M12 17h.01'}"/></svg></span><div><span class="why-label">Grounding:</span> ${data.grounded ? 'Answer grounded in evidence' : 'Limited grounding'}</div></li>`;

        html += '</ul></div>';

        content.innerHTML = html;
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';

        const close = () => { modal.classList.add('hidden'); document.body.style.overflow = ''; };
        modal.querySelector('.modal-close').onclick = close;
        modal.querySelector('.modal-backdrop').onclick = close;
    }
};
