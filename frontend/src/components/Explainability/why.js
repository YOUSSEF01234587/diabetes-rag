window.whyComponent = {
    show(responseIndex) {
        const data = window.app ? window.app.lastResponses[responseIndex] : null;
        if (!data) return;

        const modal = document.getElementById('why-modal');
        const content = document.getElementById('why-content');

        const items = [];

        items.push({ icon: 'pass', label: 'Source', text: this.getSourceSummary(data) });

        if (data.evidence && data.evidence.length > 0) {
            const sections = [...new Set(data.evidence.map(e => e.section).filter(Boolean))];
            if (sections.length > 0) {
                items.push({ icon: 'info', label: 'Matching sections', text: sections.join(', ') });
            }
        }

        if (data.verification) {
            const v = data.verification;
            const checks = v.checks || {};
            if (checks.citations) {
                items.push({
                    icon: checks.citations.passed ? 'pass' : 'warn',
                    label: 'Citation validation',
                    text: checks.citations.passed ? 'Passed' : 'Issues detected'
                });
            }
            if (checks.numerical) {
                items.push({
                    icon: checks.numerical.passed ? 'pass' : 'warn',
                    label: 'Numeric consistency',
                    text: checks.numerical.passed ? 'Passed' : 'Issues detected'
                });
            }
            if (checks.sources) {
                items.push({
                    icon: checks.sources.passed ? 'pass' : 'warn',
                    label: 'Source validation',
                    text: checks.sources.passed ? 'Passed' : 'Issues detected'
                });
            }
            if (checks.hallucination) {
                items.push({
                    icon: checks.hallucination.passed ? 'pass' : 'warn',
                    label: 'Hallucination check',
                    text: checks.hallucination.passed ? 'Passed — no fabricated content detected' : 'Potential issues detected'
                });
            }
        }

        if (data.confidence) {
            items.push({
                icon: data.confidence === 'high' ? 'pass' : 'info',
                label: 'Evidence confidence',
                text: helpers.getConfidenceLabel(data.confidence.toLowerCase())
            });
        }

        const uniqueSources = helpers.getUniqueSources(data.sources || []);
        items.push({
            icon: 'info',
            label: 'Source diversity',
            text: uniqueSources.length > 1 ? `${uniqueSources.length} independent sources support this answer` : 'Single source'
        });

        const evidenceValidation = data.evidence_validation || {};
        const conflictReport = evidenceValidation.conflict_report || {};
        if (conflictReport.total_conflicts && conflictReport.total_conflicts > 0) {
            items.push({
                icon: 'warn',
                label: 'Source conflicts',
                text: `${conflictReport.total_conflicts} conflict(s) detected`
            });
        } else {
            items.push({ icon: 'pass', label: 'Source conflicts', text: 'None detected' });
        }

        if (data.grounded) {
            items.push({ icon: 'pass', label: 'Evidence grounding', text: 'Answer is grounded in retrieved evidence' });
        }

        if (data.evidence && data.evidence.length > 0) {
            const evValidation = evidenceValidation.evidence_summary || {};
            if (evValidation.section_coherence) {
                const coherence = evValidation.section_coherence > 0.7 ? 'High' : evValidation.section_coherence > 0.4 ? 'Moderate' : 'Low';
                items.push({ icon: 'info', label: 'Section coherence', text: coherence });
            }
            if (evValidation.source_agreement) {
                const agreement = evValidation.source_agreement > 0.7 ? 'High' : evValidation.source_agreement > 0.4 ? 'Moderate' : 'Low';
                items.push({ icon: 'info', label: 'Evidence agreement', text: agreement });
            }
        }

        let html = '<ul class="why-list">';
        for (const item of items) {
            const iconSvg = item.icon === 'pass'
                ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
                : item.icon === 'warn'
                ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
                : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>';

            html += `<li><span class="why-icon ${item.icon}">${iconSvg}</span><div><span class="why-label">${helpers.sanitize(item.label)}:</span> ${helpers.sanitize(item.text)}</div></li>`;
        }
        html += '</ul>';

        content.innerHTML = html;
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';

        const closeBtn = modal.querySelector('.modal-close');
        const backdrop = modal.querySelector('.modal-backdrop');
        const close = () => { modal.classList.add('hidden'); document.body.style.overflow = ''; };
        closeBtn.onclick = close;
        backdrop.onclick = close;
    },

    getSourceSummary(data) {
        const uniqueSources = helpers.getUniqueSources(data.sources || []);
        if (uniqueSources.length === 0) return 'No sources';
        return uniqueSources.map(s => s.source_label || s.organization || 'Unknown').join('; ');
    }
};
