window.helpers = {
    formatTime(ms) {
        if (!ms) return '';
        if (ms < 1000) return `${Math.round(ms)}ms`;
        return `${(ms / 1000).toFixed(1)}s`;
    },

    formatTimestamp() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    getSafetyIcon(level) {
        const icons = {
            low: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
            medium: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
            high: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
        };
        return icons[level] || icons.low;
    },

    getSafetyLabel(level) {
        const labels = {
            low: 'Low Risk',
            medium: 'Moderate Risk',
            high: 'High Risk'
        };
        return labels[level] || 'Unknown';
    },

    getConfidenceClass(level) {
        return (level || 'insufficient').toLowerCase();
    },

    getConfidenceLabel(level) {
        const labels = {
            high: 'High Confidence',
            moderate: 'Moderate Confidence',
            low: 'Low Confidence',
            insufficient: 'Insufficient Evidence'
        };
        return labels[level] || 'Unknown';
    },

    getVerificationIcon(passed) {
        return passed
            ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
            : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
    },

    sanitize(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },

    hasTableContent(text) {
        if (!text) return false;
        return /\|.*\|.*\|/.test(text) || /[\t].*[\t]/.test(text);
    },

    getUniqueSources(sources) {
        const seen = new Set();
        const unique = [];
        for (const s of sources) {
            const key = s.source_id || s.source_label;
            if (!seen.has(key)) {
                seen.add(key);
                unique.push(s);
            }
        }
        return unique;
    },

    detectContext(text) {
        if (!text) return [];
        const contexts = [];
        const lower = text.toLowerCase();
        if (lower.includes('gestational')) contexts.push('Gestational');
        if (lower.includes('prediabetes') || lower.includes('pre-diabetes')) contexts.push('Prediabetes');
        if (lower.includes('type 1') || lower.includes('type1')) contexts.push('Type 1');
        if (lower.includes('type 2') || lower.includes('type2')) contexts.push('Type 2');
        if (lower.includes('pediatric') || lower.includes('child')) contexts.push('Pediatric');
        if (lower.includes('screening') || lower.includes('screen')) contexts.push('Screening');
        if (lower.includes('diagnosis') || lower.includes('diagnos')) contexts.push('Diagnosis');
        if (lower.includes('monitoring') || lower.includes('monitor') || lower.includes('hba1c')) contexts.push('Monitoring');
        if (contexts.length === 0) contexts.push('General');
        return contexts;
    }
};
