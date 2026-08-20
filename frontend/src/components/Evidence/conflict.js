window.conflictDetectorComponent = {
    render(conflictReport) {
        if (!conflictReport || !conflictReport.total_conflicts || conflictReport.total_conflicts === 0) return '';

        const details = conflictReport.conflict_details || [];
        const hasSourceDisagreement = conflictReport.has_source_disagreement === true;
        const hasPopulationConflict = conflictReport.has_population_conflict === true;
        const hasThresholdConflict = conflictReport.has_threshold_conflict === true;

        const isContextual = hasThresholdConflict && !hasSourceDisagreement && !hasPopulationConflict;

        let severity, label, icon, types;
        if (isContextual) {
            severity = 'info';
            label = 'Clinical Context';
            icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>';
            types = ['Different diagnostic thresholds by test'];
        } else {
            severity = 'medium';
            label = 'Clinical Context Conflict';
            icon = helpers.getSafetyIcon('medium');
            types = [];
            if (hasPopulationConflict) types.push('Different population');
            if (hasThresholdConflict) types.push('Different threshold');
            if (hasSourceDisagreement) types.push('Source disagreement');
        }

        let html = `
            <div class="safety-notice ${severity}" role="status" aria-label="${helpers.sanitize(label)}">
                <span class="safety-icon">${icon}</span>
                <div class="safety-content">
                    <span class="safety-label">${helpers.sanitize(label)}</span>
                    <div class="safety-flags">
                        ${types.length > 0 ? types.map(t => helpers.sanitize(t)).join(' · ') : `${conflictReport.total_conflicts} difference(s) noted`}
                    </div>
        `;

        if (isContextual) {
            html += `<div class="safety-flags" style="margin-top:6px;color:var(--color-text-secondary);font-size:11px;line-height:1.5;">
                Different thresholds are reported because the guideline covers multiple diagnostic tests and clinical categories. This does not mean the sources disagree.
            </div>`;
        } else if (details.length > 0) {
            html += '<div class="safety-flags" style="margin-top:4px;">';
            for (const d of details.slice(0, 3)) {
                html += `<div>\u2022 ${helpers.sanitize(d.description || d.type)}</div>`;
            }
            html += '</div>';
        }

        html += `</div></div>`;
        return html;
    }
};
