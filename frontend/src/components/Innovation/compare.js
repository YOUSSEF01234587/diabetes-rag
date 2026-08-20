window.compareSourcesComponent = {
    render(data) {
        if (!data || data.refused) return '';

        const sources = data.sources || [];
        const evidence = data.evidence || [];
        const uniqueSources = helpers.getUniqueSources(sources);

        if (uniqueSources.length < 2) return '';

        const conflicts = (data.evidence_validation || {}).conflict_report || {};
        const hasConflict = conflicts.total_conflicts && conflicts.total_conflicts > 0;

        let html = `<div class="compare-sources">`;
        html += `<div class="compare-header">`;
        html += `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>`;
        html += `<span>Compare Sources (${uniqueSources.length})</span>`;
        html += `</div>`;

        if (hasConflict) {
            html += `<div class="compare-conflict-notice">`;
            html += `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
            html += `<span>Clinical evidence differs between sources</span>`;
            html += `</div>`;
        }

        html += '<div class="compare-table-wrap"><table class="compare-table"><thead><tr><th>Source</th><th>Section</th><th>Organization</th><th>Evidence</th></tr></thead><tbody>';

        for (const src of uniqueSources) {
            const srcEvidence = evidence.filter(e => e.source_id === src.source_id);
            html += '<tr>';
            html += `<td class="compare-source-name">${helpers.sanitize(src.source_label || '')}</td>`;
            html += `<td>${helpers.sanitize(src.section || 'N/A')}</td>`;
            html += `<td>${helpers.sanitize(src.organization || 'N/A')}</td>`;
            html += `<td>${srcEvidence.length} chunk${srcEvidence.length !== 1 ? 's' : ''}</td>`;
            html += '</tr>';
        }

        html += '</tbody></table></div>';

        if (hasConflict) {
            const details = conflicts.conflict_details || [];
            if (details.length > 0) {
                html += '<div class="compare-diff-list">';
                for (const d of details.slice(0, 3)) {
                    html += `<div class="compare-diff-item">⚠ ${helpers.sanitize(d.description || d.type || 'Conflict')}</div>`;
                }
                html += '</div>';
            }
        }

        html += '</div>';
        return html;
    }
};
