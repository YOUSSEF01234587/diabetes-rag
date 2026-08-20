window.evidenceHeatmapComponent = {
    render(data) {
        if (!data || data.refused) return '';

        const evidence = data.evidence || [];
        const sources = data.sources || [];

        if (evidence.length === 0) return '';

        const positions = ['Primary', 'Supporting', 'Supporting', 'Context', 'Context'];
        const colors = ['heatmap-primary', 'heatmap-support', 'heatmap-support', 'heatmap-context', 'heatmap-context'];

        let html = '<div class="evidence-heatmap">';
        html += '<div class="heatmap-title">Evidence Contribution</div>';
        html += '<div class="heatmap-items">';

        for (let i = 0; i < evidence.length; i++) {
            const ev = evidence[i];
            const pos = positions[i] || 'Context';
            const color = colors[i] || 'heatmap-context';
            const source = sources.find(s => s.source_id === ev.source_id);
            const sourceName = source ? (source.source_label || source.organization || '') : '';

            html += `<div class="heatmap-item ${color}">`;
            html += `<div class="heatmap-rank">E${i + 1}</div>`;
            html += `<div class="heatmap-info">`;
            html += `<div class="heatmap-source">${helpers.sanitize(sourceName)}</div>`;
            html += `<div class="heatmap-position">${pos}</div>`;
            html += `</div>`;
            html += `</div>`;
        }

        html += '</div></div>';

        return html;
    }
};
