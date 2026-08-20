window.evidenceMapComponent = {
    render(data) {
        if (!data || !data.evidence || data.evidence.length === 0) return '';

        const sources = data.sources || [];
        const evidence = data.evidence || [];
        const uniqueSources = helpers.getUniqueSources(sources);

        let html = `
            <div class="evidence-map">
                <div class="evidence-map-title">Evidence Map</div>
                <div class="evidence-map-tree">
                    <div class="evidence-map-node">
                        <span class="node-icon"></span>
                        <span>Question → ${evidence.length} evidence chunks</span>
                    </div>
        `;

        for (const src of uniqueSources) {
            const srcEvidence = evidence.filter(e => e.source_id === src.source_id);
            html += `
                    <div class="evidence-map-node source-node" style="padding-left: 16px;">
                        <span class="node-icon"></span>
                        <span>${helpers.sanitize(src.source_label || src.organization || 'Source')} (${srcEvidence.length})</span>
                    </div>
            `;
        }

        html += `
                </div>
            </div>
        `;

        return html;
    }
};
