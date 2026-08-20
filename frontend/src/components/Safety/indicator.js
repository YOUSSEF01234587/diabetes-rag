window.safetyComponent = {
    render(safety) {
        if (!safety || !safety.risk_level) return '';

        const level = safety.risk_level;
        const flags = safety.flags || safety.risk_flags || [];
        const requiresPro = safety.requires_professional;
        const hasNote = safety.has_safety_note;

        let html = `
            <div class="safety-notice ${level}" role="alert" aria-label="Safety notice: ${level} risk">
                <span class="safety-icon">${helpers.getSafetyIcon(level)}</span>
                <div class="safety-content">
                    <span class="safety-label">Clinical Safety: ${helpers.getSafetyLabel(level)}</span>
        `;

        if (flags.length > 0) {
            html += `<div class="safety-flags">Flags: ${flags.map(f => helpers.sanitize(f.replace(/_/g, ' '))).join(', ')}</div>`;
        }

        if (requiresPro) {
            html += `<div class="safety-flags">This information requires professional medical evaluation.</div>`;
        }

        html += `</div></div>`;
        return html;
    }
};
