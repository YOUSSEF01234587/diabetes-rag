window.confidenceComponent = {
    render(confidence) {
        if (!confidence) return '';

        const level = confidence.toLowerCase();
        const label = helpers.getConfidenceLabel(level);

        return `
            <div class="confidence-section" aria-label="Confidence: ${label}">
                <div class="confidence-indicator">
                    <span class="confidence-label">Evidence Confidence:</span>
                    <div class="confidence-bar" role="progressbar" aria-valuenow="${this.getPercent(level)}" aria-valuemin="0" aria-valuemax="100">
                        <div class="confidence-fill ${level}"></div>
                    </div>
                    <span class="confidence-label">${label}</span>
                </div>
            </div>
        `;
    },

    getPercent(level) {
        const percents = { high: 100, moderate: 66, low: 33, insufficient: 0 };
        return percents[level] || 0;
    }
};
