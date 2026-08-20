window.verificationComponent = {
    render(verification) {
        if (!verification) return '';

        const passed = verification.passed;
        const issues = verification.issues || [];
        const checks = verification.checks || {};

        let statusClass = passed ? 'passed' : 'failed';
        let statusText = passed ? 'Evidence Verified' : 'Verification Issues Detected';

        if (!passed && issues.length > 0) {
            const hasPartial = Object.values(checks).some(c => c && c.passed);
            if (hasPartial) {
                statusClass = 'failed';
                statusText = 'Partially Verified — Review Recommended';
            }
        }

        let html = `
            <div class="verification-status ${statusClass}" role="status" aria-label="Verification: ${statusText}">
                ${helpers.getVerificationIcon(passed)}
                <span>${statusText}</span>
            </div>
        `;

        const checkNames = {
            citations: 'Citation Validation',
            sources: 'Source Validation',
            numerical: 'Numeric Consistency',
            hallucination: 'Hallucination Check',
            refusal: 'Refusal Check'
        };

        const checkEntries = Object.entries(checks).filter(([, v]) => v && typeof v === 'object' && 'passed' in v);
        if (checkEntries.length > 0) {
            html += '<div class="verification-checks">';
            for (const [key, check] of checkEntries) {
                const name = checkNames[key] || key;
                const checkClass = check.passed ? 'pass' : 'fail';
                const icon = check.passed ? '✓' : '✗';
                html += `<span class="verification-check ${checkClass}">${icon} ${name}</span>`;
            }
            html += '</div>';
        }

        return html;
    }
};
