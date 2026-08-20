window.howItWorksComponent = {
    init() {
        document.getElementById('how-it-works-btn').addEventListener('click', () => this.open());
    },

    open() {
        const modal = document.getElementById('how-it-works-modal');
        const content = document.getElementById('how-it-works-content');

        const steps = [
            { num: 1, title: 'Question', desc: 'You ask a clinical question about diabetes diagnosis or classification.' },
            { num: 2, title: 'Intent Detection', desc: 'The system classifies the question type: diagnostic criteria, comparison, gestational, emergency, or medication.' },
            { num: 3, title: 'Hybrid Retrieval', desc: 'Combines dense vector search (semantic similarity) with sparse keyword search (BM25) for comprehensive evidence retrieval.' },
            { num: 4, title: 'Evidence Selection', desc: 'Selects the most relevant evidence chunks using fusion scoring, deduplication, and section diversity.' },
            { num: 5, title: 'Conflict Detection', desc: 'Identifies clinical conflicts: different populations, thresholds, guidelines, or clinical contexts.' },
            { num: 6, title: 'Grounded Generation', desc: 'Generates an answer using ONLY the retrieved evidence. Every claim must cite a specific source.' },
            { num: 7, title: 'Citation Validation', desc: 'Validates that every [Evidence N] reference in the answer corresponds to actual retrieved evidence.' },
            { num: 8, title: 'Answer Verification', desc: 'Checks numerical consistency, source alignment, and hallucination detection.' },
            { num: 9, title: 'Safety Check', desc: 'Evaluates clinical safety: risk level, medication boundaries, emergency detection.' },
            { num: 10, title: 'Verified Answer', desc: 'Delivers the evidence-grounded answer with full traceability to source documents.' }
        ];

        content.innerHTML = `
            <div class="hiw-flow">
                ${steps.map((step, i) => `
                    <div class="hiw-step">
                        <div class="hiw-step-line">
                            <div class="hiw-step-dot">${step.num}</div>
                            ${i < steps.length - 1 ? '<div class="hiw-step-connector"></div>' : ''}
                        </div>
                        <div class="hiw-step-content">
                            <div class="hiw-step-title">${step.title}</div>
                            <div class="hiw-step-desc">${step.desc}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';

        const closeBtn = modal.querySelector('.modal-close');
        const backdrop = modal.querySelector('.modal-backdrop');
        const close = () => { modal.classList.add('hidden'); document.body.style.overflow = ''; };
        closeBtn.onclick = close;
        backdrop.onclick = close;

        document.addEventListener('keydown', function handler(e) {
            if (e.key === 'Escape') {
                close();
                document.removeEventListener('keydown', handler);
            }
        });
    }
};
