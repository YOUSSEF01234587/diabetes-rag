window.pipelineComponent = {
    stages: ['intent', 'retrieve', 'evidence', 'conflict', 'generate', 'citation', 'verify', 'safety', 'answer'],

    reset() {
        this.stages.forEach(stage => {
            const el = document.querySelector(`.pipeline-step[data-step="${stage}"]`);
            if (el) {
                el.className = 'pipeline-step';
            }
        });
    },

    animateToStage(completedStageId) {
        const completedIndex = this.stages.indexOf(completedStageId);
        this.stages.forEach((stage, index) => {
            const el = document.querySelector(`.pipeline-step[data-step="${stage}"]`);
            if (!el) return;
            if (index <= completedIndex) {
                el.className = 'pipeline-step completed';
            } else if (index === completedIndex + 1) {
                el.className = 'pipeline-step active';
            } else {
                el.className = 'pipeline-step';
            }
        });
    },

    resetAfterResponse() {
        setTimeout(() => {
            this.stages.forEach(stage => {
                const el = document.querySelector(`.pipeline-step[data-step="${stage}"]`);
                if (el) el.className = 'pipeline-step';
            });
        }, 1500);
    }
};
