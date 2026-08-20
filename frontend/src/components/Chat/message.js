window.messageComponent = {
    createUserMessage(text) {
        return `
            <div class="message user" role="article" aria-label="Your question">
                <div class="message-avatar" aria-hidden="true">U</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-role">You</span>
                        <span class="message-time">${helpers.formatTimestamp()}</span>
                    </div>
                    <div class="message-body">
                        ${helpers.sanitize(text)}
                    </div>
                </div>
            </div>
        `;
    },

    createAssistantMessage(data) {
        const { answer, confidence, grounded, citations, sources, evidence,
                refused, refusal_reason, query_type, safety, verification,
                timings, total_ms, evidence_validation } = data;

        const isRefused = refused === true;
        const uniqueSources = helpers.getUniqueSources(sources || []);
        const primarySections = [];
        const advancedSections = [];

        if (isRefused) {
            primarySections.push(trustLayerComponent.render(data));
            primarySections.push(contextCheckComponent.render(data));
            primarySections.push(this.renderRefusal(answer, refusal_reason, evidence, sources));
        } else {
            primarySections.push(`<div class="answer-section"><div class="answer-label">Clinical Answer</div><div class="answer-text">${markdown.render(answer)}</div></div>`);
        }

        if (isRefused) {
            const isSafety = ['emergency', 'medical_advice'].includes(refusal_reason);
            const isProvider = ['provider_failure', 'llm_error'].includes(refusal_reason);
            let badgeClass, badgeLabel;
            if (isSafety) {
                badgeClass = 'badge-warning';
                badgeLabel = refusal_reason === 'emergency' ? 'Emergency Safety' : 'Clinical Safety Boundary';
            } else if (isProvider) {
                badgeClass = 'badge-warning';
                badgeLabel = 'Service Temporarily Unavailable';
            } else {
                badgeClass = 'badge-info';
                badgeLabel = 'Insufficient Evidence';
            }
            primarySections.push(`<div class="meta-badges"><span class="badge ${badgeClass}">${badgeLabel}</span></div>`);
        } else if (query_type) {
            const qLabel = helpers.sanitize(query_type.replace(/_/g, ' '));
            const intelIcon = window.app && window.app.lastQuestionIntelligence ? window.app.lastQuestionIntelligence.icon : '';
            primarySections.push(`<div class="meta-badges"><span class="badge badge-primary">${intelIcon} ${qLabel}</span>${grounded ? '<span class="badge badge-success">Evidence Grounded</span>' : ''}</div>`);
        }

        if (uniqueSources.length > 0) {
            primarySections.push(this.renderEvidenceSources(uniqueSources, evidence || []));
        }

        if (!isRefused && citations && citations.length > 0) {
            primarySections.push(citationComponent.render(citations, evidence));
        }

        if (!isRefused) {
            const secondary = [];
            if (evidenceStrengthComponent.render(data)) secondary.push(evidenceStrengthComponent.render(data));
            if (verificationComponent.render(verification)) secondary.push(verificationComponent.render(verification));
            if (safetyComponent.render(safety)) secondary.push(safetyComponent.render(safety));
            if (confidenceComponent.render(confidence)) secondary.push(confidenceComponent.render(confidence));
            if (evidenceReceiptComponent.render(data)) secondary.push(evidenceReceiptComponent.render(data));
            if (secondary.filter(Boolean).length > 0) {
                primarySections.push(`<div class="answer-section trust-section" style="padding:12px 20px;">${secondary.filter(Boolean).join('')}</div>`);
            }
        }

        if (!isRefused) {
            const advanced = [];
            if (evidence && evidence.length > 0) {
                advanced.push(evidenceComponent.render(evidence, sources));
            }
            if (evidence && evidence.length > 0 && sources && sources.length > 0) {
                advanced.push(evidenceMapComponent.render(data));
            }
            if (uniqueSources.length >= 2) {
                advanced.push(compareSourcesComponent.render(data));
            }
            if (evidence && evidence.length > 0) {
                const evValidation = evidence_validation || {};
                const conflictReport = evValidation.conflict_report || {};
                if (conflictReport.total_conflicts && conflictReport.total_conflicts > 0) {
                    advanced.push(conflictDetectorComponent.render(conflictReport));
                }
            }
            if (evidence && evidence.length > 0) {
                advanced.push(sourcePassportComponent.render(data));
            }
            if (evidence && evidence.length > 0) {
                advanced.push(evidenceHeatmapComponent.render(data));
            }
            advanced.push(answerTraceComponent.render(data, window.app ? window.app.lastQuestionText : ''));

            const advancedHtml = advanced.filter(Boolean).join('');
            if (advancedHtml) {
                primarySections.push(`
                    <details class="advanced-details">
                        <summary class="advanced-summary">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                            Explainability &amp; Evidence Details
                        </summary>
                        <div class="advanced-body">
                            ${advancedHtml}
                        </div>
                    </details>
                `);
            }
        }

        if (isRefused) {
            const advanced = [];
            if (evidence && evidence.length > 0) {
                advanced.push(evidenceComponent.render(evidence, sources));
            }
            advanced.push(answerTraceComponent.render(data, window.app ? window.app.lastQuestionText : ''));
            const advancedHtml = advanced.filter(Boolean).join('');
            if (advancedHtml) {
                primarySections.push(`
                    <details class="advanced-details">
                        <summary class="advanced-summary">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                            Explainability Details
                        </summary>
                        <div class="advanced-body">
                            ${advancedHtml}
                        </div>
                    </details>
                `);
            }
        }

        primarySections.push(`<div class="why-section"><button class="btn-why" onclick="whyComponent.show(${window.app ? window.app.currentResponseIndex : 0})"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>Why this answer?</button></div>`);

        if (timings || total_ms) {
            primarySections.push(this.renderTimings(timings, total_ms));
        }

        let html = `
            <div class="message assistant" role="article" aria-label="AI response">
                <div class="message-avatar" aria-hidden="true">AI</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-role">Clinical Evidence Copilot</span>
                        <span class="message-time">${helpers.formatTimestamp()}</span>
                    </div>
                    <div class="answer-card">
                        ${primarySections.filter(Boolean).join('')}
                    </div>
                </div>
            </div>
        `;

        return html;
    },

    renderEvidenceSources(uniqueSources, evidence) {
        const count = uniqueSources.length;
        const label = count === 1 ? '1 source used' : `${count} sources used`;

        const sourceCards = uniqueSources.map(s => {
            const shortName = s.short_title || s.source_title || s.source_id || 'Unknown';
            const org = s.organization || '';
            const year = s.year ? ` (${s.year})` : '';
            return `
                <div class="source-used-card">
                    <span class="source-used-check">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>
                    </span>
                    <div>
                        <div class="source-used-name">${helpers.sanitize(shortName)}${year}</div>
                        ${org ? `<div class="source-used-org">${helpers.sanitize(org)}</div>` : ''}
                    </div>
                </div>`;
        }).join('');

        return `
            <div class="evidence-sources-used">
                <div class="evidence-sources-label">Evidence Used \u00B7 ${label}</div>
                <div class="evidence-sources-grid">${sourceCards}</div>
            </div>`;
    },

    renderRefusal(answer, reason, evidence, sources) {
        const hasEvidence = evidence && evidence.length > 0;

        const variants = {
            no_evidence: {
                variant: 'info',
                icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>',
                title: 'Not enough information',
            },
            low_relevance: {
                variant: 'info',
                icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>',
                title: 'Insufficient evidence',
            },
            low_grounding: {
                variant: 'info',
                icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>',
                title: 'Insufficient evidence',
            },
            verification_failed: {
                variant: 'caution',
                icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 12l2 2 4-4"/><path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
                title: 'Answer could not be verified',
            },
            provider_failure: {
                variant: 'service',
                icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>',
                title: 'Answer temporarily unavailable',
            },
            llm_error: {
                variant: 'service',
                icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>',
                title: 'Answer temporarily unavailable',
            },
            emergency: {
                variant: 'safety',
                icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
                title: 'Emergency situation',
            },
            medical_advice: {
                variant: 'safety',
                icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
                title: 'Clinical safety boundary',
            },
            insufficient_evidence: {
                variant: 'info',
                icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>',
                title: 'Not enough information',
            },
        };

        const cfg = variants[reason] || variants.insufficient_evidence;

        const suggestions = {
            no_evidence: 'Try asking about diabetes diagnosis, A1C thresholds, FPG criteria, or OGTT testing.',
            low_relevance: 'Try asking about specific diagnostic tests like FPG, A1C, or OGTT thresholds.',
            low_grounding: 'Try asking about specific diagnostic tests or classification criteria.',
            verification_failed: 'Try rephrasing your question or ask about a more specific topic.',
            provider_failure: 'The evidence retrieval system is working. Please try again shortly.',
            insufficient_evidence: 'Try rephrasing your question about diabetes diagnosis or testing.',
        };
        const suggestion = suggestions[reason];

        let evidenceHtml = '';
        if (hasEvidence) {
            const sourceNames = helpers.getUniqueSources(sources || []);
            const sourceList = sourceNames.map(s => {
                const name = s.short_title || s.source_title || s.source_id || 'Source';
                return `<li>${helpers.sanitize(name)}</li>`;
            }).join('');

            evidenceHtml = `
                <div class="refusal-evidence-note">
                    <div class="refusal-evidence-header">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        Relevant evidence found
                    </div>
                    <p class="refusal-evidence-desc">These sources were retrieved, but they were not sufficient to produce a verified answer.</p>
                    ${sourceList ? `<ul class="refusal-evidence-list">${sourceList}</ul>` : ''}
                </div>`;
        }

        return `
            <div class="refusal-card refusal-card--${cfg.variant}">
                <div class="refusal-header">
                    <span class="refusal-icon">${cfg.icon}</span>
                    <h4 class="refusal-title">${cfg.title}</h4>
                </div>
                <div class="refusal-body">${markdown.render(answer)}</div>
                ${suggestion ? `<div class="refusal-suggestion">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>
                    ${helpers.sanitize(suggestion)}
                </div>` : ''}
                ${evidenceHtml}
            </div>
        `;
    },

    renderTimings(timings, totalMs) {
        let items = [];
        if (timings) {
            const retMs = timings.retrieval_ms || ((timings.embedding_ms || 0) + (timings.dense_ms || 0) + (timings.bm25_ms || 0));
            if (retMs > 0) items.push(`Retrieval: ${helpers.formatTime(retMs)}`);
            if (timings.llm_ms) {
                const prov = timings.provider ? ` (${timings.provider})` : '';
                items.push(`Generation: ${helpers.formatTime(timings.llm_ms)}${prov}`);
            }
        }
        if (totalMs) items.push(`Total: ${helpers.formatTime(totalMs)}`);

        if (items.length === 0) return '';

        let html = `<div class="timing-info">${items.map(item => `<span class="timing-item">${item}</span>`).join('')}</div>`;

        if (timings && timings.provider_results && timings.provider_results.length > 1) {
            html += `<div class="timing-chain">Provider chain: ${timings.provider_results.join(' \u2192 ')}</div>`;
        }

        return html;
    }
};
