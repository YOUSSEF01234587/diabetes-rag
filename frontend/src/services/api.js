const API_BASE = "http://127.0.0.1:8000";

class ApiService {
    constructor() {
        this.baseUrl = API_BASE;
    }

    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        if (!response.ok) throw new Error('Health check failed');
        return response.json();
    }

    async chat(message, options = {}) {
        const { topK = 8, evidenceK = 5, timeout = 120000 } = options;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
            const response = await fetch(`${this.baseUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    top_k: topK,
                    evidence_k: evidenceK
                }),
                signal: controller.signal
            });

            if (response.status === 429) {
                throw new ApiError('Rate limited', 429, 'The AI generation service is temporarily unavailable. Please try again shortly.');
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new ApiError(
                    errorData?.detail || 'Request failed',
                    response.status,
                    this.getErrorMessage(response.status)
                );
            }

            return response.json();
        } finally {
            clearTimeout(timeoutId);
        }
    }

    async search(query, topK = 8) {
        const response = await fetch(`${this.baseUrl}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: topK })
        });

        if (!response.ok) throw new ApiError('Search failed', response.status);
        return response.json();
    }

    async getSources() {
        const response = await fetch(`${this.baseUrl}/api/sources`);
        if (!response.ok) throw new ApiError('Failed to load sources', response.status);
        return response.json();
    }

    async getStats() {
        const response = await fetch(`${this.baseUrl}/api/stats`);
        if (!response.ok) throw new ApiError('Failed to load stats', response.status);
        return response.json();
    }

    getErrorMessage(status) {
        const messages = {
            400: 'Invalid request. Please check your question.',
            401: 'Authentication required.',
            403: 'Access denied.',
            404: 'Service not found.',
            429: 'The AI generation service is temporarily unavailable or rate-limited. Please try again shortly.',
            500: 'Server error. Please try again.',
            503: 'Service temporarily unavailable.'
        };
        return messages[status] || 'An unexpected error occurred.';
    }
}

class ApiError extends Error {
    constructor(message, status, userMessage) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.userMessage = userMessage || message;
    }
}

window.api = new ApiService();
