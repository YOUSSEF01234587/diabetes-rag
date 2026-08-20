const STORAGE_KEY = "dcc_recent_chats";
const MAX_CONVERSATIONS = 50;
const MAX_SIZE_BYTES = 4 * 1024 * 1024;

class ChatStorage {
    constructor() {
        this.conversations = [];
    }

    init() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) {
                this.conversations = JSON.parse(raw);
            }
        } catch {
            this.conversations = [];
        }
        this._cleanup();
    }

    generateId() {
        return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    }

    generateTitle(firstMessage) {
        if (!firstMessage || typeof firstMessage !== "string") return "New Chat";
        let text = firstMessage.trim();
        const sentenceEnd = text.search(/[.!?\n]/);
        if (sentenceEnd > 0) text = text.slice(0, sentenceEnd);
        if (text.length > 60) text = text.slice(0, 57) + "...";
        return text || "New Chat";
    }

    saveConversation(id, messages) {
        const now = Date.now();
        const existing = this.conversations.find((c) => c.id === id);

        const firstUserMsg = messages.find((m) => m.role === "user");
        const title = existing?.title || this.generateTitle(firstUserMsg?.text);

        if (existing) {
            existing.messages = messages;
            existing.updatedAt = now;
            existing.title = title;
        } else {
            this.conversations.push({
                id,
                title,
                createdAt: now,
                updatedAt: now,
                messages,
            });
        }

        this._cleanup();
        this._persist();
    }

    getConversations() {
        return [...this.conversations].sort(
            (a, b) => b.updatedAt - a.updatedAt
        );
    }

    getConversation(id) {
        return this.conversations.find((c) => c.id === id) || null;
    }

    deleteConversation(id) {
        this.conversations = this.conversations.filter((c) => c.id !== id);
        this._persist();
    }

    renameConversation(id, newTitle) {
        const conv = this.conversations.find((c) => c.id === id);
        if (!conv) return false;
        conv.title = newTitle || "New Chat";
        conv.updatedAt = Date.now();
        this._persist();
        return true;
    }

    clearAll() {
        this.conversations = [];
        this._persist();
    }

    _cleanup() {
        if (this.conversations.length <= MAX_CONVERSATIONS) return;
        this.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
        this.conversations = this.conversations.slice(0, MAX_CONVERSATIONS);
    }

    _persist() {
        try {
            let serialized = JSON.stringify(this.conversations);
            if (serialized.length > MAX_SIZE_BYTES) {
                this.conversations.sort(
                    (a, b) => b.updatedAt - a.updatedAt
                );
                while (
                    this.conversations.length > 1 &&
                    JSON.stringify(this.conversations).length > MAX_SIZE_BYTES
                ) {
                    this.conversations.pop();
                }
                serialized = JSON.stringify(this.conversations);
            }
            localStorage.setItem(STORAGE_KEY, serialized);
        } catch {
            // localStorage unavailable or full — silently ignore
        }
    }
}

window.chatStorage = new ChatStorage();
