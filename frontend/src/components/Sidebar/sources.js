window.sidebarSourcesComponent = {
    async load() {
        const container = document.getElementById('sidebar-sources');
        if (!container) return;

        try {
            const data = await window.api.getSources();
            const sources = data.sources || [];
            if (sources.length === 0) {
                container.innerHTML = '<div class="sidebar-loading">No sources available.</div>';
                return;
            }

            container.innerHTML = sources.map(source => `
                <div class="source-mini">
                    <div class="source-mini-name">${helpers.sanitize(source.short_title || source.title || 'Untitled')}</div>
                    <div class="source-mini-org">${helpers.sanitize(source.organization || 'Unknown')} ${source.year ? `(${source.year})` : ''}</div>
                </div>
            `).join('');
        } catch (error) {
            container.innerHTML = '<div class="sidebar-loading">Unable to load sources.</div>';
        }
    }
};
