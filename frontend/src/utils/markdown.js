window.markdown = {
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    normalizeMath(text) {
        if (!text) return '';
        let s = text;
        s = s.replace(/\$\$\\(?:ge|geq)\s+([\d.]+)\s*(mmol\/L|mg\/dL|%)?\$\$/g, (m, num, unit) => `\u2265${num}${unit ? ' ' + unit : ''}`);
        s = s.replace(/\$\\(?:ge|geq)\s+([\d.]+)\s*(mmol\/L|mg\/dL|%?)\$/g, (m, num, unit) => `\u2265${num}${unit ? ' ' + unit : ''}`);
        s = s.replace(/\\(?:ge|geq)\s+([\d.]+)\s*(mmol\/L|mg\/dL|%)/g, (m, num, unit) => `\u2265${num} ${unit}`);
        s = s.replace(/\$\$\\(?:le|leq)\s+([\d.]+)\s*(mmol\/L|mg\/dL|%)?\$\$/g, (m, num, unit) => `\u2264${num}${unit ? ' ' + unit : ''}`);
        s = s.replace(/\$\\(?:le|leq)\s+([\d.]+)\s*(mmol\/L|mg\/dL|%?)\$/g, (m, num, unit) => `\u2264${num}${unit ? ' ' + unit : ''}`);
        s = s.replace(/\\(?:le|leq)\s+([\d.]+)\s*(mmol\/L|mg\/dL|%)/g, (m, num, unit) => `\u2264${num} ${unit}`);
        s = s.replace(/\$\$\\pm\$\$/g, '\u00B1');
        s = s.replace(/\$\\pm\$/g, '\u00B1');
        s = s.replace(/\\pm/g, '\u00B1');
        s = s.replace(/\$\$\\times\$\$/g, '\u00D7');
        s = s.replace(/\$\\times\$/g, '\u00D7');
        s = s.replace(/\\times/g, '\u00D7');
        s = s.replace(/\$([^$]+)\$/g, (m, inner) => inner);
        return s;
    },

    render(text) {
        if (!text) return '';

        text = this.normalizeMath(text);

        let html = this.escapeHtml(text);

        html = html.replace(/\[Evidence\s+(\d+)\]/g, (match, num) => {
            return `<span class="citation-ref" role="button" tabindex="0" data-evidence="${num}" title="See Evidence ${num}">E${num}</span>`;
        });

        html = html.replace(/\[Evidence\s+(\d+(?:\s*(?:,|and)\s*\d+)*)\]/g, (match, nums) => {
            return nums.split(/(?:\s*(?:,|and)\s*)/).map(n =>
                `<span class="citation-ref" role="button" tabindex="0" data-evidence="${n.trim()}" title="See Evidence ${n.trim()}">E${n.trim()}</span>`
            ).join(' ');
        });

        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        html = html.replace(/^##\s+(.+)$/gm, '<div class="md-heading">$1</div>');

        const lines = html.split('\n');
        const processed = [];
        let i = 0;
        while (i < lines.length) {
            const line = lines[i];
            const bulletMatch = line.match(/^[-*]\s+(.+)/);
            if (bulletMatch) {
                const items = [];
                while (i < lines.length && lines[i].match(/^[-*]\s+(.+)/)) {
                    items.push(lines[i].replace(/^[-*]\s+/, ''));
                    i++;
                }
                processed.push('<ul class="md-list">' + items.map(item => `<li>${item}</li>`).join('') + '</ul>');
                continue;
            }

            const numMatch = line.match(/^\d+[.)]\s+(.+)/);
            if (numMatch) {
                const items = [];
                while (i < lines.length && lines[i].match(/^\d+[.)]\s+(.+)/)) {
                    items.push(lines[i].replace(/^\d+[.)]\s+/, ''));
                    i++;
                }
                processed.push('<ol class="md-list">' + items.map(item => `<li>${item}</li>`).join('') + '</ol>');
                continue;
            }

            processed.push(line);
            i++;
        }
        html = processed.join('\n');

        html = html.split('\n\n').map(block => {
            if (!block.trim()) return '';
            if (block.startsWith('<ul') || block.startsWith('<ol') || block.startsWith('<div class="md-heading')) {
                return block;
            }
            return `<p>${block.replace(/\n/g, '<br>')}</p>`;
        }).join('');

        html = html.replace(/([^>])\n([^<])/g, '$1<br>$2');

        return html;
    }
};
