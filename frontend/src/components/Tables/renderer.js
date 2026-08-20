window.tableRenderer = {
    renderInEvidence(text) {
        if (!text) return '';

        const lines = text.split('\n');
        let inTable = false;
        let tableLines = [];
        let result = [];
        let currentParagraph = [];

        for (const line of lines) {
            const trimmed = line.trim();

            if (this.isTableRow(trimmed)) {
                if (currentParagraph.length > 0) {
                    result.push(`<p>${helpers.sanitize(currentParagraph.join(' '))}</p>`);
                    currentParagraph = [];
                }
                inTable = true;
                tableLines.push(trimmed);
            } else if (inTable && trimmed === '') {
                if (tableLines.length > 0) {
                    result.push(this.renderTable(tableLines));
                    tableLines = [];
                }
                inTable = false;
            } else {
                if (inTable && tableLines.length > 0) {
                    result.push(this.renderTable(tableLines));
                    tableLines = [];
                    inTable = false;
                }
                currentParagraph.push(trimmed);
            }
        }

        if (tableLines.length > 0) {
            result.push(this.renderTable(tableLines));
        }
        if (currentParagraph.length > 0) {
            result.push(`<p>${helpers.sanitize(currentParagraph.join(' '))}</p>`);
        }

        return result.join('') || helpers.sanitize(text);
    },

    isTableRow(line) {
        if (!line) return false;
        if (line.includes('|') && (line.match(/\|/g) || []).length >= 2) return true;
        if (line.includes('\t') && (line.match(/\t/g) || []).length >= 2) return true;
        return false;
    },

    renderTable(lines) {
        if (lines.length === 0) return '';

        const firstLine = lines[0];
        const delimiter = firstLine.includes('|') ? '|' : '\t';

        const rows = lines.map(line =>
            line.split(delimiter)
                .map(cell => cell.trim())
                .filter(cell => cell !== '')
        ).filter(row => row.length > 0);

        if (rows.length === 0) return '';

        const dataRows = rows.filter(row =>
            !row.every(cell => /^[-:]+$/.test(cell))
        );

        if (dataRows.length === 0) return '';

        const headerRow = dataRows[0];
        const bodyRows = dataRows.slice(1);

        let html = '<div class="table-container"><span class="table-badge">Table evidence</span><table class="evidence-table">';

        html += '<thead><tr>';
        for (const cell of headerRow) {
            html += `<th>${helpers.sanitize(cell)}</th>`;
        }
        html += '</tr></thead>';

        html += '<tbody>';
        for (const row of bodyRows) {
            html += '<tr>';
            for (const cell of row) {
                html += `<td>${helpers.sanitize(cell)}</td>`;
            }
            for (let i = row.length; i < headerRow.length; i++) {
                html += '<td></td>';
            }
            html += '</tr>';
        }
        html += '</tbody></table></div>';

        return html;
    }
};
