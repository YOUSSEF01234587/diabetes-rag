// Simulate browser globals
global.window = {};
global.document = { createElement: () => ({ textContent: '', innerHTML: '' }) };
global.Set = Set;
global.Math = Math;
global.Date = Date;
global.console = console;

// Load each script in order (simulating browser)
const fs = require('fs');
const vm = require('vm');
const path = require('path');

const scripts = [
    'frontend/src/services/api.js',
    'frontend/src/utils/markdown.js',
    'frontend/src/utils/helpers.js',
    'frontend/src/components/Tables/renderer.js',
    'frontend/src/components/Safety/indicator.js',
    'frontend/src/components/Chat/message.js',
    'frontend/src/components/Chat/input.js',
];

const ctx = vm.createContext(global);
for (const script of scripts) {
    const fullPath = path.join('D:/diabetes-rag', script);
    const code = fs.readFileSync(fullPath, 'utf8');
    try {
        vm.runInContext(code, ctx, { filename: script });
        console.log('OK:', script);
    } catch(e) {
        console.log('FAIL:', script, '-', e.message);
        break;
    }
}

// Test if key objects exist
console.log('');
console.log('window.api:', typeof ctx.window.api);
console.log('window.helpers:', typeof ctx.window.helpers);
console.log('window.messageComponent:', typeof ctx.window.messageComponent);
console.log('window.inputComponent:', typeof ctx.window.inputComponent);
if (ctx.window.helpers) {
    console.log('helpers.formatTimestamp:', typeof ctx.window.helpers.formatTimestamp);
    console.log('helpers.sanitize:', typeof ctx.window.helpers.sanitize);
}
