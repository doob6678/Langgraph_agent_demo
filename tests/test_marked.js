const { marked } = require('./frontend/lib/marked.min.js');

marked.use({
    renderer: {
        image(href, title, text) {
            if (href && !href.startsWith('http') && !href.startsWith('/') && !href.startsWith('data:')) {
                href = '/assets/' + href;
            }
            return `<img src="${href}" alt="${text}" title="${title || ''}" class="markdown-image">`;
        }
    }
});

console.log(marked.parse("![马库斯](10727.jpg)"));
console.log(marked.parse("![马库斯](10727.jpg"));
console.log(marked.parse("![马库斯]( 10727.jpg )"));
