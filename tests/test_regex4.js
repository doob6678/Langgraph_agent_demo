const content1 = "![10725](10";
const content2 = "![10725](10725.jp";

function formatContent(content) {
    let formatted = content.replace(/!\[([^\]]+)\](?:\(([^)\n]*)(?:\))?|(?!\())/g, (match, alt, url_part) => {
        let url = url_part !== undefined ? url_part.trim() : alt.trim();
        if (!url) url = alt.trim();
        
        const spaceIndex = url.indexOf(' ');
        let extraText = '';
        if (spaceIndex !== -1) {
            extraText = url.substring(spaceIndex);
            url = url.substring(0, spaceIndex);
        }
        
        const hasExtension = /\.(jpg|jpeg|png|gif|webp)$/i.test(url);
        if (!hasExtension) {
            url += '.jpg';
        }
        
        return `![${alt}](${url})` + extraText;
    });
    
    return formatted;
}

console.log(formatContent(content1));
console.log(formatContent(content2));