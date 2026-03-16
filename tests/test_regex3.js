const content1 = "1. ![马库斯](10727.jpg)";
const content2 = "根据搜索结果，为你找到相关图片： \n\n ![10725](";
const content3 = "![马库斯](";
const content4 = "![10725]";
const content5 = "![马库斯](10727.jpg";

function formatContent(content) {
    let formatted = content.replace(/!\[([^\]]+)\](?:\(|(?!\())/g, (match, alt) => {
        return match;
    });

    formatted = content.replace(/!\[([^\]]+)\](?:\(([^)\n]*)(?:\))?|(?!\())/g, (match, alt, url_part) => {
        let url = url_part ? url_part.trim() : alt.trim();
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
    
    formatted = formatted.replace(
        /(\d+\.\s+)([^(\n]+\.(?:jpg|jpeg|png|gif|webp))(\s+\(相似度: [\d.]+\))?/gi, 
        '$1![$2]($2)$3'
    );
    return formatted;
}

console.log(formatContent(content1));
console.log(formatContent(content2));
console.log(formatContent(content3));
console.log(formatContent(content4));
console.log(formatContent(content5));