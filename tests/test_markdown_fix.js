const contents = [
    "![马库斯](10727.jpg",
    "![马库斯]( 10727.jpg )",
    "![马库斯](10727)",
    "![马库斯](10727 ",
    "![马库斯](  10727  ",
    "![马库斯](10727.png 怎么永远少一个括号",
    "1. ![flower](flower1999.jpg) 相似度..."
];

function fixMarkdownImages(content) {
    // 匹配 ![alt_text](url 的情况，直到遇到右括号、空格或换行
    // 这里我们稍微宽泛一点匹配
    return content.replace(/!\[([^\]]*)\]\(([^)\n]+)(?:\))?/g, (match, alt, url_part) => {
        // 去除首尾空格
        let url = url_part.trim();
        
        // 如果url包含多余的字符比如后面跟着文本，我们可能需要截断，但通常情况大模型给出的是单纯的文件名
        // 比如 "10727.jpg 怎么永远少一个括号"
        // 这种情况可以通过只取第一个连续的非空格字符串作为url
        const spaceIndex = url.indexOf(' ');
        let extraText = '';
        if (spaceIndex !== -1) {
            extraText = url.substring(spaceIndex); // 把后面的文字保留下来
            url = url.substring(0, spaceIndex);
        }

        // 检查是否有合法的图片后缀
        const hasExtension = /\.(jpg|jpeg|png|gif|webp)$/i.test(url);
        if (!hasExtension) {
            // 默认补充 .png 或根据具体业务可以选
            url += '.jpg'; // 或者.png，先看看能不能匹配
        }
        
        return `![${alt}](${url})` + extraText;
    });
}

contents.forEach(c => {
    console.log("Original:", c);
    console.log("Fixed   :", fixMarkdownImages(c));
    console.log("---");
});
