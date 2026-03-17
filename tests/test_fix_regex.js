const content1 = "![马库斯](10727.jpg";
const content2 = "![马库斯](10727.jpg)";
const content3 = "![马库斯](10727.jpg 怎么永远少一个括号";

function formatContent(content) {
    let formatted = content.replace(/(!\[[^\]]*\]\([^)\n]+\.(?:jpg|jpeg|png|gif|webp))(?!\))/gi, '$1)');
    return formatted;
}

console.log(formatContent(content1));
console.log(formatContent(content2));
console.log(formatContent(content3));
