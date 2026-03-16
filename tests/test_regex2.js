const content1 = "1. ![马库斯](10727.jpg)";
const content2 = "根据搜索结果，为你找到相关图片： \n\n ![马库斯](10727.jpg)";
const content3 = "![马库斯](10727.jpg)";
const content4 = "根据搜索结果，为你找到相关图片： \n\n 1. ![马库斯](10727.jpg)";

function formatContent(content) {
    let formatted = content.replace(
        /(\d+\.\s+)([^(\n]+\.(?:jpg|jpeg|png|gif|webp))(\s+\(相似度: [\d.]+\))?/gi, 
        '$1![$2]($2)$3'
    );
    return formatted;
}

console.log("1:", formatContent(content1));
console.log("2:", formatContent(content2));
console.log("3:", formatContent(content3));
console.log("4:", formatContent(content4));
