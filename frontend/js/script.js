const messagesDiv = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const useRag = document.getElementById('use-rag');
const useSearch = document.getElementById('use-search');
const imagePreviewContainer = document.getElementById('image-preview-container');
const imagePreview = document.getElementById('image-preview');
let selectedFile = null;

// Configure marked with custom renderer for images
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

marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true
});

// Helper to format text with images
function formatContent(content) {
    let formatted = content;
    
    // 2. Try to convert rag_image_search output to markdown images
    // Pattern: 1. filename.jpg (Similarity: 0.xxx) -> 1. ![filename.jpg](filename.jpg) (Similarity: 0.xxx)
    // The renderer will then prepend /assets/ to the image path
    formatted = formatted.replace(
        /(\d+\.\s+)([^(\n]+\.(?:jpg|jpeg|png|gif|webp))(\s+\(相似度: [\d.]+\))?/gi, 
        '$1![$2]($2)$3'
    );
    return formatted;
}

function addMessage(content, type, isHtml = false) {
    const div = document.createElement('div');
    div.className = `message ${type}-message`;
    if (isHtml) {
        div.innerHTML = content;
    } else {
        div.textContent = content;
    }
    
    // If it's a user message and has an image, append it
    if (type === 'user' && selectedFile && !content.includes('<img')) {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(selectedFile);
        img.style.maxHeight = '200px';
        img.style.display = 'block';
        img.style.marginTop = '10px';
        div.appendChild(img);
    }

    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return div;
}

function createTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant-message typing-indicator';
    div.textContent = '思考中';
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return div;
}

// Paste Handler
document.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (const item of items) {
        if (item.kind === 'file' && item.type.startsWith('image/')) {
            const blob = item.getAsFile();
            handleImageFile(blob);
            e.preventDefault();
        }
    }
});

function handleImageFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        imagePreviewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedFile = null;
    imagePreview.src = '';
    imagePreviewContainer.style.display = 'none';
    document.getElementById('image-input').value = ''; // Reset file input
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text && !selectedFile) return;

    // Add user message
    addMessage(text, 'user');
    
    // Disable input
    const submitBtn = chatForm.querySelector('button');
    submitBtn.disabled = true;
    
    // Add typing indicator
    const typingIndicator = createTypingIndicator();
    let currentAssistantMessageDiv = null;
    let currentContent = '';

    try {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('use_rag', useRag.checked);
        formData.append('use_search', useSearch.checked);
        formData.append('stream', 'true');
        
        if (selectedFile) {
            formData.append('image', selectedFile);
        }

        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        // Remove typing indicator
        messagesDiv.removeChild(typingIndicator);
        currentAssistantMessageDiv = addMessage('', 'assistant', true);
        
        // Create containers for thoughts and content
        const thoughtsContainer = document.createElement('div');
        thoughtsContainer.className = 'thoughts-container expanded';
        thoughtsContainer.style.display = 'none'; // Hide initially
        thoughtsContainer.innerHTML = `
            <div class="thoughts-header" onclick="this.parentElement.classList.toggle('expanded')">
                <span class="thoughts-status-text">思考与执行中...</span>
                <span class="thoughts-toggle-icon">▼</span>
            </div>
            <div class="thoughts-content"></div>
        `;
        currentAssistantMessageDiv.appendChild(thoughtsContainer);
        const thoughtsContent = thoughtsContainer.querySelector('.thoughts-content');
        const thoughtsStatusText = thoughtsContainer.querySelector('.thoughts-status-text');
        
        const contentContainer = document.createElement('div');
        contentContainer.className = 'markdown-content';
        currentAssistantMessageDiv.appendChild(contentContainer);

        // Clear input and image
        userInput.value = '';
        removeImage();

        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep the last incomplete line in the buffer
            
            for (const line of lines) {
                const trimmedLine = line.trim();
                if (trimmedLine.startsWith('data: ')) {
                    const dataStr = trimmedLine.slice(6);
                    if (dataStr === '[DONE]') continue;
                    
                    try {
                        const data = JSON.parse(dataStr);
                        
                        // Handle Tool Events
                        if (data.x_tool_event) {
                            thoughtsContainer.style.display = 'block';
                            const event = data.x_tool_event;
                            const toolName = event.tool;
                            
                            if (event.status === 'started') {
                                // If there is any streamed text, move it to thoughts
                                if (currentContent.trim() !== '') {
                                    const thoughtText = document.createElement('div');
                                    thoughtText.className = 'thought-text';
                                    thoughtText.textContent = currentContent;
                                    thoughtsContent.appendChild(thoughtText);
                                    
                                    // Reset current content
                                    currentContent = '';
                                    contentContainer.innerHTML = '';
                                }

                                const statusItem = document.createElement('div');
                                statusItem.className = 'status-item';
                                statusItem.setAttribute('data-tool', toolName);
                                
                                // Format parameters safely
                                let paramsStr = '';
                                try {
                                    if (event.input) {
                                        paramsStr = typeof event.input === 'string' ? event.input : JSON.stringify(event.input, null, 2);
                                        if (paramsStr.length > 100) paramsStr = paramsStr.substring(0, 100) + '...';
                                    }
                                } catch(e) {}

                                let paramsHtml = paramsStr ? `<div class="tool-params">参数: ${paramsStr}</div>` : '';

                                statusItem.innerHTML = `
                                    <div class="status-item-header">
                                        <span class="status-icon"></span> 
                                        <span>正在使用 ${toolName}...</span>
                                    </div>
                                    ${paramsHtml}
                                `;
                                thoughtsContent.appendChild(statusItem);
                            } else if (event.status === 'completed') {
                                // Find the last running tool with this name
                                const items = thoughtsContent.querySelectorAll(`[data-tool="${toolName}"]`);
                                let statusItem = null;
                                for (let i = items.length - 1; i >= 0; i--) {
                                    if (!items[i].classList.contains('completed')) {
                                        statusItem = items[i];
                                        break;
                                    }
                                }
                                
                                if (statusItem) {
                                    statusItem.classList.add('completed');
                                    statusItem.querySelector('.status-icon').classList.add('done');
                                    statusItem.querySelector('.status-item-header span:last-child').textContent = `已完成 ${toolName}`;
                                    
                                    // If there is a result preview, show it
                                    if (event.result_preview) {
                                        const resultDiv = document.createElement('div');
                                        resultDiv.className = 'tool-result';
                                        resultDiv.style.fontSize = '0.8em';
                                        resultDiv.style.marginLeft = '1.5em';
                                        resultDiv.style.color = '#4b5563';
                                        
                                        // Parse result to check for images
                                        let resultText = event.result_preview;
                                        
                                        // If result contains image pattern like "1. filename.jpg (相似度: 0.xxx)"
                                        if (toolName === 'rag_image_search') {
                                             const imgRegex = /(\d+\.\s+)([^(\n]+\.(?:jpg|jpeg|png|gif|webp))/gi;
                                             let match;
                                             const imagesContainer = document.createElement('div');
                                             imagesContainer.style.display = 'flex';
                                             imagesContainer.style.gap = '5px';
                                             imagesContainer.style.marginTop = '5px';
                                             imagesContainer.style.flexWrap = 'wrap';
                                             
                                             let hasImages = false;
                                             while ((match = imgRegex.exec(resultText)) !== null) {
                                                 const filename = match[2];
                                                 const img = document.createElement('img');
                                                 img.src = '/assets/' + filename;
                                                 img.style.height = '40px';
                                                 img.style.borderRadius = '3px';
                                                 img.style.border = '1px solid #ddd';
                                                 img.title = filename;
                                                 img.onclick = function() {
                                                     window.open(this.src, '_blank');
                                                 };
                                                 img.style.cursor = 'pointer';
                                                 imagesContainer.appendChild(img);
                                                 hasImages = true;
                                             }
                                             if (hasImages) {
                                                 resultDiv.appendChild(imagesContainer);
                                                 statusItem.appendChild(resultDiv);
                                             }
                                        } else {
                                             // For other tools, optionally show a short result snippet
                                             // resultDiv.textContent = resultText.length > 100 ? resultText.substring(0, 100) + '...' : resultText;
                                             // statusItem.appendChild(resultDiv);
                                        }
                                    }
                                }
                            }
                        }

                        if (data.object === 'chat.completion.chunk') {
                            if (data.choices && data.choices.length > 0 && data.choices[0].delta && data.choices[0].delta.content) {
                                currentContent += data.choices[0].delta.content;
                                // Apply formatting (images etc)
                                try {
                                    const formatted = formatContent(currentContent);
                                    contentContainer.innerHTML = marked.parse(formatted);
                                    
                                    contentContainer.querySelectorAll('pre code').forEach((block) => {
                                        hljs.highlightElement(block);
                                    });
                                } catch (renderError) {
                                    console.error('Markdown render error:', renderError);
                                    // Fallback to basic text if marked or hljs fails
                                    contentContainer.innerText = currentContent;
                                }
                            }
                        }
                    } catch (e) {
                        console.error('Error parsing JSON:', e);
                    }
                }
            }
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // After stream is done, collapse thoughts if they were used
        if (thoughtsContainer.style.display !== 'none') {
            thoughtsContainer.classList.remove('expanded');
            thoughtsStatusText.textContent = '已完成思考与执行';
        }
        
    } catch (error) {
        console.error('Error:', error);
        if (messagesDiv.contains(typingIndicator)) {
            messagesDiv.removeChild(typingIndicator);
        }
        addMessage('抱歉，发生了一些错误。', 'assistant');
    } finally {
        submitBtn.disabled = false;
        userInput.focus();
    }
});
