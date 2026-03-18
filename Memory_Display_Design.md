# Memory Display Design Document

## 1. Overview
The goal is to visualize the Agent's internal memory state (Short-term, Long-term, and Image memory) on the frontend interface. This enhances transparency and allows users to understand what the Agent "knows" and how it retrieves information.

## 2. Frontend Layout Design
We will add a **Memory Panel** to the right side of the chat interface.

### 2.1 Panel Structure
- **Title**: "Agent Memory State"
- **Toggle Button**: A button in the header to show/hide the memory panel.
- **Tabs/Sections**:
  1.  **Context (Short-term)**: Shows the immediate conversation context retrieved.
  2.  **Facts (Long-term)**: Shows extracted user facts (e.g., profession, preferences).
  3.  **Images**: A gallery of relevant images retrieved from memory.
- **Content Display**:
  - **Text Memory**: Displayed as cards or a list with timestamps (if available).
  - **Image Memory**: Thumbnail grid with hover-over descriptions.
  - **Keywords**: Highlight keywords that triggered the memory recall (if backend provides).

### 2.2 User Interaction
- **Real-time Update**: The panel updates automatically as the Agent processes the request and recalls memory.
- **Collapsible**: The panel can be collapsed to save screen space.

## 3. Data Flow Architecture

### 3.1 Backend (Python/LangGraph)
1.  **Memory Recall**: The `memory_recall_node` in `graph_new_real.py` retrieves memory and updates `state.memory_data`.
2.  **Stream Event**: The `stream_chat_graph` function in `graph_stream_impl.py` listens for the `memory_recall` node completion.
3.  **SSE Emission**: When `memory_recall` completes, the backend emits a Server-Sent Event (SSE) with a custom field `x_memory_event`.
    ```json
    {
      "x_memory_event": {
        "short_term": [...],
        "long_term": [...],
        "images": [...]
      }
    }
    ```

### 3.2 Frontend (JS/HTML)
1.  **Event Listener**: `script.js` listens for `x_memory_event` in the SSE stream.
2.  **State Management**: Stores the latest memory data in a global variable or state object.
3.  **Rendering**: A `renderMemoryPanel(data)` function updates the DOM based on the received data.

## 4. Implementation Steps

### 4.1 Backend
- **File**: `backend/agent/stream_ex/graph_stream_impl.py`
- **Action**:
  - Update import to use `backend.agent.graph_new_real`.
  - In `stream_chat_graph`, detect `on_chain_end` for `memory_recall` node.
  - Extract `memory_data` and yield an SSE event.

### 4.2 Frontend
- **File**: `frontend/index.html`
  - Add the HTML structure for the Memory Panel (hidden by default or as a sidebar).
  - Add CSS styles for the panel.
- **File**: `frontend/js/script.js`
  - Add logic to parse `x_memory_event`.
  - Implement `renderMemoryPanel` function.
  - Add UI controls to toggle the panel.

## 5. Keyword Handling & Simplification
- **Simplification**: The memory data from backend might be raw JSON. The frontend should format it into readable text (e.g., "User is a Java Architect" instead of `{"content": "...", "metadata": ...}`).
- **Keyword Handling**: The backend `MemoryRecallNode` can optionally return "triggered keywords". If not, the frontend can highlight words in the memory text that match the user's current input query (simple client-side matching).

## 6. CSS Styling (Preview)
```css
.memory-panel {
    position: fixed;
    right: 0;
    top: 0;
    width: 300px;
    height: 100vh;
    background: #f8f9fa;
    border-left: 1px solid #ddd;
    overflow-y: auto;
    padding: 1rem;
    transform: translateX(100%); /* Hidden by default */
    transition: transform 0.3s ease;
}
.memory-panel.open {
    transform: translateX(0);
}
.memory-card {
    background: white;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
```
