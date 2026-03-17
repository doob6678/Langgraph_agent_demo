import re

class ImageMarkdownBuffer:
    """
    Buffer to ensure Markdown image syntax is emitted completely in one go.
    It intercepts chunks and buffers them if they look like the start of an image tag.
    Once a complete tag is found (or the buffer overflows/fails to match), it releases the content.
    """
    def __init__(self, max_buffer_size: int = 1000):
        self.buffer = ""
        self.max_buffer_size = max_buffer_size

    def process(self, chunk: str) -> str:
        if not chunk:
            return ""
            
        self.buffer += chunk
        output = ""
        
        while True:
            # 1. If buffer is empty, break
            if not self.buffer:
                break
                
            # 2. Find start of potential image tag '!['
            start_idx = self.buffer.find("![")
            
            # Edge case: buffer ends with '!', next chunk might be '['
            if start_idx == -1:
                if self.buffer.endswith("!"):
                    # Output everything before '!'
                    output += self.buffer[:-1]
                    self.buffer = "!" 
                    break
                else:
                    # No image start found, flush everything
                    output += self.buffer
                    self.buffer = ""
                    break
            
            # 3. Found '![' at start_idx
            # Output everything before '!['
            if start_idx > 0:
                output += self.buffer[:start_idx]
                self.buffer = self.buffer[start_idx:]
                # Now buffer starts with '!['
            
            # 4. Try to find closing structure
            # Format: ![alt](url)
            # We need to find '](', then the closing ')'
            
            # Find ']'
            close_bracket_idx = self.buffer.find("]")
            
            if close_bracket_idx == -1:
                # No ']' yet, keep buffering unless overflow
                if len(self.buffer) > self.max_buffer_size:
                    output += self.buffer
                    self.buffer = ""
                break 
            
            # Check if ']' is followed by '('
            if close_bracket_idx + 1 >= len(self.buffer):
                # Need more data to check for '('
                 if len(self.buffer) > self.max_buffer_size:
                    output += self.buffer
                    self.buffer = ""
                 break
            
            if self.buffer[close_bracket_idx + 1] != '(':
                # ']' not followed by '(', so not a standard markdown image.
                # Treat '![' as literal text.
                output += self.buffer[:2] # Output '!['
                self.buffer = self.buffer[2:] # Remove '!['
                continue
                
            # Found '](', now find closing ')'
            open_paren_idx = close_bracket_idx + 1
            close_paren_idx = self.buffer.find(")", open_paren_idx)
            
            if close_paren_idx == -1:
                 # No closing ')', keep buffering
                 if len(self.buffer) > self.max_buffer_size:
                    output += self.buffer
                    self.buffer = ""
                 break
            
            # 5. Found complete ![...](...)
            full_image_str = self.buffer[:close_paren_idx+1]
            output += full_image_str
            self.buffer = self.buffer[close_paren_idx+1:]
            
            # Loop continues to process remaining buffer
            
        return output

    def flush(self) -> str:
        """Flush remaining buffer at end of stream"""
        res = self.buffer
        self.buffer = ""
        return res
