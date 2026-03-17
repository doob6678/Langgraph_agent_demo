from backend.agent.stream_ex.image_buffer import ImageMarkdownBuffer

def test_buffer():
    print("=== Test Case 1: Split markdown image ===")
    buffer = ImageMarkdownBuffer()
    chunks = ["Here is an image: ", "![", "alt", "](", "url.png", ")", " and some text."]
    # Expected: 
    # "Here is an image: " -> output
    # "![" -> buffered
    # "alt" -> buffered
    # "](" -> buffered
    # "url.png" -> buffered
    # ")" -> output "![alt](url.png)"
    # " and some text." -> output
    
    for chunk in chunks:
        res = buffer.process(chunk)
        print(f"Chunk: '{chunk}' -> Output: '{res}' (Buffer: '{buffer.buffer}')")
    
    remaining = buffer.flush()
    if remaining:
        print(f"Remaining: '{remaining}'")

    print("\n=== Test Case 2: '!' at end of chunk ===")
    buffer = ImageMarkdownBuffer()
    chunks = ["Wait for it... !", "[", "image", "](", "pic.png", ")"]
    for chunk in chunks:
        res = buffer.process(chunk)
        print(f"Chunk: '{chunk}' -> Output: '{res}' (Buffer: '{buffer.buffer}')")

    print("\n=== Test Case 3: Broken/Incomplete image at end ===")
    buffer = ImageMarkdownBuffer()
    chunks = ["Some text ", "![", "incomplete"]
    for chunk in chunks:
        res = buffer.process(chunk)
        print(f"Chunk: '{chunk}' -> Output: '{res}' (Buffer: '{buffer.buffer}')")
    remaining = buffer.flush()
    print(f"Remaining: '{remaining}'")
    
    print("\n=== Test Case 4: False alarm (not an image) ===")
    buffer = ImageMarkdownBuffer()
    chunks = ["This is not an image: ![", " but just text with brackets]"]
    # The buffer logic checks for '](', if not found it keeps buffering until max_size or flush.
    # Wait, my logic:
    # If ']' found, check next char.
    # If next char is not '(', treat '![' as text.
    
    for chunk in chunks:
        res = buffer.process(chunk)
        print(f"Chunk: '{chunk}' -> Output: '{res}' (Buffer: '{buffer.buffer}')")
    remaining = buffer.flush()
    print(f"Remaining: '{remaining}'")

if __name__ == "__main__":
    test_buffer()
