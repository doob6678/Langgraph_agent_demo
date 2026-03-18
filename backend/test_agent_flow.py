import asyncio
import aiohttp
import time
import sys
import json
import base64
from PIL import Image
import io

# Configuration
BASE_URL = "http://127.0.0.1:8002"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"

async def test_permission_flow():
    print("=== Starting Agent Permission Flow Smoke Test ===")
    
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color='blue')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_bytes = buf.getvalue()
    
    # 1. User A (Dept 1) uploads a PRIVATE image
    print("\n[Step 1] User A uploads PRIVATE image...")
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('text', 'Analyze this blue image')
        data.add_field('use_rag', 'false') # Don't search yet
        data.add_field('use_search', 'false')
        data.add_field('user_id', 'user_A')
        data.add_field('dept_id', 'dept_1')
        data.add_field('visibility', 'private')
        data.add_field('image', img_bytes, filename='blue_private.jpg', content_type='image/jpeg')
        
        async with session.post(CHAT_ENDPOINT, data=data) as resp:
            if resp.status != 200:
                print(f"Error: {await resp.text()}")
                return
            res_json = await resp.json()
            print(f"User A Upload Response: {res_json.get('response')[:100]}...")
            
    # 2. User A (Dept 1) uploads a DEPARTMENT image
    print("\n[Step 2] User A uploads DEPARTMENT image...")
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('text', 'Analyze this shared blue image')
        data.add_field('use_rag', 'false')
        data.add_field('use_search', 'false')
        data.add_field('user_id', 'user_A')
        data.add_field('dept_id', 'dept_1')
        data.add_field('visibility', 'department')
        data.add_field('image', img_bytes, filename='blue_shared.jpg', content_type='image/jpeg')
        
        async with session.post(CHAT_ENDPOINT, data=data) as resp:
            if resp.status != 200:
                print(f"Error: {await resp.text()}")
                return
            res_json = await resp.json()
            print(f"User A Upload Response: {res_json.get('response')[:100]}...")

    # Wait for indexing (mock or real)
    print("Waiting for indexing...")
    await asyncio.sleep(2)

    # 3. User A searches for "blue" -> Should find BOTH
    print("\n[Step 3] User A searches for 'blue'...")
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('text', 'find blue image')
        data.add_field('use_rag', 'true')
        data.add_field('use_search', 'false')
        data.add_field('user_id', 'user_A')
        data.add_field('dept_id', 'dept_1')
        data.add_field('stream', 'false')
        
        async with session.post(CHAT_ENDPOINT, data=data) as resp:
            res_json = await resp.json()
            images = res_json.get('images', [])
            print(f"User A found {len(images)} images.")
            filenames = [img.get('filename') for img in images]
            print(f"Filenames: {filenames}")
            
            if not any('blue_private' in f for f in filenames):
                print("FAIL: User A did not find private image.")
            else:
                print("PASS: User A found private image.")
                
            if not any('blue_shared' in f for f in filenames):
                print("FAIL: User A did not find shared image.")
            else:
                print("PASS: User A found shared image.")

    # 4. User B (Dept 1) searches for "blue" -> Should find SHARED only
    print("\n[Step 4] User B (Same Dept) searches for 'blue'...")
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('text', 'find blue image')
        data.add_field('use_rag', 'true')
        data.add_field('use_search', 'false')
        data.add_field('user_id', 'user_B')
        data.add_field('dept_id', 'dept_1') # Same Dept
        data.add_field('stream', 'false')
        
        async with session.post(CHAT_ENDPOINT, data=data) as resp:
            res_json = await resp.json()
            images = res_json.get('images', [])
            print(f"User B found {len(images)} images.")
            filenames = [img.get('filename') for img in images]
            print(f"Filenames: {filenames}")
            
            if any('blue_private' in f for f in filenames):
                print("FAIL: User B found User A's private image!")
            else:
                print("PASS: User B did not find private image.")
                
            if not any('blue_shared' in f for f in filenames):
                print("FAIL: User B did not find shared image.")
            else:
                print("PASS: User B found shared image.")

    # 5. User C (Dept 2) searches for "blue" -> Should find NOTHING
    print("\n[Step 5] User C (Diff Dept) searches for 'blue'...")
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('text', 'find blue image')
        data.add_field('use_rag', 'true')
        data.add_field('use_search', 'false')
        data.add_field('user_id', 'user_C')
        data.add_field('dept_id', 'dept_2') # Diff Dept
        data.add_field('stream', 'false')
        
        async with session.post(CHAT_ENDPOINT, data=data) as resp:
            res_json = await resp.json()
            images = res_json.get('images', [])
            print(f"User C found {len(images)} images.")
            filenames = [img.get('filename') for img in images]
            print(f"Filenames: {filenames}")
            
            if len(images) > 0:
                print("FAIL: User C found images (should be 0).")
            else:
                print("PASS: User C found no images.")

if __name__ == "__main__":
    asyncio.run(test_permission_flow())
