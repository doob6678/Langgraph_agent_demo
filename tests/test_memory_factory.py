
import sys
import os
import asyncio
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agent.node_ex.memory_node import get_memory_manager, MemoryManagerFactory
from backend.agent.memory_ex.memory_manager import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_factory():
    logger.info("Starting MemoryManagerFactory test...")
    
    # Test 1: Get manager for tenant A
    tenant_a = "tenant_A"
    manager_a1 = get_memory_manager(tenant_a)
    manager_a2 = get_memory_manager(tenant_a)
    
    assert manager_a1 is manager_a2, "Factory should return the same instance for the same tenant"
    logger.info(f"Tenant A instances match: {manager_a1 is manager_a2}")
    
    # Test 2: Get manager for tenant B
    tenant_b = "tenant_B"
    manager_b = get_memory_manager(tenant_b)
    
    assert manager_a1 is not manager_b, "Factory should return different instances for different tenants"
    logger.info(f"Tenant A and B instances are different: {manager_a1 is not manager_b}")
    
    # Test 3: Verify embedding model sharing
    emb_model_a = manager_a1.embedding_model
    emb_model_b = manager_b.embedding_model
    
    assert emb_model_a is not None, "Embedding model should be initialized"
    assert emb_model_a is emb_model_b, "Embedding model should be shared across tenants"
    logger.info(f"Embedding models are shared: {emb_model_a is emb_model_b}")
    
    logger.info("All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_factory())
