from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id = "damo/nlp_corom_sentence-embedding_chinese-base"
try:
    pipe = pipeline(Tasks.sentence_embedding, model=model_id)
    text = "你好"
    
    logger.info("--- Test 1: input=text ---")
    try:
        res = pipe(input=text)
        logger.info(f"Result type: {type(res)}")
        logger.info(f"Result: {res}")
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")

    logger.info("--- Test 2: input={'source_sentence': [text]} ---")
    try:
        res = pipe(input={'source_sentence': [text]})
        logger.info(f"Result type: {type(res)}")
        logger.info(f"Result: {res}")
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")

    logger.info("--- Test 3: input=[text] ---")
    try:
        res = pipe(input=[text])
        logger.info(f"Result type: {type(res)}")
        logger.info(f"Result: {res}")
    except Exception as e:
        logger.error(f"Test 3 failed: {e}")

except Exception as e:
    logger.error(f"Pipeline init failed: {e}")
