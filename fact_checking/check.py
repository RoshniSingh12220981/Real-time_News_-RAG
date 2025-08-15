
import os
import time
import hashlib
import json
from typing import Tuple, Optional
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Successfully configured Gemini API")
except Exception as e:
    logger.error(f"Error configuring Gemini API: {e}")
    model = None

# Simple in-memory cache for fact-checking results
fact_check_cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

def get_cache_key(query: str, context: str) -> str:
    """Generate a cache key from query and context"""
    combined = f"{query}|{context}"
    return hashlib.md5(combined.encode()).hexdigest()

def is_cache_valid(timestamp: float) -> bool:
    """Check if cached result is still valid"""
    return time.time() - timestamp < CACHE_EXPIRY

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying in {delay} seconds... (attempt {attempt + 1})")
                time.sleep(delay)
            else:
                raise e

def local_fact_check_fallback(query: str, context: str) -> Tuple[str, str]:
    """Simple local fact-checking fallback using keyword matching"""
    query_lower = query.lower()
    context_lower = context.lower()
    
    # Extract key terms from the query
    query_words = set(query_lower.split())
    context_words = set(context_lower.split())
    
    # Calculate overlap
    overlap = len(query_words.intersection(context_words))
    total_query_words = len(query_words)
    
    if total_query_words == 0:
        return ("Unverified", "Cannot analyze empty query with local fallback")
    
    overlap_ratio = overlap / total_query_words
    
    # Enhanced keyword analysis
    contradiction_words = {"not", "no", "false", "incorrect", "wrong", "untrue", "denies", "refutes"}
    support_words = {"confirms", "supports", "proves", "shows", "demonstrates", "validates"}
    
    context_words_list = context_lower.split()
    has_contradiction = any(word in context_words_list for word in contradiction_words)
    has_support = any(word in context_words_list for word in support_words)
    
    if overlap_ratio > 0.7:
        if has_contradiction:
            explanation = f"Local analysis: High keyword overlap ({overlap_ratio:.2f}) but contradiction indicators found"
            return ("Conflicting Information", explanation)
        elif has_support:
            explanation = f"Local analysis: High keyword overlap ({overlap_ratio:.2f}) with support indicators"
            return ("Partially Verified", explanation)
        else:
            explanation = f"Local analysis: High keyword overlap ({overlap_ratio:.2f}) suggests claim may be supported"
            return ("Partially Verified", explanation)
    elif overlap_ratio > 0.3:
        explanation = f"Local analysis: Moderate keyword overlap ({overlap_ratio:.2f}) suggests partial relevance"
        return ("Partially Verified", explanation)
    else:
        explanation = f"Local analysis: Low keyword overlap ({overlap_ratio:.2f}) suggests claim may not be supported"
        return ("Unverified", explanation)

def gemini_fact_check(query: str, context: str) -> Tuple[str, str]:
    """Fact-checking using Gemini API"""
    if model is None:
        return ("API_ERROR", "Gemini model not configured properly")
    
    prompt = f"""
You are an expert fact-checker. Analyze the following claim against the provided context.

CLAIM TO VERIFY: {query}

CONTEXT/EVIDENCE: {context}

Please provide a fact-check analysis following this format:

VERDICT: [Choose one: "SUPPORTED", "REFUTED", "PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"]

EXPLANATION: [Provide a clear, concise explanation of your reasoning. Mention specific evidence from the context that supports or contradicts the claim. Be objective and precise.]

CONFIDENCE: [Rate your confidence from 1-10, where 10 is completely certain]

REASONING: [Brief explanation of the logical steps you took to reach this conclusion]
"""

    def make_api_call():
        response = model.generate_content(prompt)
        return response.text.strip()
    
    try:
        answer = retry_with_backoff(make_api_call)
        
        # Parse the response
        lines = answer.split('\n')
        verdict_line = next((line for line in lines if line.startswith('VERDICT:')), '')
        explanation_line = next((line for line in lines if line.startswith('EXPLANATION:')), '')
        confidence_line = next((line for line in lines if line.startswith('CONFIDENCE:')), '')
        reasoning_line = next((line for line in lines if line.startswith('REASONING:')), '')
        
        # Extract verdict
        if 'SUPPORTED' in verdict_line.upper():
            if 'PARTIALLY' in verdict_line.upper():
                verdict = "Partially Verified"
            else:
                verdict = "Likely True"
        elif 'REFUTED' in verdict_line.upper():
            verdict = "Likely False"
        elif 'INSUFFICIENT' in verdict_line.upper():
            verdict = "Unverified"
        else:
            verdict = "Unverified"
        
        # Combine explanation parts
        full_explanation = f"Gemini analysis: {explanation_line.replace('EXPLANATION:', '').strip()}"
        
        if confidence_line:
            full_explanation += f" | {confidence_line.strip()}"
        
        if reasoning_line:
            full_explanation += f" | {reasoning_line.strip()}"
        
        return (verdict, full_explanation)
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return ("API_ERROR", f"Gemini API error: {e}")

def enhanced_fact_check(query: str, context: str) -> Tuple[str, str]:
    """
    Enhanced fact-checking using both Gemini and fallback methods
    """
    try:
        # First try Gemini API
        verdict, explanation = gemini_fact_check(query, context)
        
        # If API fails, use fallback
        if verdict == "API_ERROR":
            logger.warning(f"Gemini API failed: {explanation}")
            logger.info("Falling back to keyword-based analysis")
            
            fallback_verdict, fallback_explanation = local_fact_check_fallback(query, context)
            combined_explanation = f"{explanation} | Fallback: {fallback_explanation}"
            
            return (fallback_verdict, combined_explanation)
        
        return (verdict, explanation)
        
    except Exception as e:
        logger.error(f"Enhanced fact-check error: {e}")
        return local_fact_check_fallback(query, context)

def fact_check(query: str, context: str) -> Tuple[str, str]:
    """
    Main fact-checking function with caching and fallbacks
    Returns: (verdict, explanation)
    """
    # Validate inputs
    if not query or not context:
        return ("ERROR", "Both query and context are required")
    
    if not query.strip() or not context.strip():
        return ("ERROR", "Query and context cannot be empty")
    
    # Check cache first
    cache_key = get_cache_key(query, context)
    if cache_key in fact_check_cache:
        cached_result, timestamp = fact_check_cache[cache_key]
        if is_cache_valid(timestamp):
            logger.info("Using cached fact-check result")
            return cached_result
        else:
            # Remove expired cache entry
            del fact_check_cache[cache_key]
    
    # Perform fact-checking
    verdict, explanation = enhanced_fact_check(query, context)
    
    # Cache successful results (avoid caching errors)
    if not verdict.endswith("ERROR"):
        fact_check_cache[cache_key] = ((verdict, explanation), time.time())
        logger.info("Cached fact-check result")
    
    return (verdict, explanation)

def batch_fact_check(queries: list, contexts: list) -> list:
    """
    Batch fact-checking for multiple queries
    """
    if len(queries) != len(contexts):
        raise ValueError("Number of queries must match number of contexts")
    
    results = []
    for i, (query, context) in enumerate(zip(queries, contexts)):
        logger.info(f"Processing batch item {i+1}/{len(queries)}")
        result = fact_check(query, context)
        results.append(result)
        
        # Rate limiting - Gemini free tier allows 15 requests/minute
        time.sleep(4)  # 4 seconds between requests to stay under limit
    
    return results

def detailed_analysis(query: str, context: str) -> dict:
    """
    Provide detailed analysis including multiple perspectives
    """
    verdict, explanation = fact_check(query, context)
    
    # Also get fallback analysis for comparison
    fallback_verdict, fallback_explanation = local_fact_check_fallback(query, context)
    
    return {
        "query": query,
        "context": context[:200] + "..." if len(context) > 200 else context,
        "primary_analysis": {
            "verdict": verdict,
            "explanation": explanation
        },
        "keyword_analysis": {
            "verdict": fallback_verdict,
            "explanation": fallback_explanation
        },
        "timestamp": time.time()
    }

def get_api_info() -> dict:
    """Get information about the API configuration"""
    return {
        "api_configured": model is not None,
        "model_name": "gemini-1.5-flash",
        "rate_limits": {
            "requests_per_minute": 15,
            "requests_per_day": 1500,
            "free_tier": True
        }
    }

def get_cache_stats() -> dict:
    """Get cache statistics for monitoring"""
    total_entries = len(fact_check_cache)
    valid_entries = sum(1 for _, (_, timestamp) in fact_check_cache.items() 
                       if is_cache_valid(timestamp))
    
    return {
        "total_cached": total_entries,
        "valid_cached": valid_entries,
        "expired_cached": total_entries - valid_entries,
        "cache_hit_rate": "Not available"  # Would need request counting to calculate
    }

def clear_expired_cache():
    """Manually clear expired cache entries"""
    expired_keys = [key for key, (_, timestamp) in fact_check_cache.items() 
                   if not is_cache_valid(timestamp)]
    
    for key in expired_keys:
        del fact_check_cache[key]
    
    logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    return len(expired_keys)

# Example usage and testing
if __name__ == "__main__":
    # Test the fact checker
    test_cases = [
        ("The Earth is round", "Scientific evidence and satellite imagery confirm that Earth has a spherical shape."),
        ("Water boils at 50°C", "Water boils at 100°C (212°F) at standard atmospheric pressure."),
        ("Python is a programming language", "Python is a popular programming language used for web development, data science, and automation.")
    ]
    
    print("API Info:", get_api_info())
    print("\nTesting fact checker...")
    
    for query, context in test_cases:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"Context: {context}")
        
        verdict, explanation = fact_check(query, context)
        print(f"Verdict: {verdict}")
        print(f"Explanation: {explanation}")
    
    print(f"\nCache Stats: {get_cache_stats()}")