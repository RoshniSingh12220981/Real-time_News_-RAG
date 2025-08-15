
import os
import time
import hashlib
import re
from dotenv import load_dotenv
from typing import Tuple, List, Dict
import logging
import google.generativeai as genai

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Successfully configured Gemini API for misinformation detection")
except Exception as e:
    logger.error(f"Error configuring Gemini API: {e}")
    model = None

# Simple in-memory cache for misinformation detection results
misinfo_cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

def get_cache_key(article_text: str) -> str:
    """Generate a cache key from article text"""
    return hashlib.md5(article_text.encode()).hexdigest()

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

def local_misinfo_detection_fallback(article_text: str) -> Tuple[str, str]:
    """
    Enhanced local misinformation detection using rule-based analysis
    Checks for common indicators of misinformation
    """
    article_lower = article_text.lower()
    
    # Enhanced misinformation indicators
    sensational_keywords = [
        "shocking", "unbelievable", "incredible", "amazing", "you won't believe",
        "doctors hate", "secret", "conspiracy", "they don't want you to know",
        "miracle", "breakthrough", "exclusive", "leaked", "insider reveals",
        "bombshell", "explosive", "stunning", "mind-blowing", "earth-shattering"
    ]
    
    clickbait_patterns = [
        r"\d+ .* that will .* you",
        r"you won't believe what happened",
        r"this .* will shock you",
        r"the .* they don't want you to see",
        r"number \d+ will amaze you",
        r"what happens next will .* you",
        r"the truth about .* revealed"
    ]
    
    unreliable_sources = [
        "facebook post", "tweet", "anonymous source", "unnamed official",
        "according to rumors", "it is said", "people are saying",
        "sources claim", "allegedly", "unconfirmed reports"
    ]
    
    bias_indicators = [
        "mainstream media", "fake news", "deep state", "they", "establishment",
        "wake up", "sheeple", "open your eyes", "truth", "exposed"
    ]
    
    emotional_manipulation = [
        "outraged", "furious", "devastated", "terrified", "panic",
        "crisis", "emergency", "urgent", "immediate", "now"
    ]
    
    # Count indicators
    sensational_count = sum(1 for keyword in sensational_keywords if keyword in article_lower)
    clickbait_matches = sum(1 for pattern in clickbait_patterns if re.search(pattern, article_lower))
    unreliable_mentions = sum(1 for source in unreliable_sources if source in article_lower)
    bias_count = sum(1 for indicator in bias_indicators if indicator in article_lower)
    emotional_count = sum(1 for word in emotional_manipulation if word in article_lower)
    
    # Check for excessive capitalization
    words = article_text.split()
    if len(words) > 0:
        caps_ratio = sum(1 for word in words if word.isupper() and len(word) > 2) / len(words)
    else:
        caps_ratio = 0
    
    # Check for excessive punctuation
    exclamation_count = article_text.count('!')
    question_count = article_text.count('?')
    total_sentences = max(1, article_text.count('.') + article_text.count('!') + article_text.count('?'))
    punct_ratio = (exclamation_count + question_count) / total_sentences
    
    # Check for lack of sources
    source_indicators = ["according to", "study shows", "research", "data", "statistics", "reported by"]
    has_sources = any(indicator in article_lower for indicator in source_indicators)
    
    # Scoring system
    risk_score = 0
    risk_factors = []
    
    if sensational_count > 2:
        risk_score += 3
        risk_factors.append(f"Multiple sensational keywords ({sensational_count})")
    
    if clickbait_matches > 0:
        risk_score += 4
        risk_factors.append(f"Clickbait patterns detected ({clickbait_matches})")
    
    if unreliable_mentions > 1:
        risk_score += 3
        risk_factors.append(f"References to unreliable sources ({unreliable_mentions})")
    
    if bias_count > 2:
        risk_score += 2
        risk_factors.append(f"Bias indicators present ({bias_count})")
    
    if emotional_count > 3:
        risk_score += 2
        risk_factors.append(f"Emotional manipulation language ({emotional_count})")
    
    if caps_ratio > 0.15:
        risk_score += 2
        risk_factors.append(f"Excessive capitalization ({caps_ratio:.2%})")
    
    if punct_ratio > 0.3:
        risk_score += 1
        risk_factors.append(f"Excessive punctuation ({punct_ratio:.2%})")
    
    if not has_sources and len(article_text) > 200:
        risk_score += 2
        risk_factors.append("Lack of credible source citations")
    
    # Determine verdict based on score
    if risk_score >= 7:
        verdict = "Likely Misinformation"
        explanation = f"Local analysis: High risk score ({risk_score}/15). Risk factors: {'; '.join(risk_factors)}"
    elif risk_score >= 4:
        verdict = "Potentially Misleading"
        explanation = f"Local analysis: Moderate risk score ({risk_score}/15). Risk factors: {'; '.join(risk_factors)}"
    else:
        verdict = "Likely Safe"
        if risk_factors:
            explanation = f"Local analysis: Low risk score ({risk_score}/15). Minor risk factors: {'; '.join(risk_factors)}"
        else:
            explanation = "Local analysis: No significant misinformation indicators detected"
    
    return (verdict, explanation)

def gemini_misinfo_detection(article_text: str) -> Tuple[str, str]:
    """Misinformation detection using Gemini API"""
    if model is None:
        return ("API_ERROR", "Gemini model not configured properly")
    
    # Truncate very long articles to stay within token limits
    max_chars = 8000  # Gemini can handle more text than GPT-3.5
    if len(article_text) > max_chars:
        article_text = article_text[:max_chars] + "..."
        logger.info("Article truncated for API call")
    
    prompt = f"""
You are an expert misinformation analyst. Analyze the following article for potential misinformation, disinformation, sensationalism, or misleading claims.

ARTICLE TO ANALYZE:
{article_text}

Please provide a comprehensive analysis following this format:

VERDICT: [Choose one: "LIKELY_SAFE", "POTENTIALLY_MISLEADING", "LIKELY_MISINFORMATION"]

CONFIDENCE: [Rate your confidence from 1-10, where 10 is completely certain]

RISK_FACTORS: [List specific elements that suggest misinformation, if any]
- Sensationalism or clickbait language
- Lack of credible sources
- Emotional manipulation
- Unsubstantiated claims
- Logical fallacies
- Bias indicators

POSITIVE_INDICATORS: [List elements that suggest credibility, if any]
- Credible sources cited
- Balanced reporting
- Factual language
- Verifiable claims

EXPLANATION: [Provide a detailed explanation of your analysis, mentioning specific examples from the text]

RECOMMENDATIONS: [Suggest what readers should do - verify claims, check sources, etc.]
"""

    def make_api_call():
        response = model.generate_content(prompt)
        return response.text.strip()
    
    try:
        answer = retry_with_backoff(make_api_call)
        
        # Parse the response
        lines = answer.split('\n')
        verdict_line = next((line for line in lines if line.startswith('VERDICT:')), '')
        confidence_line = next((line for line in lines if line.startswith('CONFIDENCE:')), '')
        risk_factors_section = []
        positive_section = []
        explanation_section = []
        recommendations_section = []
        
        current_section = None
        for line in lines:
            if line.startswith('RISK_FACTORS:'):
                current_section = 'risk'
            elif line.startswith('POSITIVE_INDICATORS:'):
                current_section = 'positive'
            elif line.startswith('EXPLANATION:'):
                current_section = 'explanation'
            elif line.startswith('RECOMMENDATIONS:'):
                current_section = 'recommendations'
            elif line.strip() and current_section:
                if current_section == 'risk':
                    risk_factors_section.append(line.strip())
                elif current_section == 'positive':
                    positive_section.append(line.strip())
                elif current_section == 'explanation':
                    explanation_section.append(line.strip())
                elif current_section == 'recommendations':
                    recommendations_section.append(line.strip())
        
        # Extract verdict
        if 'LIKELY_SAFE' in verdict_line.upper():
            verdict = "Likely Safe"
        elif 'POTENTIALLY_MISLEADING' in verdict_line.upper():
            verdict = "Potentially Misleading"
        elif 'LIKELY_MISINFORMATION' in verdict_line.upper():
            verdict = "Likely Misinformation"
        else:
            verdict = "Unknown"
        
        # Combine all analysis parts
        full_explanation = f"Gemini analysis: {' '.join(explanation_section)}"
        
        if confidence_line:
            full_explanation += f" | {confidence_line.strip()}"
        
        if risk_factors_section:
            full_explanation += f" | Risk factors: {'; '.join(risk_factors_section)}"
        
        if positive_section:
            full_explanation += f" | Positive indicators: {'; '.join(positive_section)}"
        
        if recommendations_section:
            full_explanation += f" | Recommendations: {'; '.join(recommendations_section)}"
        
        return (verdict, full_explanation)
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return ("API_ERROR", f"Gemini API error: {e}")

def enhanced_misinfo_detection(article_text: str) -> Tuple[str, str]:
    """
    Enhanced misinformation detection using both Gemini and fallback methods
    """
    try:
        # First try Gemini API
        verdict, explanation = gemini_misinfo_detection(article_text)
        
        # If API fails, use fallback
        if verdict == "API_ERROR":
            logger.warning(f"Gemini API failed: {explanation}")
            logger.info("Falling back to rule-based analysis")
            
            fallback_verdict, fallback_explanation = local_misinfo_detection_fallback(article_text)
            combined_explanation = f"{explanation} | Fallback: {fallback_explanation}"
            
            return (fallback_verdict, combined_explanation)
        
        return (verdict, explanation)
        
    except Exception as e:
        logger.error(f"Enhanced misinformation detection error: {e}")
        return local_misinfo_detection_fallback(article_text)

def detect_misinformation(article_text: str) -> Tuple[str, str]:
    """
    Main misinformation detection function with caching and fallbacks
    Returns: (verdict, explanation)
    """
    if not article_text or not article_text.strip():
        return ("Unknown", "Cannot analyze empty article text")
    
    # Check cache first
    cache_key = get_cache_key(article_text)
    if cache_key in misinfo_cache:
        cached_result, timestamp = misinfo_cache[cache_key]
        if is_cache_valid(timestamp):
            logger.info("Using cached misinformation detection result")
            return cached_result
        else:
            # Remove expired cache entry
            del misinfo_cache[cache_key]
    
    # Perform misinformation detection
    verdict, explanation = enhanced_misinfo_detection(article_text)
    
    # Cache successful results (avoid caching errors)
    if not verdict.endswith("ERROR") and verdict != "Unknown":
        misinfo_cache[cache_key] = ((verdict, explanation), time.time())
        logger.info("Cached misinformation detection result")
    
    return (verdict, explanation)

def batch_misinfo_detection(articles: List[str]) -> List[Tuple[str, str]]:
    """
    Batch misinformation detection for multiple articles
    """
    results = []
    for i, article in enumerate(articles):
        logger.info(f"Processing article {i+1}/{len(articles)}")
        result = detect_misinformation(article)
        results.append(result)
        
        # Rate limiting - Gemini free tier allows 15 requests/minute
        time.sleep(4)  # 4 seconds between requests to stay under limit
    
    return results

def analyze_article_metrics(article_text: str) -> Dict:
    """
    Provide detailed metrics about the article for analysis
    """
    words = article_text.split()
    sentences = re.split(r'[.!?]+', article_text)
    
    # Basic metrics
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_length = word_count / max(1, sentence_count)
    
    # Punctuation analysis
    exclamation_count = article_text.count('!')
    question_count = article_text.count('?')
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
    
    # Source analysis
    source_keywords = ["according to", "study", "research", "data", "reported", "source"]
    source_mentions = sum(1 for keyword in source_keywords if keyword in article_text.lower())
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "exclamation_marks": exclamation_count,
        "question_marks": question_count,
        "capitalized_words": caps_words,
        "caps_ratio": round(caps_words / max(1, word_count), 3),
        "source_mentions": source_mentions,
        "punct_intensity": round((exclamation_count + question_count) / max(1, sentence_count), 2)
    }

def comprehensive_analysis(article_text: str) -> Dict:
    """
    Provide comprehensive analysis including metrics and detection results
    """
    verdict, explanation = detect_misinformation(article_text)
    metrics = analyze_article_metrics(article_text)
    fallback_verdict, fallback_explanation = local_misinfo_detection_fallback(article_text)
    
    return {
        "article_preview": article_text[:200] + "..." if len(article_text) > 200 else article_text,
        "primary_analysis": {
            "verdict": verdict,
            "explanation": explanation
        },
        "rule_based_analysis": {
            "verdict": fallback_verdict,
            "explanation": fallback_explanation
        },
        "article_metrics": metrics,
        "timestamp": time.time(),
        "analysis_method": "Gemini API + Rule-based fallback"
    }

def get_api_info() -> Dict:
    """Get information about the API configuration"""
    return {
        "api_configured": model is not None,
        "model_name": "gemini-1.5-flash",
        "rate_limits": {
            "requests_per_minute": 15,
            "requests_per_day": 1500,
            "free_tier": True
        },
        "max_article_length": "8000 characters"
    }

def get_cache_stats() -> Dict:
    """Get cache statistics for monitoring"""
    total_entries = len(misinfo_cache)
    valid_entries = sum(1 for _, (_, timestamp) in misinfo_cache.items() 
                       if is_cache_valid(timestamp))
    
    return {
        "total_cached": total_entries,
        "valid_cached": valid_entries,
        "expired_cached": total_entries - valid_entries,
        "cache_hit_rate": "Not available"  # Would need request counting to calculate
    }

def clear_expired_cache():
    """Manually clear expired cache entries"""
    expired_keys = [key for key, (_, timestamp) in misinfo_cache.items() 
                   if not is_cache_valid(timestamp)]
    
    for key in expired_keys:
        del misinfo_cache[key]
    
    logger.info(f"Cleared {len(expired_keys)} expired misinformation cache entries")
    return len(expired_keys)

# Example usage and testing
if __name__ == "__main__":
    # Test articles
    test_articles = [
        # Safe article
        "Scientists at MIT published a peer-reviewed study in Nature showing that renewable energy costs have decreased by 30% over the past decade. The research analyzed data from 50 countries and found consistent trends across different technologies.",
        
        # Potentially misleading
        "SHOCKING! Doctors don't want you to know this ONE SIMPLE TRICK that will cure everything! Anonymous sources reveal the secret that Big Pharma has been hiding for years!!!",
        
        # Misinformation indicators
        "You won't believe what the mainstream media isn't telling you about vaccines! Wake up, sheeple! The deep state conspiracy is real and they're coming for your children. Share this before it gets deleted!"
    ]
    
    print("API Info:", get_api_info())
    print("\nTesting misinformation detector...")
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n{'='*60}")
        print(f"TEST ARTICLE {i}:")
        print(f"Preview: {article[:100]}...")
        
        verdict, explanation = detect_misinformation(article)
        print(f"\nVerdict: {verdict}")
        print(f"Explanation: {explanation}")
        
        # Also show metrics
        metrics = analyze_article_metrics(article)
        print(f"\nArticle Metrics: {metrics}")
    
    print(f"\nCache Stats: {get_cache_stats()}")