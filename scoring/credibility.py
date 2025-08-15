# # Placeholder: In production, use a real API or database
# CREDIBILITY_SCORES = {
#     "CNN": 4/5,
#     "BBC": 5/5
# }

# def get_source_credibility(source):
#     if not source:
#         return 0.5
#     source_lower = source.lower()
#     for key, score in CREDIBILITY_SCORES.items():
#         if key.lower() in source_lower:
#             return score
#     # Add more sources as needed
#     if "reuters" in source_lower:
#         return 0.85
#     if "al jazeera" in source_lower:
#         return 0.7
#     if "fox" in source_lower:
#         return 0.4
#     if "nyt" in source_lower or "new york times" in source_lower:
#         return 0.8
#     return 0.5

import re
import json
import os
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive credibility scores based on media bias fact check and similar organizations
# Scale: 0.0 (completely unreliable) to 1.0 (completely reliable)
CREDIBILITY_SCORES = {
    # Tier 1: Highest credibility (0.9-1.0)
    "BBC": 0.95,
    "Reuters": 0.95,
    "Associated Press": 0.95,
    "AP News": 0.95,
    "NPR": 0.92,
    "PBS": 0.92,
    "The Guardian": 0.90,
    
    # Tier 2: High credibility (0.8-0.89)
    "CNN": 0.85,
    "New York Times": 0.88,
    "NYT": 0.88,
    "Washington Post": 0.87,
    "Wall Street Journal": 0.86,
    "WSJ": 0.86,
    "ABC News": 0.84,
    "CBS News": 0.84,
    "NBC News": 0.84,
    "The Times": 0.85,
    "Financial Times": 0.88,
    "Bloomberg": 0.87,
    "Economist": 0.89,
    
    # Tier 3: Good credibility (0.7-0.79)
    "Al Jazeera": 0.75,
    "USA Today": 0.72,
    "Time Magazine": 0.76,
    "Newsweek": 0.74,
    "The Atlantic": 0.78,
    "Politico": 0.76,
    "The Hill": 0.72,
    "Yahoo News": 0.70,
    
    # Tier 4: Moderate credibility (0.5-0.69)
    "Fox News": 0.58,
    "MSNBC": 0.62,
    "CNN Opinion": 0.55,
    "New York Post": 0.52,
    "Daily Mail": 0.45,
    "Huffington Post": 0.65,
    "BuzzFeed News": 0.68,
    "Vox": 0.67,
    
    # Tier 5: Low credibility (0.3-0.49)
    "Breitbart": 0.35,
    "InfoWars": 0.15,
    "RT": 0.25,
    "Sputnik": 0.25,
    "Daily Caller": 0.42,
    "Gateway Pundit": 0.20,
    
    # Social media and blogs (lower by default)
    "Facebook": 0.30,
    "Twitter": 0.35,
    "Reddit": 0.40,
    "Medium": 0.45,
    "Blog": 0.35,
}

# Domain-based credibility for common news websites
DOMAIN_CREDIBILITY = {
    "bbc.com": 0.95,
    "bbc.co.uk": 0.95,
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "npr.org": 0.92,
    "pbs.org": 0.92,
    "theguardian.com": 0.90,
    "cnn.com": 0.85,
    "nytimes.com": 0.88,
    "washingtonpost.com": 0.87,
    "wsj.com": 0.86,
    "abcnews.go.com": 0.84,
    "cbsnews.com": 0.84,
    "nbcnews.com": 0.84,
    "ft.com": 0.88,
    "bloomberg.com": 0.87,
    "economist.com": 0.89,
    "aljazeera.com": 0.75,
    "usatoday.com": 0.72,
    "time.com": 0.76,
    "newsweek.com": 0.74,
    "theatlantic.com": 0.78,
    "politico.com": 0.76,
    "thehill.com": 0.72,
    "news.yahoo.com": 0.70,
    "foxnews.com": 0.58,
    "msnbc.com": 0.62,
    "nypost.com": 0.52,
    "dailymail.co.uk": 0.45,
    "huffpost.com": 0.65,
    "buzzfeednews.com": 0.68,
    "vox.com": 0.67,
    "breitbart.com": 0.35,
    "infowars.com": 0.15,
    "rt.com": 0.25,
    "sputniknews.com": 0.25,
}

# Red flag patterns that indicate potentially unreliable sources
RED_FLAG_PATTERNS = [
    r"\.tk$",  # Free domains often used by fake news
    r"\.ml$",
    r"\.ga$",
    r"\.cf$",
    r"fake.*news",
    r"conspiracy",
    r"patriot.*news",
    r"freedom.*eagle",
    r"truth.*news",
    r"real.*news.*now",
    r"american.*news",
    r"conservative.*daily",
    r"liberal.*tears",
]

# Positive indicators for credibility
POSITIVE_INDICATORS = [
    "news",
    "times",
    "post",
    "herald",
    "tribune",
    "journal",
    "gazette",
    "press",
    "wire",
    "media",
]

class CredibilityAssessment:
    def __init__(self, score: float, confidence: float, factors: list, tier: str):
        self.score = score
        self.confidence = confidence
        self.factors = factors
        self.tier = tier
    
    def to_dict(self):
        return {
            "score": self.score,
            "confidence": self.confidence,
            "factors": self.factors,
            "tier": self.tier
        }

def extract_domain(source: str) -> Optional[str]:
    """Extract domain from various source formats"""
    if not source:
        return None
    
    # Handle URLs
    if "://" in source:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(source)
            return parsed.netloc.lower()
        except:
            pass
    
    # Handle domain-like strings
    source_clean = source.lower().strip()
    
    # Remove common prefixes
    prefixes = ["www.", "m.", "mobile.", "amp."]
    for prefix in prefixes:
        if source_clean.startswith(prefix):
            source_clean = source_clean[len(prefix):]
    
    return source_clean

def assess_by_red_flags(source: str) -> Tuple[float, list]:
    """Check for red flag patterns that indicate unreliable sources"""
    red_flags = []
    penalty = 0.0
    
    source_lower = source.lower()
    
    for pattern in RED_FLAG_PATTERNS:
        if re.search(pattern, source_lower):
            red_flags.append(f"Red flag pattern: {pattern}")
            penalty += 0.2
    
    # Additional checks
    if len(source.split('.')) > 3:  # Many subdomains can be suspicious
        red_flags.append("Multiple subdomains")
        penalty += 0.1
    
    if any(word in source_lower for word in ["fake", "hoax", "satire"]):
        red_flags.append("Contains suspicious keywords")
        penalty += 0.3
    
    return max(0.0, 1.0 - penalty), red_flags

def assess_by_positive_indicators(source: str) -> Tuple[float, list]:
    """Check for positive indicators of credibility"""
    indicators = []
    bonus = 0.0
    
    source_lower = source.lower()
    
    for indicator in POSITIVE_INDICATORS:
        if indicator in source_lower:
            indicators.append(f"Contains '{indicator}'")
            bonus += 0.05
    
    # Check for established TLDs
    if source_lower.endswith(('.com', '.org', '.net', '.edu', '.gov')):
        indicators.append("Established TLD")
        bonus += 0.1
    
    return min(1.0, bonus), indicators

def get_credibility_tier(score: float) -> str:
    """Categorize credibility score into tiers"""
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.8:
        return "High"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.5:
        return "Moderate"
    elif score >= 0.3:
        return "Low"
    else:
        return "Very Low"

def get_source_credibility(source: str, detailed: bool = False) -> float | CredibilityAssessment:
    """
    Enhanced credibility assessment with multiple scoring methods
    
    Args:
        source: Source name, URL, or identifier
        detailed: If True, return detailed CredibilityAssessment object
    
    Returns:
        float: Credibility score (0.0-1.0) if detailed=False
        CredibilityAssessment: Detailed assessment if detailed=True
    """
    if not source or not source.strip():
        base_score = 0.5
        if detailed:
            return CredibilityAssessment(
                score=base_score,
                confidence=0.1,
                factors=["No source provided"],
                tier=get_credibility_tier(base_score)
            )
        return base_score
    
    source_clean = source.strip()
    source_lower = source_clean.lower()
    factors = []
    confidence = 0.5
    base_score = 0.5
    
    # Method 1: Direct name matching
    name_match_score = None
    for key, score in CREDIBILITY_SCORES.items():
        if key.lower() in source_lower:
            name_match_score = score
            factors.append(f"Direct match: {key}")
            confidence = 0.9
            break
    
    # Method 2: Domain matching
    domain = extract_domain(source_clean)
    domain_match_score = None
    if domain and domain in DOMAIN_CREDIBILITY:
        domain_match_score = DOMAIN_CREDIBILITY[domain]
        factors.append(f"Domain match: {domain}")
        confidence = max(confidence, 0.8)
    
    # Method 3: Red flag assessment
    red_flag_score, red_flags = assess_by_red_flags(source_clean)
    factors.extend(red_flags)
    
    # Method 4: Positive indicator assessment
    positive_score, positive_indicators = assess_by_positive_indicators(source_clean)
    factors.extend(positive_indicators)
    
    # Combine scores with weighted average
    scores = []
    weights = []
    
    if name_match_score is not None:
        scores.append(name_match_score)
        weights.append(0.6)  # Highest weight for direct matches
    
    if domain_match_score is not None:
        scores.append(domain_match_score)
        weights.append(0.4)
    
    # Always include red flag and positive assessments
    scores.extend([red_flag_score, base_score + positive_score])
    weights.extend([0.3, 0.2])
    
    # Calculate weighted average
    if scores and weights:
        final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        final_score = max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
    else:
        final_score = base_score
    
    # Adjust confidence based on number of factors
    if len(factors) >= 3:
        confidence = min(1.0, confidence + 0.1)
    elif len(factors) == 0:
        confidence = max(0.1, confidence - 0.2)
        factors.append("No specific credibility indicators found")
    
    # Log the assessment for debugging
    logger.debug(f"Credibility assessment for '{source}': {final_score:.2f} (confidence: {confidence:.2f})")
    
    if detailed:
        return CredibilityAssessment(
            score=round(final_score, 3),
            confidence=round(confidence, 3),
            factors=factors,
            tier=get_credibility_tier(final_score)
        )
    
    return round(final_score, 3)

def update_credibility_scores(new_scores: Dict[str, float], save_to_file: bool = False):
    """
    Update credibility scores with new data
    
    Args:
        new_scores: Dictionary of source names and their credibility scores
        save_to_file: Whether to save updated scores to a file
    """
    global CREDIBILITY_SCORES
    
    for source, score in new_scores.items():
        if 0.0 <= score <= 1.0:
            CREDIBILITY_SCORES[source] = score
            logger.info(f"Updated credibility score for {source}: {score}")
        else:
            logger.warning(f"Invalid credibility score for {source}: {score} (must be 0.0-1.0)")
    
    if save_to_file:
        try:
            with open("credibility_scores.json", "w") as f:
                json.dump(CREDIBILITY_SCORES, f, indent=2)
            logger.info("Credibility scores saved to file")
        except Exception as e:
            logger.error(f"Failed to save credibility scores: {e}")

def load_credibility_scores(file_path: str = "credibility_scores.json"):
    """Load credibility scores from a file"""
    global CREDIBILITY_SCORES
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                loaded_scores = json.load(f)
                CREDIBILITY_SCORES.update(loaded_scores)
                logger.info(f"Loaded {len(loaded_scores)} credibility scores from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load credibility scores from {file_path}: {e}")

def get_credibility_stats() -> dict:
    """Get statistics about the credibility scoring system"""
    scores = list(CREDIBILITY_SCORES.values())
    return {
        "total_sources": len(CREDIBILITY_SCORES),
        "average_score": sum(scores) / len(scores) if scores else 0,
        "high_credibility_sources": sum(1 for s in scores if s >= 0.8),
        "low_credibility_sources": sum(1 for s in scores if s < 0.5),
        "score_distribution": {
            "excellent": sum(1 for s in scores if s >= 0.9),
            "high": sum(1 for s in scores if 0.8 <= s < 0.9),
            "good": sum(1 for s in scores if 0.7 <= s < 0.8),
            "moderate": sum(1 for s in scores if 0.5 <= s < 0.7),
            "low": sum(1 for s in scores if s < 0.5),
        }
    }

# Initialize by loading any existing credibility scores
load_credibility_scores()