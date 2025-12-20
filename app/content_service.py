# content_service.py
"""
Content Services powered by Google Gemini API.
Provides:
- Similarity checking
- Social media caption generation
- Text summarization

Requires GEMINI_API_KEY in .env file.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
# Look for .env in the project root (parent of app directory)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("[Warning] GEMINI_API_KEY not found in environment. Content services will not work.")

# Content store for similarity checking
CONTENT_STORE_FILE = Path(__file__).resolve().parent / "content_store.json"


def get_gemini_client():
    """Get initialized Gemini client."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured. Please set it in .env file.")
    return genai.Client(api_key=GEMINI_API_KEY)


def load_content_store() -> list:
    """Load existing content for similarity comparison."""
    if CONTENT_STORE_FILE.exists():
        try:
            return json.loads(CONTENT_STORE_FILE.read_text())
        except json.JSONDecodeError:
            return []
    return []


def save_content_store(content: list):
    """Save content store to file."""
    CONTENT_STORE_FILE.write_text(json.dumps(content, indent=2))


# ==========================================================
#   SIMILARITY CHECK
# ==========================================================
async def check_similarity(content_1: str, content_2: str) -> dict:
    """
    Check similarity between two provided contents using Gemini.
    Provides detailed analysis of originality and similarity.
    
    Args:
        content_1: First content text to compare
        content_2: Second content text to compare
        
    Returns:
        dict with is_similar, similarity_level, originality_assessment, detailed_analysis
    """
    try:
        client = get_gemini_client()
        
        prompt = f"""You are a content originality and similarity analyzer. Compare the following two texts and provide a detailed analysis.

Text 1 (Content being checked):
{content_1}

Text 2 (Reference content):
{content_2}

Analyze these texts and return a JSON object with the following fields:

1. "is_similar": boolean - true if the contents share significant similarity in meaning, topic, or phrasing; false if they are substantially different

2. "similarity_level": string - one of these exact values:
   - "identical" - texts are the same or nearly word-for-word copies
   - "very_similar" - same topic with very similar ideas and structure, possibly paraphrased
   - "similar" - same topic with overlapping ideas but different presentation
   - "somewhat_similar" - related topics or themes but different focus
   - "different" - unrelated or completely different content

3. "originality_assessment": string - A 2-3 sentence assessment of whether Text 1 appears to be original content or if it seems derived from Text 2. Consider if it's a copy, paraphrase, inspired by, or completely independent.

4. "detailed_analysis": string - A 3-4 sentence detailed analysis explaining:
   - What specific similarities exist (if any)
   - What differences exist
   - Key themes or topics in each text
   - Your conclusion about the relationship between the texts

Respond ONLY with valid JSON, no markdown formatting or code blocks."""

        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1024
            )
        )
        
        # Parse response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (```json and ```)
            response_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        return {
            "is_similar": bool(result.get("is_similar", False)),
            "similarity_level": str(result.get("similarity_level", "different")),
            "originality_assessment": str(result.get("originality_assessment", "Unable to assess originality.")),
            "detailed_analysis": str(result.get("detailed_analysis", "Unable to provide detailed analysis."))
        }
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return default with error info
        return {
            "is_similar": False,
            "similarity_level": "different",
            "originality_assessment": "Unable to assess due to processing error.",
            "detailed_analysis": f"Analysis failed due to response parsing error: {str(e)}"
        }
    except Exception as e:
        raise Exception(f"Similarity check failed: {str(e)}")


# ==========================================================
#   SOCIAL CAPTION GENERATION
# ==========================================================
async def generate_social_caption(platform: str, title: str, description: str) -> dict:
    """
    Generate a social media caption for the given content.
    
    Args:
        platform: Target platform (instagram, twitter, facebook, etc.)
        title: Content title
        description: Content description
        
    Returns:
        dict with generated caption
    """
    try:
        client = get_gemini_client()
        
        # Platform-specific instructions
        platform_guides = {
            "instagram": "Use emojis, hashtags, and keep it engaging. Mention 'link in bio' for URLs. Max 2200 characters.",
            "twitter": "Keep it concise (under 280 characters). Use relevant hashtags. Be punchy and shareable.",
            "facebook": "Can be longer and more descriptive. Encourage engagement with questions or calls to action.",
            "tiktok": "Use trendy language, emojis, and relevant hashtags. Keep it fun and casual.",
            "linkedin": "Professional tone. Focus on value and insights. Use relevant industry hashtags.",
        }
        
        platform_guide = platform_guides.get(platform.lower(), "Create an engaging caption suitable for social media.")
        
        prompt = f"""Generate a social media caption for {platform}.

Content Title: {title}
Content Description: {description}

Platform Guidelines: {platform_guide}

Generate ONLY the caption text, nothing else. Do not include any explanations or metadata."""

        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=500
            )
        )
        
        caption = response.text.strip()
        
        return {"caption": caption}
        
    except Exception as e:
        raise Exception(f"Caption generation failed: {str(e)}")


# ==========================================================
#   TEXT SUMMARIZATION
# ==========================================================
async def summarize_text(content: str) -> dict:
    """
    Summarize the given text content.
    
    Args:
        content: Text content to summarize
        
    Returns:
        dict with summary
    """
    try:
        client = get_gemini_client()
        
        prompt = f"""Summarize the following content concisely. Focus on the key points and main takeaways.
Keep the summary brief but informative.

Content:
{content}

Provide ONLY the summary, nothing else."""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=300
            )
        )
        
        summary = response.text.strip()
        
        return {"summary": summary}
        
    except Exception as e:
        raise Exception(f"Summarization failed: {str(e)}")
