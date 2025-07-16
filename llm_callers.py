# llm_callers.py
import os
from openai import OpenAI
import anthropic
import google.generativeai as genai

def call_openai(vec):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key)
    prompt = f"Reconstruct a coherent English sentence from this embedding: {vec}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=128, temperature=0.7
    )
    return res.choices[0].message.content.strip()

def call_claude(vec):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Client(key)
    prompt = anthropic.HUMAN_PROMPT + f"{vec}" + anthropic.AI_PROMPT
    res = client.completions.create(
        model="claude-2", prompt=prompt, max_tokens_to_sample=128
    )
    return res.completion.strip()

def call_gemini(vec):
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Reconstruct a coherent English sentence from this embedding: {vec}"
    res = model.generate_content(prompt)
    return getattr(res, "text", "").strip()

