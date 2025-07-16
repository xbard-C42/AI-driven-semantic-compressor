# llm_callers.py
import os
from openai import OpenAI
import anthropic
import google.generativeai as genai

def call_openai(vec):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    prompt = f"Reconstruct a coherent English sentence based on this semantic embedding:\n{vec}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.7
    )
    return res.choices[0].message.content.strip()

def call_claude(vec):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Client(api_key)
    prompt = (
        anthropic.HUMAN_PROMPT
        + f"Reconstruct a coherent English sentence based on this semantic embedding:\n{vec}"
        + anthropic.AI_PROMPT
    )
    res = client.completions.create(
        model="claude-2",
        prompt=prompt,
        max_tokens_to_sample=128
    )
    return res.completion.strip()

def call_gemini(vec):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Reconstruct a coherent English sentence based on this semantic embedding:\n{vec}"
    res = model.generate_content(prompt)
    return getattr(res, "text", "[No response generated]").strip()
