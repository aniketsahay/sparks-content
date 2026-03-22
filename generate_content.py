#!/usr/bin/env python3
"""
Sparks Content Generation Pipeline
====================================
Each run:
1. Fetches the top 15 posts from Reddit per category (hot feed only)
2. Pulls 10 random entries from the existing generated content on GitHub
3. Combines them (up to 25 style examples) and asks Claude to generate 35 new entries
4. Pushes the new sparks_content.json back to GitHub

Using the existing generated content as part of the style input means Claude
sees fresh variation every run, even when Reddit hasn't changed much.

Usage:
    python generate_content.py

Requirements:
    pip install anthropic requests python-dotenv
"""

import os
import json
import base64
import random
import datetime
import requests
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER      = os.getenv("GITHUB_OWNER")
GITHUB_REPO       = os.getenv("GITHUB_REPO")
GITHUB_BRANCH     = os.getenv("GITHUB_BRANCH", "main")

# URL of the current published content — sampled as style examples each run
CONTENT_URL = (
    f"https://raw.githubusercontent.com/{os.getenv('GITHUB_OWNER', 'aniketsahay')}"
    f"/{os.getenv('GITHUB_REPO', 'sparks-content')}/main/sparks_content.json"
)

TODAY        = datetime.date.today().strftime("%Y%m%d")
DATE_DISPLAY = datetime.date.today().strftime("%Y-%m-%d")
GENERATED_AT = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

REDDIT_HEADERS  = {"User-Agent": "SparksContentGen/1.0"}
REDDIT_COUNT    = 15   # examples to pull from Reddit
GENERATED_COUNT = 10   # examples to sample from existing generated content

# ── Reddit source config ────────────────────────────────────────────────────────
# Hot feed only — we don't need weekly/monthly ranges because the existing
# generated content already provides variety between runs.

REDDIT_SOURCES = {
    "shower": {
        "urls": [
            "https://www.reddit.com/r/Showerthoughts/hot.json?limit=100",
        ],
        "filter": lambda p: (
            not p.get("over_18") and
            not p.get("stickied") and
            p.get("selftext", "") == "" and
            20 < len(p.get("title", "")) < 280 and
            p.get("score", 0) > 50
        ),
        "transform": lambda p: p["title"],
    },
    "motivational": {
        "urls": [
            "https://www.reddit.com/r/GetMotivated/hot.json?limit=100",
            "https://www.reddit.com/r/quotes/hot.json?limit=100",
        ],
        "filter": lambda p: (
            not p.get("over_18") and
            not p.get("stickied") and
            30 < len(p.get("title", "")) < 300 and
            p.get("score", 0) > 10 and
            (
                "[quote]" in p.get("title", "").lower() or
                not p.get("title", "").startswith("[")
            )
        ),
        "transform": lambda p: (
            p["title"].replace("[Quote]", "").replace("[quote]", "").strip()
        ),
    },
    "darkjoke": {
        "urls": [
            "https://www.reddit.com/r/darkjokes/hot.json?limit=100",
            "https://www.reddit.com/r/DarkHumor/hot.json?limit=100",
        ],
        "filter": lambda p: (
            not p.get("over_18") and
            not p.get("stickied") and
            len(p.get("title", "")) > 15 and
            (p.get("selftext", "") == "" or len(p.get("selftext", "")) < 400) and
            p.get("score", 0) > 10
        ),
        "transform": lambda p: (
            f"{p['title']}\n\n{p['selftext']}"
            if p.get("selftext") else p["title"]
        ),
    },
}

# ── Fallback examples ──────────────────────────────────────────────────────────

FALLBACK_EXAMPLES = {
    "shower": [
        "The brain is the only organ that named itself.",
        "We don't know what we don't know, and that's the most terrifying part.",
        "Your future self is a stranger you're writing letters to.",
        "Every expert was once a beginner who didn't quit.",
        "The universe is under no obligation to make sense to you.",
    ],
    "motivational": [
        '"The secret of getting ahead is getting started." — Mark Twain',
        '"It always seems impossible until it\'s done." — Nelson Mandela',
        '"You don\'t have to be great to start, but you have to start to be great." — Zig Ziglar',
        '"What you do today can improve all your tomorrows." — Ralph Marston',
        '"Act as if what you do makes a difference. It does." — William James',
    ],
    "darkjoke": [
        "I told my doctor I broke my arm in two places. He told me to stop going to those places.",
        "I asked the librarian if they had books about paranoia. She whispered, 'They're right behind you.'",
        "I used to hate funerals, but I've really warmed up to them.",
        "My therapist says I have trouble accepting reality. We'll see about that.",
        "My wife said I needed to grow up. I said nothing and kept building my LEGO set.",
    ],
}

# ── Claude prompts ──────────────────────────────────────────────────────────────

PROMPTS = {
    "shower": """\
You are generating original Shower Thoughts for an app. These are genuine observations \
that make people pause and say "wait... that's actually true."

Here are style examples — a mix of real Reddit posts and previously generated entries \
— to set the tone:

{examples}

Now generate 35 NEW, ORIGINAL shower thoughts. Rules:
- Each is a standalone sentence or two — no numbering, bullets, or prefixes
- Do NOT copy or paraphrase any of the examples above
- Vary the length: some short and punchy, some 2-3 sentences
- Genuinely mind-bending — not "clever-sounding but obvious"
- Avoid clichés like "if you think about it" or "have you ever noticed"
- No politics, no religion bashing
- Output the 35 thoughts one per entry, separated by a blank line""",

    "motivational": """\
You are generating motivational quotes for an app. They should feel like real quotes — \
the kind you'd screenshot and save.

Here are style examples — a mix of real Reddit posts and previously generated entries \
— to set the tone:

{examples}

Now generate 35 NEW, ORIGINAL motivational quotes. Rules:
- Each ends with an attribution: "— [Name]", "— Unknown", or "— Ancient Proverb"
- Attribute to real historical figures, real authors, or use "— Unknown"
- Do NOT use living celebrities or current public figures
- Do NOT copy or paraphrase any of the examples above
- Vary the length: some short and punchy, some longer and philosophical
- Feel authentic and earned, not like corporate motivational posters
- No numbering, no bullets, no prefixes
- Output the 35 quotes one per entry, separated by a blank line""",

    "darkjoke": """\
You are generating dark humor jokes for an app. They must be genuinely funny with real \
punchlines — not just edgy observations.

Here are style examples — a mix of real Reddit posts and previously generated entries \
— to set the tone:

{examples}

Now generate 35 NEW, ORIGINAL dark jokes. Rules:
- Each has a real punchline — the kind that makes you laugh then feel slightly bad
- Two-liner format works great: setup on one line, punchline on the next
- Do NOT copy or paraphrase any of the examples above
- Dark but not gratuitously offensive — punch up or sideways, not pure shock value
- Vary the format: one-liners, two-liners, short anecdotes
- No numbering, no bullets, no prefixes
- Output the 35 jokes one per entry, separated by a blank line""",
}


# ── Step 1a: Fetch Reddit examples ─────────────────────────────────────────────

def fetch_reddit_examples(category: str) -> list[str]:
    """
    Pulls hot posts from Reddit, filters them, and returns up to REDDIT_COUNT
    as plain text strings sorted by score.
    """
    config = REDDIT_SOURCES[category]
    posts = []
    seen_ids: set[str] = set()

    for url in config["urls"]:
        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, timeout=10)
            resp.raise_for_status()
            for item in resp.json().get("data", {}).get("children", []):
                p = item.get("data", {})
                pid = p.get("id", "")
                if pid in seen_ids:
                    continue
                if config["filter"](p):
                    posts.append(p)
                    seen_ids.add(pid)
        except Exception as exc:
            print(f"    Warning: could not fetch {url} — {exc}")

    if not posts:
        print(f"    Warning: no Reddit posts for {category}, using fallback")
        return FALLBACK_EXAMPLES[category]

    posts.sort(key=lambda p: p.get("score", 0), reverse=True)
    return [config["transform"](p) for p in posts[:REDDIT_COUNT]]


# ── Step 1b: Sample from existing generated content ────────────────────────────

def fetch_existing_samples(category: str) -> list[str]:
    """
    Downloads the current sparks_content.json from GitHub and returns
    GENERATED_COUNT randomly sampled entries for the given category as strings.
    Returns [] on first run (when the file doesn't exist yet).
    """
    try:
        resp = requests.get(CONTENT_URL, timeout=10)
        if resp.status_code == 404:
            return []  # First run — no existing content yet
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("content", {}).get(category, [])
        if not entries:
            return []
        sample = random.sample(entries, min(GENERATED_COUNT, len(entries)))
        return [e["thought"] for e in sample]
    except Exception as exc:
        print(f"    Warning: could not fetch existing content — {exc}")
        return []


# ── Step 2: Generate with Claude ────────────────────────────────────────────────

def generate_with_claude(
    category: str,
    examples: list[str],
    client: anthropic.Anthropic,
) -> list[str]:
    """
    Sends combined style examples to Claude and returns up to 35 new entries.
    """
    examples_text = "\n".join(f"• {ex}" for ex in examples)
    prompt = PROMPTS[category].format(examples=examples_text)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    entries = [block.strip() for block in raw.split("\n\n") if block.strip()]

    filtered = []
    for entry in entries:
        if entry.lower().startswith(("here are", "sure,", "certainly", "of course")):
            continue
        if len(entry) < 10:
            continue
        filtered.append(entry)

    return filtered[:35]


# ── Step 3: Build output JSON ───────────────────────────────────────────────────

def build_output_json(content_map: dict[str, list[str]]) -> dict:
    prefix_map = {"shower": "shower", "motivational": "motiv", "darkjoke": "dark"}

    output = {
        "version": DATE_DISPLAY,
        "generated_at": GENERATED_AT,
        "content": {},
    }

    for category, entries in content_map.items():
        prefix = prefix_map[category]
        output["content"][category] = [
            {
                "id": f"gen_{prefix}_{TODAY}_{str(i + 1).zfill(3)}",
                "thought": entry,
                "type": category,
                "source": "generated",
            }
            for i, entry in enumerate(entries)
        ]

    return output


# ── Step 4: Push to GitHub ─────────────────────────────────────────────────────

def push_to_github(local_path: str) -> str:
    if not all([GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO]):
        raise ValueError("GITHUB_TOKEN, GITHUB_OWNER, and GITHUB_REPO must be set in .env")

    file_path = "sparks_content.json"
    api_url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/contents/{file_path}"
    )
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    with open(local_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    sha = None
    try:
        check = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH})
        if check.status_code == 200:
            sha = check.json()["sha"]
    except Exception as exc:
        print(f"    Warning: could not check existing file SHA — {exc}")

    payload = {
        "message": f"content: update for {DATE_DISPLAY}",
        "content": encoded,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    resp = requests.put(api_url, headers=headers, json=payload)
    resp.raise_for_status()

    return (
        f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/{GITHUB_BRANCH}/{file_path}"
    )


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Sparks Content Generation Pipeline")
    print(f"  {DATE_DISPLAY}")
    print("=" * 60)

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    generated: dict[str, list[str]] = {}

    for category in ["shower", "motivational", "darkjoke"]:
        print(f"\n── {category} ──────────────────────────────────────────")

        # 1a. Reddit examples (hot feed, top 15 by score)
        print(f"  Fetching Reddit examples...")
        reddit_examples = fetch_reddit_examples(category)
        print(f"    → {len(reddit_examples)} from Reddit")

        # 1b. Sample from existing generated content
        print(f"  Sampling existing generated content...")
        generated_samples = fetch_existing_samples(category)
        print(f"    → {len(generated_samples)} from previous run")

        # Combine: Reddit first (higher weight as anchors), then generated samples
        combined_examples = reddit_examples + generated_samples
        print(f"    → {len(combined_examples)} total style examples")

        # 2. Generate 35 new entries
        print(f"  Generating with Claude...")
        entries = generate_with_claude(category, combined_examples, client)
        generated[category] = entries
        print(f"    → {len(entries)} new entries generated")

    # 3. Build and save JSON
    print(f"\n── Output ──────────────────────────────────────────────")
    output = build_output_json(generated)

    local_path = "sparks_content.json"
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in output["content"].values())
    print(f"  {total} total entries saved to {local_path}")

    # 4. Push to GitHub
    print(f"  Pushing to GitHub...")
    try:
        url = push_to_github(local_path)
        print(f"\n{'=' * 60}")
        print(f"  Done! → {url}")
        print(f"{'=' * 60}\n")
    except Exception as exc:
        print(f"\n  Warning: GitHub push failed — {exc}")
        print(f"  Content saved locally to {local_path}\n")


if __name__ == "__main__":
    main()
