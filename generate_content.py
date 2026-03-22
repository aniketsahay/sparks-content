#!/usr/bin/env python3
"""
Sparks Content Generation Pipeline
====================================
1. Scrapes Reddit for real style examples (10-15 per category)
2. Uses Claude API to generate 35 new original entries per category
3. Saves the result as sparks_content.json
4. Pushes it to a public GitHub repo via the GitHub API
   → The app fetches the raw file from raw.githubusercontent.com

Usage:
    python generate_content.py

Requirements:
    pip install anthropic requests python-dotenv
"""

import os
import json
import base64
import datetime
import requests
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN")       # Personal Access Token (repo scope)
GITHUB_OWNER      = os.getenv("GITHUB_OWNER")       # Your GitHub username
GITHUB_REPO       = os.getenv("GITHUB_REPO")        # e.g. "sparks-content"
GITHUB_BRANCH     = os.getenv("GITHUB_BRANCH", "main")

TODAY        = datetime.date.today().strftime("%Y%m%d")
DATE_DISPLAY = datetime.date.today().strftime("%Y-%m-%d")
GENERATED_AT = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

REDDIT_HEADERS = {"User-Agent": "SparksContentGen/1.0"}

# ── Reddit source config ────────────────────────────────────────────────────────

REDDIT_SOURCES = {
    "shower": {
        "urls": [
            "https://www.reddit.com/r/Showerthoughts/hot.json?limit=100",
            "https://www.reddit.com/r/Showerthoughts/top.json?t=week&limit=100",
        ],
        "filter": lambda p: (
            not p.get("over_18") and
            not p.get("stickied") and
            p.get("selftext", "") == "" and
            20 < len(p.get("title", "")) < 280 and
            p.get("score", 0) > 50
        ),
        "transform": lambda p: p["title"],
        "count": 15,
    },
    "motivational": {
        "urls": [
            "https://www.reddit.com/r/GetMotivated/hot.json?limit=100",
            "https://www.reddit.com/r/GetMotivated/top.json?t=week&limit=100",
            "https://www.reddit.com/r/quotes/hot.json?limit=100",
            "https://www.reddit.com/r/quotes/top.json?t=week&limit=100",
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
        "count": 15,
    },
    "darkjoke": {
        "urls": [
            "https://www.reddit.com/r/darkjokes/hot.json?limit=100",
            "https://www.reddit.com/r/darkjokes/top.json?t=week&limit=100",
            "https://www.reddit.com/r/DarkHumor/hot.json?limit=100",
            "https://www.reddit.com/r/DarkHumor/top.json?t=week&limit=100",
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
        "count": 15,
    },
}

# ── Fallback examples ──────────────────────────────────────────────────────────
# Used when Reddit is unreachable — keeps the script from crashing.

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

Here are {count} real examples from Reddit's r/Showerthoughts (sorted by upvotes) to \
set the tone and style:

{examples}

Now generate 35 NEW, ORIGINAL shower thoughts. Rules:
- Each is a standalone sentence or two — no numbering, bullets, or prefixes like "Here's one:"
- Do NOT copy or paraphrase any of the examples above
- Vary the length: some short and punchy, some 2-3 sentences
- They should be genuinely mind-bending — not "clever-sounding but obvious"
- Avoid clichés like "if you think about it" or "have you ever noticed"
- No politics, no religion bashing
- Output the 35 thoughts one per entry, separated by a blank line""",

    "motivational": """\
You are generating motivational quotes for an app. They should feel like real quotes — \
the kind you'd screenshot and save.

Here are {count} real examples from Reddit's r/GetMotivated and r/quotes (sorted by \
upvotes) to set the tone:

{examples}

Now generate 35 NEW, ORIGINAL motivational quotes. Rules:
- Each ends with an attribution like "— [Name]", "— Unknown", or "— Ancient Proverb"
- You can attribute to real historical figures, real authors, or use "— Unknown"
- Do NOT use living celebrities or current public figures as the attribution
- Do NOT copy or paraphrase any of the examples above
- Vary the length: some short and punchy, some longer and philosophical
- They should feel authentic and earned, not like corporate motivational posters
- No numbering, no bullets, no prefixes
- Output the 35 quotes one per entry, separated by a blank line""",

    "darkjoke": """\
You are generating dark humor jokes for an app. They must be genuinely funny with real \
punchlines — not just edgy observations.

Here are {count} real examples from Reddit's r/darkjokes and r/DarkHumor (sorted by \
upvotes) to set the tone:

{examples}

Now generate 35 NEW, ORIGINAL dark jokes. Rules:
- Each has a real punchline — the kind that makes you laugh then feel slightly bad
- Two-liner format works great: setup on one line, punchline on the next
- Do NOT copy or paraphrase any of the examples above
- Dark but not gratuitously offensive — punch up or sideways, not pure shock value
- Vary the format: one-liners, two-liners, short anecdotes
- No numbering, no bullets, no prefixes
- Output the 35 jokes one per entry, separated by a blank line (two-liners = two lines \
within the same entry)""",
}


# ── Step 1: Scrape Reddit ───────────────────────────────────────────────────────

def fetch_reddit_examples(category: str) -> list[str]:
    """
    Fetches posts from Reddit for the given category and returns the
    top-scoring ones as plain text strings.
    Falls back to hardcoded examples if Reddit is unreachable.
    """
    config = REDDIT_SOURCES[category]
    posts = []
    seen_ids: set[str] = set()

    for url in config["urls"]:
        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", {}).get("children", []):
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
        print(f"    Warning: no posts for {category}, using fallback examples")
        return FALLBACK_EXAMPLES[category]

    posts.sort(key=lambda p: p.get("score", 0), reverse=True)
    top = posts[:config["count"]]
    return [config["transform"](p) for p in top]


# ── Step 2: Generate with Claude ────────────────────────────────────────────────

def generate_with_claude(
    category: str,
    examples: list[str],
    client: anthropic.Anthropic,
) -> list[str]:
    """
    Sends the examples to Claude and returns a list of up to 35 generated strings.
    """
    examples_text = "\n".join(f"• {ex}" for ex in examples)
    prompt = PROMPTS[category].format(examples=examples_text, count=len(examples))

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Each entry is separated by a blank line
    entries = [block.strip() for block in raw.split("\n\n") if block.strip()]

    # Drop any meta-commentary Claude might prepend
    filtered = []
    for entry in entries:
        lower = entry.lower()
        if lower.startswith(("here are", "sure,", "certainly", "of course")):
            continue
        if len(entry) < 10:
            continue
        filtered.append(entry)

    return filtered[:35]


# ── Step 3: Build output JSON ───────────────────────────────────────────────────

def build_output_json(content_map: dict[str, list[str]]) -> dict:
    """
    Wraps generated strings in the structured format the app expects.
    """
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
    """
    Creates or updates sparks_content.json in the configured GitHub repo
    using the GitHub Contents API. No local git installation needed.

    Returns the raw.githubusercontent.com URL the app can fetch from.
    """
    if not all([GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO]):
        raise ValueError(
            "GITHUB_TOKEN, GITHUB_OWNER, and GITHUB_REPO must all be set in .env"
        )

    file_path = "sparks_content.json"
    api_url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/contents/{file_path}"
    )
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Read local file and base64-encode it (required by the GitHub API)
    with open(local_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    # If the file already exists we need its current SHA to update it
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
        payload["sha"] = sha  # Required when updating an existing file

    resp = requests.put(api_url, headers=headers, json=payload)
    resp.raise_for_status()

    raw_url = (
        f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/{GITHUB_BRANCH}/{file_path}"
    )
    return raw_url


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Sparks Content Generation Pipeline")
    print(f"  {DATE_DISPLAY}")
    print("=" * 60)

    # 1. Scrape Reddit
    examples: dict[str, list[str]] = {}
    for category in ["shower", "motivational", "darkjoke"]:
        print(f"\n[1/4] Fetching Reddit examples — {category}...")
        examples[category] = fetch_reddit_examples(category)
        print(f"    → {len(examples[category])} examples collected")

    # 2. Generate with Claude
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set.\nAdd it to .env: ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    generated: dict[str, list[str]] = {}

    for category in ["shower", "motivational", "darkjoke"]:
        print(f"\n[2/4] Generating {category} content with Claude...")
        entries = generate_with_claude(category, examples[category], client)
        generated[category] = entries
        print(f"    → {len(entries)} entries generated")

    # 3. Build and save JSON locally
    print(f"\n[3/4] Building output JSON...")
    output = build_output_json(generated)

    local_path = "sparks_content.json"
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in output["content"].values())
    print(f"    → {total} total entries saved to {local_path}")

    # 4. Push to GitHub
    print(f"\n[4/4] Pushing to GitHub...")
    try:
        url = push_to_github(local_path)
        print(f"\n{'=' * 60}")
        print(f"  Done!")
        print(f"  Raw URL: {url}")
        print(f"\n  Paste this into redditService.js as GENERATED_CONTENT_URL")
        print(f"  (only needed once — the URL stays the same on every run)")
        print(f"{'=' * 60}\n")
    except Exception as exc:
        print(f"\n  Warning: GitHub push failed — {exc}")
        print(f"  Content saved locally to {local_path}")
        print(f"  Check GITHUB_TOKEN / GITHUB_OWNER / GITHUB_REPO in .env\n")


if __name__ == "__main__":
    main()
