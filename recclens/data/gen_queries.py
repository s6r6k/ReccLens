"""Generate shopper-style search queries per listing with Claude.

One Message Batches request per listing (50% of standard price), with a JSON
schema so every reply parses. Queries are written in en, id, ms, th and vi.

    python -m recclens.data.gen_queries submit            # writes the batch id
    python -m recclens.data.gen_queries collect           # waits, then writes queries.csv
    python -m recclens.data.gen_queries submit --limit 200  # small sample to hand-check

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""

import argparse
import json
import time
from pathlib import Path

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from omegaconf import OmegaConf

LANGUAGE_NAMES = {
    "en": "English",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
}

SYSTEM = (
    "You write realistic e-commerce search queries for Southeast Asian shoppers. "
    "Given a product listing title, write what a shopper looking for that exact product "
    "would type into a marketplace search bar. Queries are short (2 to 7 words), mostly "
    "lowercase, may use common abbreviations or slang, and must not copy the title verbatim. "
    "Keep brand names and model numbers as shoppers would type them."
)


def output_schema(languages: list[str]) -> dict:
    query_list = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {lang: query_list for lang in languages},
        "required": languages,
        "additionalProperties": False,
    }


def build_request(
    posting_id: str, title: str, languages: list[str], per_language: int, model: str, effort: str
) -> Request:
    wanted = ", ".join(f"{LANGUAGE_NAMES[lang]} ({lang})" for lang in languages)
    prompt = (
        f"Listing title: {title}\n\n"
        f"Write {per_language} different search queries in each of: {wanted}."
    )
    return Request(
        custom_id=posting_id,
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=2048,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": output_schema(languages)},
            },
        ),
    )


def submit(cfg, items_csv: Path, limit: int | None, state_file: Path) -> None:
    items = pd.read_csv(items_csv)
    if limit:
        items = items.sample(n=min(limit, len(items)), random_state=cfg.seed)
    q = cfg.data.queries
    requests = [
        build_request(pid, title, list(q.languages), q.per_listing, q.model, q.effort)
        for pid, title in zip(items["posting_id"], items["title"].astype(str), strict=True)
    ]
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"batch_id": batch.id, "n_requests": len(requests)}))
    print(f"Submitted batch {batch.id} with {len(requests)} requests; id saved to {state_file}")


def collect(cfg, state_file: Path, out_csv: Path, poll_seconds: int) -> None:
    batch_id = json.loads(state_file.read_text())["batch_id"]
    client = anthropic.Anthropic()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        print(f"{batch.processing_status}: {batch.request_counts.processing} still processing")
        time.sleep(poll_seconds)

    languages = list(cfg.data.queries.languages)
    rows, failed = [], []
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            failed.append((result.custom_id, result.result.type))
            continue
        message = result.result.message
        if message.stop_reason in ("refusal", "max_tokens"):
            failed.append((result.custom_id, message.stop_reason))
            continue
        text = next(b.text for b in message.content if b.type == "text")
        data = json.loads(text)
        for lang in languages:
            for query in dict.fromkeys(q.strip() for q in data[lang] if q.strip()):
                rows.append({"posting_id": result.custom_id, "language": lang, "query": query})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} queries to {out_csv}")
    if failed:
        print(f"{len(failed)} listings failed; first few: {failed[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["submit", "collect"])
    parser.add_argument("--config", type=Path, default=Path("configs"))
    parser.add_argument("--items", type=Path, default=None, help="defaults to data.train_csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--state", type=Path, default=Path("data/processed/query_batch.json"))
    parser.add_argument("--out", type=Path, default=None, help="defaults to data.queries.output")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load(args.config / "config.yaml"),
        {"data": OmegaConf.load(args.config / "data" / "shopee.yaml")},
    )
    if args.command == "submit":
        submit(cfg, args.items or Path(cfg.data.train_csv), args.limit, args.state)
    else:
        collect(cfg, args.state, args.out or Path(cfg.data.queries.output), args.poll_seconds)


if __name__ == "__main__":
    main()
