"""Simple standard-library CLI for the local ApicalAI backend."""
import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base_url, path, payload=None, key=None):
    url = base_url.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=body, headers=headers)
    with urlopen(req, timeout=120) as response:
        return json.load(response)


def main(argv=None):
    parser = argparse.ArgumentParser(description="ApicalAI inference CLI")
    parser.add_argument("--url", default=os.getenv("APICALAI_API_URL", "http://127.0.0.1:8080"))
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8080)
    commands.add_parser("status")
    commands.add_parser("models")
    infer = commands.add_parser("infer")
    infer.add_argument("--model", default="demo-sentiment-v1")
    infer.add_argument("text")
    args = parser.parse_args(argv)
    if args.command == "serve":
        os.environ["PORT"] = str(args.port)
        from api.app import main as serve_api
        serve_api()
        return 0
    try:
        key = os.getenv("APICALAI_API_KEY")
        if args.command == "infer":
            result = request(args.url, "/api/inference", {"model": args.model, "input": {"text": args.text}}, key)
        else:
            path = "/api/health" if args.command == "status" else "/api/models"
            result = request(args.url, path, key=key)
        print(json.dumps(result, indent=2))
        return 0
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        print("Request failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
