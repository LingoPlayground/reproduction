#!/usr/bin/env python3
"""Fetch canvas node data from LibLib API and format for Stage 3.

Usage:
  python3 skills/timeline_plan/fetch_canvas.py <shareId> [--output formatted.json]
"""
import argparse
import json
import ssl
import sys
import urllib.request

CANVAS_API = "https://api.liblib.tv/api/canvas/project/share/detail"


def fetch_canvas(share_id: str) -> dict:
    url = f"{CANVAS_API}?shareId={share_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "CanvasStoryboard/3.0"})
    for verify in [True, False]:
        try:
            ctx = ssl.create_default_context()
            if not verify:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                data = json.loads(resp.read())
            if data.get("code") != 0:
                sys.exit(f"API error: {data}")
            return data["data"]
        except (ssl.SSLError, Exception):
            if verify: continue
            raise
    sys.exit("Cannot connect to canvas API")


def format_nodes(raw: dict) -> list:
    node_list = raw.get("nodeList", [])
    conn_list = raw.get("connectionList", [])

    node_map = {}
    for n in node_list:
        nk = n.get("nodeKey", "")
        nd = n.get("data", {})
        if isinstance(nd, str): nd = json.loads(nd)
        p = nd.get("params", {})
        if isinstance(p, str): p = json.loads(p)
        node_map[nk] = {
            "type": n.get("type", 0), "name": n.get("name", ""),
            "prompt": p.get("prompt", ""),
            "video_url": nd.get("url", ""),
            "ref_images": [],
        }
        # Unwrap list URLs
        if isinstance(node_map[nk]["video_url"], list):
            node_map[nk]["video_url"] = node_map[nk]["video_url"][0] if node_map[nk]["video_url"] else ""

    img_by_key = {nk: v for nk, v in node_map.items() if v["type"] == 2}
    for conn in conn_list:
        src = conn.get("source", ""); tgt = conn.get("target", "")
        if src in img_by_key and tgt in node_map and node_map[tgt]["type"] == 3:
            url = img_by_key[src].get("video_url", "")
            if not url:
                sn = [n for n in node_list if n.get("nodeKey") == src]
                if sn:
                    sd = sn[0].get("data", {})
                    if isinstance(sd, str): sd = json.loads(sd)
                    url = sd.get("url", "") or sd.get("poster", "")
            if url:
                if isinstance(url, list): url = url[0] if url else ""
                if url:
                    node_map[tgt]["ref_images"].append({"nodeId": src, "url": url})

    nodes = []
    for nk, v in node_map.items():
        if v["type"] != 3: continue
        nodes.append({"nodeId": nk, "name": v["name"], "prompt": v["prompt"],
                       "video_url": v["video_url"], "reference_images": v["ref_images"]})
    return nodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("share_id")
    p.add_argument("--output", "-o", default=None)
    args = p.parse_args()
    print(f"Fetching canvas for {args.share_id}...")
    raw = fetch_canvas(args.share_id)
    nodes = format_nodes(raw)
    imgs = sum(len(n["reference_images"]) for n in nodes)
    print(f"Found {len(nodes)} video nodes ({imgs} ref images)")
    out = args.output or f"runs/canvas_{args.share_id}.json"
    with open(out, "w") as f: json.dump(nodes, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
