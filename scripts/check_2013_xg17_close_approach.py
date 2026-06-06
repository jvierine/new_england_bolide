#!/usr/bin/env python3
"""Check JPL SBDB close approaches for the closest NEA match, 2013 XG17."""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request


CAD_URL = "https://ssd-api.jpl.nasa.gov/cad.api"
AU_KM = 149_597_870.7
LD_KM = 384_400.0


def fetch_cad(args: argparse.Namespace) -> dict:
    params = {
        "des": args.designation,
        "body": args.body,
        "date-min": args.date_min,
        "date-max": args.date_max,
        "dist-max": args.dist_max,
        "sort": "-date",
    }
    url = f"{CAD_URL}?{urllib.parse.urlencode(params)}"
    context = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, timeout=args.timeout, context=context) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc) and not args.insecure:
            raise SystemExit(
                "TLS certificate verification failed. Re-run with --insecure on this machine, "
                "or fix the local Python CA bundle."
            ) from exc
        raise
    payload["_query_url"] = url
    return payload


def row_dict(fields: list[str], row: list[str]) -> dict[str, str]:
    return {name: value for name, value in zip(fields, row, strict=True)}


def format_approach(row: dict[str, str]) -> str:
    dist_au = float(row["dist"])
    dist_km = dist_au * AU_KM
    dist_ld = dist_km / LD_KM
    return (
        f"{row['des']} close approach to Earth: {row['cd']} UTC\n"
        f"  distance = {dist_au:.9f} au = {dist_km:,.0f} km = {dist_ld:.2f} LD\n"
        f"  relative speed = {float(row['v_rel']):.3f} km/s\n"
        f"  time uncertainty = {row.get('t_sigma_f', '--')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report JPL SBDB close approaches for 2013 XG17, the closest NEA orbital match."
    )
    parser.add_argument("--designation", default="2013 XG17")
    parser.add_argument("--body", default="Earth")
    parser.add_argument("--date-min", default="1900-01-01")
    parser.add_argument("--date-max", default="2026-05-30", help="Default is the bolide date.")
    parser.add_argument("--dist-max", default="0.5", help="Maximum close-approach distance in au.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification if local CA setup is broken.")
    parser.add_argument("--show", type=int, default=5, help="Number of latest records to print.")
    args = parser.parse_args()

    payload = fetch_cad(args)
    fields = payload.get("fields", [])
    rows = [row_dict(fields, row) for row in payload.get("data", [])]
    if not rows:
        print(f"No close approaches returned by JPL CAD API for {args.designation}.", file=sys.stderr)
        print(payload.get("_query_url", CAD_URL), file=sys.stderr)
        raise SystemExit(1)

    latest = rows[0]
    closest = min(rows, key=lambda row: float(row["dist"]))

    print(f"Source: {payload.get('signature', {}).get('source', 'JPL SBDB Close-Approach Data API')}")
    print(f"Query: {payload['_query_url']}")
    print(f"Returned records: {len(rows)}\n")
    print("Latest close approach in queried interval")
    print(format_approach(latest))
    print()
    print("Closest close approach in queried interval")
    print(format_approach(closest))

    if args.show > 0:
        print(f"\nLatest {min(args.show, len(rows))} returned records")
        for row in rows[: args.show]:
            print(f"  {row['cd']} UTC  dist={float(row['dist']):.9f} au  v_rel={float(row['v_rel']):.3f} km/s")


if __name__ == "__main__":
    main()
