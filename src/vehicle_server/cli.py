"""Command-line entry point: `vehicle-server <command>` (import, serve, report, bringup, list,
derive, golden, export)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vehicle_server import bringup, index, report
from vehicle_server.config import load_settings
from vehicle_server.ingest import (
    InvalidSessionError,
    SessionConflictError,
    ingest_path,
    rederive_session,
)

EXIT_OK_OR_WARN = 0
EXIT_FAIL_OR_ERROR = 1
EXIT_CONFLICT = 2

# `vehicle-server bringup`: exit code per gate verdict (an unknown session is EXIT_FAIL_OR_ERROR).
BRINGUP_EXIT = {"pass": 0, "fail": 1, "incomplete": 2}


def _cmd_import(args: argparse.Namespace) -> int:
    settings = load_settings()
    source = Path(args.path).resolve()
    try:
        result = ingest_path(source, settings)
    except InvalidSessionError as exc:
        print(f"error: invalid session: {exc}", file=sys.stderr)
        return EXIT_FAIL_OR_ERROR
    except SessionConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFLICT

    print(report.render_text(result.report), end="")
    return EXIT_OK_OR_WARN if result.report["status"] in ("ok", "warn") else EXIT_FAIL_OR_ERROR


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("vehicle_server.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    settings = load_settings()
    report_path = settings.sessions_dir / args.session_id / "report.json"
    if not report_path.is_file():
        print(f"error: unknown session: {args.session_id}", file=sys.stderr)
        return EXIT_FAIL_OR_ERROR

    report_dict = json.loads(report_path.read_text())
    if args.format == "text":
        print(report.render_text(report_dict), end="")
    else:
        print(json.dumps(report_dict, indent=2))
    return EXIT_OK_OR_WARN if report_dict["status"] in ("ok", "warn") else EXIT_FAIL_OR_ERROR


def _cmd_bringup(args: argparse.Namespace) -> int:
    settings = load_settings()
    session_dir = settings.sessions_dir / args.session_id
    report_path = session_dir / "report.json"
    if not report_path.is_file():
        print(f"error: unknown session: {args.session_id}", file=sys.stderr)
        return EXIT_FAIL_OR_ERROR

    result = bringup.check_gate(
        args.gate, json.loads(report_path.read_text()), bringup.load_inputs(session_dir)
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(bringup.format_text(result), end="")
    return BRINGUP_EXIT[result.verdict]


def _cmd_list(_args: argparse.Namespace) -> int:
    settings = load_settings()
    for row in index.list_sessions(settings.index_db_path):
        print(
            f"{row.session_id}  status={row.status}  packets={row.packet_count}  "
            f"loss={row.loss_percent:.2f}%  imu={'yes' if row.has_imu else 'no'}  "
            f"gps={'yes' if row.has_gps else 'no'}  "
            f"signals={'yes' if row.has_vehicle_signals else 'no'}  "
            f"derived={'yes' if row.has_derived else 'no'}"
        )
    return EXIT_OK_OR_WARN


def _cmd_derive(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.all:
        session_ids = [row.session_id for row in index.list_sessions(settings.index_db_path)]
    else:
        session_ids = [args.session_id]

    exit_code = EXIT_OK_OR_WARN
    for session_id in session_ids:
        try:
            block, summary_block = rederive_session(settings, session_id)
        except FileNotFoundError:
            print(f"error: unknown session: {session_id}", file=sys.stderr)
            exit_code = EXIT_FAIL_OR_ERROR
            continue
        if "error" in block:
            print(f"{session_id}  derived error: {block['error']}", file=sys.stderr)
            exit_code = EXIT_FAIL_OR_ERROR
            continue
        if "error" in summary_block:
            print(f"{session_id}  derived summary error: {summary_block['error']}", file=sys.stderr)
            exit_code = EXIT_FAIL_OR_ERROR
        summary = ", ".join(f"{n} {status}" for status, n in sorted(block["summary"].items()))
        print(f"{session_id}  derived rows={block['rows']} valid={block['valid_rows']}  {summary}")
        if "error" not in summary_block:
            counts = ", ".join(
                f"{n} {status}" for status, n in sorted(summary_block["summary"].items())
            )
            print(f"{session_id}  derived summary  {counts}")
    return exit_code


def _cmd_golden(args: argparse.Namespace) -> int:
    from vehicle_server.derived import golden

    return golden.run(args.out, args.manifest)


def _cmd_export(args: argparse.Namespace) -> int:
    try:
        from vehicle_server.mdf4 import export_mdf4
    except ModuleNotFoundError as exc:
        print(
            f"error: MDF4 export needs the mdf4 extra ({exc.name} is missing): "
            "uv sync --extra mdf4",
            file=sys.stderr,
        )
        return EXIT_FAIL_OR_ERROR

    settings = load_settings()
    session_dir = settings.sessions_dir / args.session_id
    try:
        output = export_mdf4(session_dir, include_position=args.include_position)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAIL_OR_ERROR
    print(output)
    return EXIT_OK_OR_WARN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vehicle-server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="ingest a session folder or zip")
    import_parser.add_argument("path", help="path to a session folder or .zip file")
    import_parser.set_defaults(func=_cmd_import)

    serve_parser = subparsers.add_parser("serve", help="run the HTTP API (uvicorn)")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=_cmd_serve)

    report_parser = subparsers.add_parser("report", help="print a session's report")
    report_parser.add_argument("session_id")
    report_parser.add_argument("--format", choices=["json", "text"], default="text")
    report_parser.set_defaults(func=_cmd_report)

    bringup_parser = subparsers.add_parser(
        "bringup", help="check a session against bring-up gate B0, V0 or V1"
    )
    bringup_parser.add_argument("session_id")
    bringup_parser.add_argument("--gate", required=True, type=str.upper, choices=bringup.GATES)
    bringup_parser.add_argument("--json", action="store_true", help="print the JSON result")
    bringup_parser.set_defaults(func=_cmd_bringup)

    list_parser = subparsers.add_parser("list", help="list ingested sessions")
    list_parser.set_defaults(func=_cmd_list)

    derive_parser = subparsers.add_parser(
        "derive", help="recompute the derived signals of stored sessions (D-070)"
    )
    derive_parser.add_argument("session_id", nargs="?", help="a stored session id")
    derive_parser.add_argument("--all", action="store_true", help="every stored session")
    derive_parser.set_defaults(func=_cmd_derive)

    golden_parser = subparsers.add_parser(
        "golden", help="write the golden C vectors for the rt-core host tests (D-070)"
    )
    from vehicle_server.derived import golden

    golden.add_arguments(golden_parser)
    golden_parser.set_defaults(func=_cmd_golden)

    export_parser = subparsers.add_parser(
        "export", help="write a session's MDF4 archive next to its parquet files"
    )
    export_parser.add_argument("session_id")
    export_parser.add_argument(
        "--include-position",
        action="store_true",
        help="include GPS position columns (off by default, D-076)",
    )
    export_parser.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "derive" and bool(args.all) == bool(args.session_id):
        parser.error("derive needs exactly one of <session_id> or --all")
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
