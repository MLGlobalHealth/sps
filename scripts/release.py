#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "dl4bi-sps"
IMPORT_NAME = "dl4bi_sps"
REQUIRED_TOKENS = ("TEST_PYPI_TOKEN", "PYPI_TOKEN")
TEST_PYPI_PUBLISH_URL = "https://test.pypi.org/legacy/"
TEST_PYPI_CHECK_URL = "https://test.pypi.org/simple/"


class ReleaseError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Build and publish a {PACKAGE_NAME} release from a local .env file."
    )
    parser.add_argument(
        "env_file",
        type=Path,
        help="Path to a .env file containing TEST_PYPI_TOKEN and PYPI_TOKEN.",
    )
    parser.add_argument(
        "message",
        help="Release message appended to the version for the commit and tag.",
    )
    return parser.parse_args()


def read_env_file(path: Path) -> dict[str, str]:
    env_path = path.expanduser()
    if not env_path.is_file():
        raise ReleaseError(f"expected env file at {env_path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise ReleaseError(f"invalid env line {line_number}: {raw_line}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ReleaseError(f"invalid env key on line {line_number}")
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        values[key] = value
    return values


def require_tokens(values: dict[str, str]) -> tuple[str, str]:
    missing = [key for key in REQUIRED_TOKENS if not values.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise ReleaseError(f"missing required token(s) in env file: {joined}")
    return values["TEST_PYPI_TOKEN"], values["PYPI_TOKEN"]


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"+ {format_command(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def capture_command(cmd: list[str]) -> str:
    print(f"+ {format_command(cmd)}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def ensure_main_branch() -> None:
    branch = capture_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch != "main":
        raise ReleaseError(f"expected to release from main, found {branch}")


def ensure_clean_tracked_tree() -> None:
    checks = (
        ["git", "diff", "--quiet", "--exit-code"],
        ["git", "diff", "--cached", "--quiet", "--exit-code"],
    )
    for cmd in checks:
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise ReleaseError(
                "expected a clean tracked working tree before creating a release"
            )


def ensure_tag_does_not_exist(tag_name: str) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag_name}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        raise ReleaseError(f"tag already exists: {tag_name}")


def rebuild_dist() -> None:
    dist_dir = REPO_ROOT / "dist"
    if dist_dir.exists():
        if not dist_dir.is_dir():
            raise ReleaseError(f"expected dist to be a directory, found {dist_dir}")
        print(f"+ rm -rf {dist_dir.relative_to(REPO_ROOT)}", flush=True)
        shutil.rmtree(dist_dir)


def build_release() -> tuple[str, str]:
    version = capture_command(
        ["uv", "version", "--bump", "patch", "--frozen", "--dry-run", "--short"]
    )
    if not version:
        raise ReleaseError("failed to preview the bumped package version")
    tag_name = f"v{version}"
    ensure_tag_does_not_exist(tag_name)
    run_command(["uv", "version", "--bump", "patch", "--frozen"])
    actual_version = capture_command(["uv", "version", "--short"])
    if actual_version != version:
        raise ReleaseError(
            f"expected bumped version {version}, found {actual_version} after update"
        )
    return actual_version, tag_name


def publish_release(
    version: str,
    tag_name: str,
    message: str,
    test_token: str,
    pypi_token: str,
) -> None:
    release_message = message.strip()
    if not release_message:
        raise ReleaseError("release message must not be empty")

    commit_message = f"{tag_name} {release_message}"
    base_env = os.environ.copy()

    run_command(["git", "commit", "--no-verify", "-am", commit_message])
    rebuild_dist()
    run_command(["uv", "build", "--no-sources"])
    run_command(["git", "tag", "-a", tag_name, "-m", commit_message])

    test_env = base_env | {"UV_PUBLISH_TOKEN": test_token}
    run_command(
        [
            "uv",
            "publish",
            "--publish-url",
            TEST_PYPI_PUBLISH_URL,
            "--check-url",
            TEST_PYPI_CHECK_URL,
        ],
        env=test_env,
    )

    pypi_env = base_env | {"UV_PUBLISH_TOKEN": pypi_token}
    run_command(["uv", "publish"], env=pypi_env)

    run_command(["git", "push", "origin", "main"])
    run_command(["git", "push", "origin", tag_name])

    smoke_targets = (
        f"{PACKAGE_NAME}=={version}",
        f"{PACKAGE_NAME}[cpu]=={version}",
        f"{PACKAGE_NAME}[cuda12]=={version}",
        f"{PACKAGE_NAME}[cuda13]=={version}",
    )
    for target in smoke_targets:
        run_command(
            [
                "uv",
                "run",
                "--isolated",
                "--with",
                target,
                "--no-project",
                "--",
                "python",
                "-c",
                f"import {IMPORT_NAME}",
            ]
        )


def main() -> int:
    args = parse_args()
    env_values = read_env_file(args.env_file)
    test_token, pypi_token = require_tokens(env_values)
    ensure_main_branch()
    ensure_clean_tracked_tree()
    version, tag_name = build_release()
    publish_release(version, tag_name, args.message, test_token, pypi_token)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as exc:
        print(f"release: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
