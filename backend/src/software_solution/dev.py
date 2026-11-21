"""Launch the FastAPI backend and the React frontend together."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SOLUTION_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = SOLUTION_ROOT / "frontend"


def _npm_argv(*arguments: str) -> list[str]:
    """Return a command that runs npm on both Windows and POSIX systems."""
    if shutil.which("npm") is None:
        raise RuntimeError("npm was not found on PATH. Install Node.js and npm first.")

    if os.name == "nt":
        return ["cmd", "/c", "npm", *arguments]

    npm = shutil.which("npm")
    assert npm is not None
    return [npm, *arguments]


def _process_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate(process: subprocess.Popen) -> None:
    """Stop a process and all of its children."""
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


def _run() -> int:
    parser = argparse.ArgumentParser(
        description="Start the FastAPI backend and the React frontend together."
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="Port for the FastAPI server (default: 8000).",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=None,
        help="Port for the React dev server (default: 3000 or PORT from frontend/.env).",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable automatic backend reload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without starting anything.",
    )
    args = parser.parse_args()

    try:
        npm_command = _npm_argv("start")
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    backend_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "software_solution.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.backend_port),
    ]
    if not args.no_reload:
        backend_command.append("--reload")

    environment = os.environ.copy()
    environment["REACT_APP_API_URL"] = (
        environment.get("REACT_APP_API_URL")
        or f"http://127.0.0.1:{args.backend_port}"
    )
    if args.frontend_port is not None:
        environment["PORT"] = str(args.frontend_port)

    if args.dry_run:
        print(f"Backend:  {' '.join(backend_command)}")
        print(f"Frontend: {' '.join(npm_command)}")
        print(f"Frontend directory: {FRONTEND_ROOT}")
        return 0

    if not (FRONTEND_ROOT / "node_modules").is_dir():
        print("Frontend dependencies are missing. Running npm install...", flush=True)
        install = subprocess.run(
            _npm_argv("install"),
            cwd=FRONTEND_ROOT,
            env=environment,
            check=False,
        )
        if install.returncode != 0:
            return install.returncode

    frontend_port = args.frontend_port or environment.get("PORT", "3000")
    print(f"Backend:  http://127.0.0.1:{args.backend_port}", flush=True)
    print(f"Frontend: http://localhost:{frontend_port}", flush=True)
    print("Press Ctrl+C to stop both services.", flush=True)

    processes: list[subprocess.Popen] = []
    try:
        processes.append(
            subprocess.Popen(
                backend_command,
                cwd=BACKEND_ROOT,
                env=environment,
                **_process_options(),
            )
        )
        processes.append(
            subprocess.Popen(
                npm_command,
                cwd=FRONTEND_ROOT,
                env=environment,
                **_process_options(),
            )
        )

        while True:
            for name, process in zip(("Backend", "Frontend"), processes):
                exit_code = process.poll()
                if exit_code is not None:
                    print(
                        f"{name} stopped with exit code {exit_code}; shutting down...",
                        flush=True,
                    )
                    return exit_code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
        return 0
    finally:
        for process in processes:
            _terminate(process)


def main() -> None:
    raise SystemExit(_run())


if __name__ == "__main__":
    main()
