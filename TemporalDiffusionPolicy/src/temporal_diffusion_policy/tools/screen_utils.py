import argparse
from pathlib import Path
import subprocess
import sys

def get_screen_session_pid(session: str) -> int | None:
    result = subprocess.run(
        ["screen", "-ls"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        line = line.strip()
        if f".{session}" in line:
            first_field = line.split()[0]          # e.g. "12345.mysession"
            pid_str = first_field.split(".", 1)[0] # e.g. "12345"
            if pid_str.isdigit():
                return int(pid_str)

    return None

def ensure_interactive_session(session: str) -> tuple[bool, str | None]:
    screen_cmd = [
        "screen",
        "-T", "screen-256color",    # advertise 256-color capability inside each screen window
        "-S", session,          # -S: target/create the session
        "-dm",                  # -d -m: start detached
        "bash",
        "-l",
        "-i",                   # -l -i: login + interactive shell
    ]
    result = subprocess.run(screen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        pid = get_screen_session_pid(session)
        print(f"Session '{session}' ready for commands. pid={pid}")
        return True, f"{pid}.{session}" if pid is not None else session

    err = (result.stderr or b"").decode().strip()
    if "There is a screen on" in err:
        pid = get_screen_session_pid(session)
        print(f"Session '{session}' already running; reusing it.")
        return True, f"{pid}.{session}" if pid is not None else session

    print(f"Warning: failed to start session '{session}'. {err}", file=sys.stderr)
    return False, None


def run_command_in_session(session: str, command: str) -> bool:
    script_path = Path(f"/tmp/{session}.sh")
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "shopt -s expand_aliases\n"
        "source ~/.bashrc 2>/dev/null || true\n"
        f"{command}\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    payload = f"source {script_path}\r"

    # payload = f"{command}\n" # 100 CHARACTER LIMIT!! NEED TO USE A SCRIPT.
    screen_cmd = [
        "screen",
        "-S", session,
        "-p", "0",
        "-X", "stuff",
        payload,
    ]
    result = subprocess.run(screen_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = (result.stderr or b"").decode().strip()
        print(f"Warning: failed to dispatch command to '{session}'. {err}", file=sys.stderr)
        return False

    print(f"Dispatched '{command}' to session '{session}'.")
    return True


def start_sessions(commands: dict, stagger_interval_seconds: int, dryrun: bool = False) -> int:
    '''
    dict key should be name of screen session, value should be command to run in that session
    '''
    valid_entries = []
    for i, (session_name, raw_command) in enumerate(commands.items()):
        session = session_name.strip()
        cmd_str = raw_command.strip()
        if stagger_interval_seconds > 0:
            sleep_duration = i * stagger_interval_seconds  # stagger startups by the configured interval
            cmd_str = f"sleep {sleep_duration}s && {cmd_str}"
        if not session or not cmd_str:
            print("Error: each COMMANDS entry must map a non-empty session name to a command.", file=sys.stderr)
            continue
        valid_entries.append((session, cmd_str))

    if not valid_entries:
        return 1

    if dryrun:
        print("[Dry Run]")
        for session, cmd_str in valid_entries:
            print(f"Session: {session}")
            print(f"Command: {cmd_str}")
            print()
        return 0

    created_sessions = set()
    created_sessions_full_names = dict()  # maps session name to full name with PID if available
    launch_failed = False
    for session, _ in valid_entries:
        success, session_full_name = ensure_interactive_session(session)
        if success:
            created_sessions.add(session)
            created_sessions_full_names[session] = session_full_name
        else:
            launch_failed = True

    exec_failed = False
    for session, cmd_str in valid_entries:
        if session not in created_sessions:
            continue
        if not run_command_in_session(created_sessions_full_names[session], cmd_str):
            exec_failed = True
    return 0 if not (launch_failed or exec_failed) else 1


def stop_sessions(commands: dict) -> int:
    stop_failed = False

    for session_name in commands:
        session = session_name.strip()
        if not session:
            continue

        screen_cmd = [
            "screen",
            "-S", session,      # -S: target the given session name
            "-X", "quit",       # -X quit: send the 'quit' command to terminate the session
        ]

        try:
            result = subprocess.run(screen_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("Error: 'screen' not found. Install it with 'sudo apt install screen'.", file=sys.stderr)
            return 2

        if result.returncode != 0:
            err = (result.stderr or b"").decode().strip()
            print(f"Warning: failed to close session '{session}'. {err}", file=sys.stderr)
            stop_failed = True
        else:
            print(f"Closed screen session '{session}'.")

    return 0 if not stop_failed else 1