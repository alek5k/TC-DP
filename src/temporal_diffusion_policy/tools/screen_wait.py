import argparse
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import List

SESSION_LINE = re.compile(
    r"""^
        (?P<pid>\d+)\.
        (?P<name>.+?)          # session name (lazy, can contain dots)
        \s+\([^)]*\)\s+        # timestamp block
        \((?P<state>Attached|Detached)\)
        \s*$
    """,
    re.VERBOSE,
)
parser = argparse.ArgumentParser(description=("Poll GNU Screen until provided session exits."))
parser.add_argument("--pattern", required=True, help="Name or screen ID used to extract session names from 'screen -ls' lines.")

def get_active_screen_sessions() -> List[str]:
    """Return the identifiers of currently active GNU Screen sessions."""
    try:
        result = subprocess.run(["screen", "-ls"], check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit("GNU Screen is not installed or not in PATH.") from exc

    output = (result.stdout or "") + (result.stderr or "")
    sessions: List[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = SESSION_LINE.match(line)
        if match:
            sessions.append(line)
    return sessions

def wait_for_screen_exit(pattern: str, poll_interval: float = 5.0, break_on_training_completed: bool = True) -> None:
    """Keep polling screen until no sessions remain, then exit."""
    try:
        while True:
            time.sleep(poll_interval)
            sessions = get_active_screen_sessions()

            found = False
            for session in sessions:
                session = session.replace("\t", " ")
                match = SESSION_LINE.match(session)
                
                if match and break_on_training_completed and Path(f"/tmp/{match.group('pid')}.{pattern}_train_complete.json").exists():
                    print("Training completion file detected; exiting immediately.")
                    sys.exit(0)

                if match and (pattern == match.group("pid") or pattern == match.group("name")):
                    found = True
                    break

            if not found:
                print("No matching screen sessions remain; exiting.")
                sys.exit(0)
            
    except KeyboardInterrupt:
        print("Interrupted by user; exiting without waiting for sessions to close.")
        sys.exit(1)


if __name__ == "__main__":
    args = parser.parse_args()
    wait_for_screen_exit(args.pattern)

