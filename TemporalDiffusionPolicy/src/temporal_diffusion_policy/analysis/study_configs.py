"""Read and edit the manual analysis study configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STUDY_CONFIG_DIRECTORY = Path(__file__).parent


def _strip_json_comments(source: str) -> str:
    """Remove JSONC comments without treating comment markers in strings as comments."""
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(source):
        character = source[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue

        if character == '"':
            in_string = True
            output.append(character)
            index += 1
        elif character == "/" and index + 1 < len(source) and source[index + 1] == "/":
            index = source.find("\n", index + 2)
            if index == -1:
                break
            output.append("\n")
            index += 1
        elif character == "/" and index + 1 < len(source) and source[index + 1] == "*":
            comment_end = source.find("*/", index + 2)
            if comment_end == -1:
                raise ValueError("Unterminated block comment in study configuration.")
            output.extend("\n" for value in source[index:comment_end] if value == "\n")
            index = comment_end + 2
        else:
            output.append(character)
            index += 1
    return "".join(output)


def _filename_component(value: str) -> str:
    """Convert an environment or study name to its stable filename component."""
    characters: list[str] = []
    for index, character in enumerate(value):
        if character.isupper() and index and value[index - 1].islower():
            characters.append("_")
        characters.append(character.lower() if character.isalnum() else "_")
    return "".join(characters).strip("_")


def study_config_path(environment: str, study: str) -> Path:
    """Return ``study_{environment}_{study}.json`` for one study."""
    return STUDY_CONFIG_DIRECTORY / (
        f"study_{_filename_component(environment)}_{_filename_component(study)}.json"
    )


def load_study_configs() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return the manual study configuration, validating its top-level shape."""
    configs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for config_path in sorted(STUDY_CONFIG_DIRECTORY.glob("study_*.json")):
        if config_path.name == "study_configs.json":
            continue
        config = json.loads(_strip_json_comments(config_path.read_text(encoding="utf-8")))
        environment = config.get("environment")
        study = config.get("study")
        runs = config.get("runs")
        if not isinstance(environment, str) or not isinstance(study, str) or not isinstance(runs, list):
            raise ValueError(f"Invalid study configuration: {config_path}")
        configs.setdefault(environment, {})[study] = runs
    return configs


def _save_study_config(
    environment: str, study: str, runs: list[dict[str, Any]]
) -> None:
    """Atomically save a single study configuration as standard JSON."""
    config_path = study_config_path(environment, study)
    temporary_path = config_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(
            {"environment": environment, "study": study, "runs": runs},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(config_path)


def save_study_configs(configs: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    """Atomically persist manually edited study configurations."""
    for environment, studies in configs.items():
        for study, runs in studies.items():
            _save_study_config(environment, study, runs)


def add_study_run(
    environment: str,
    study: str,
    run: dict[str, Any],
) -> None:
    """Add a run, rejecting an accidental duplicate path in the same study."""
    configs = load_study_configs()
    runs = configs.setdefault(environment, {}).setdefault(study, [])
    if any(existing.get("path") == run["path"] for existing in runs):
        raise ValueError(f"{run['path']} is already listed for {environment}/{study}.")
    runs.append(run)
    _save_study_config(environment, study, runs)


def remove_study_run(environment: str, study: str, index: int) -> dict[str, Any]:
    """Remove and return one run by its displayed zero-based index."""
    configs = load_study_configs()
    try:
        run = configs[environment][study].pop(index)
    except (KeyError, IndexError) as error:
        raise ValueError(f"No run {index + 1} in {environment}/{study}.") from error
    _save_study_config(environment, study, configs[environment][study])
    return run
