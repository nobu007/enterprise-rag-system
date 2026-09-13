"""
Regression guard against orphaned CI workflows

The 2026-09-13 slim-down (ae1fd31) removed the Dockerfile but left
.github/workflows/docker.yml behind, so every push to main ran
`docker build .` against a tree with no Dockerfile — a guaranteed-red
job (bf05f21 removed the workflow). These tests keep that class of
drift out of CI:

- the workflows directory keeps at least one workflow (test.yml is the
  verification of record named by .concept/autopilot.yml), and
- any workflow that builds a container image must reference a
  Dockerfile that actually exists in the tree, whether via `-f PATH`,
  the docker/build-push-action `file:` input, or the implicit
  repo-root default.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Matched per line; a comment mentioning "docker build" also trips the
# guard, which is the safe direction for a consistency check.
DOCKER_BUILD_LINE_RE = re.compile(r"\bdocker\s+build\b")
DOCKERFILE_FLAG_RE = re.compile(r"(?:^|\s)(?:-f|--file)\s+(\S+)")
BUILD_PUSH_ACTION = "docker/build-push-action"
FILE_INPUT_RE = re.compile(
    r"^\s*(?:-\s+)?file:\s*['\"]?([^'\"\s]+)", re.MULTILINE
)
CONTEXT_INPUT_RE = re.compile(
    r"^\s*(?:-\s+)?context:\s*['\"]?([^'\"\s]+)", re.MULTILINE
)


def _workflow_files():
    if not WORKFLOWS_DIR.is_dir():
        return []
    return sorted(
        p for p in WORKFLOWS_DIR.iterdir() if p.suffix in (".yml", ".yaml")
    )


def _dockerfile_candidates(text):
    """Dockerfile paths a workflow's image builds would resolve to.

    Returns repo-root-relative paths, including implicit defaults: a bare
    `docker build .` and a docker/build-push-action step without a `file:`
    input both build <context>/Dockerfile. Unresolvable expressions
    (${{ ... }}) and remote refs are ignored — they cannot be validated
    statically.
    """
    candidates = []
    if BUILD_PUSH_ACTION in text:
        context_match = CONTEXT_INPUT_RE.search(text)
        context = context_match.group(1) if context_match else "."
        file_inputs = FILE_INPUT_RE.findall(text)
        if file_inputs:
            candidates.extend(file_inputs)
        elif context == ".":
            candidates.append("Dockerfile")
        else:
            candidates.append(f"{context}/Dockerfile")
    for line in text.splitlines():
        if DOCKER_BUILD_LINE_RE.search(line):
            flag = DOCKERFILE_FLAG_RE.search(line)
            candidates.append(flag.group(1) if flag else "Dockerfile")
    return candidates


def _statically_checkable(path):
    return not (
        path.startswith("${{")
        or path.startswith("http://")
        or path.startswith("https://")
        or "$" in path
    )


class TestCiWorkflowsReferenceRealBuildContexts:
    def test_workflows_directory_keeps_a_workflow(self):
        files = _workflow_files()
        assert files, (
            "no GitHub workflow found; test.yml is the verification of record "
            "named by .concept/autopilot.yml"
        )

    def test_docker_builds_reference_existing_dockerfiles(self):
        for workflow in _workflow_files():
            text = workflow.read_text(encoding="utf-8")
            for candidate in _dockerfile_candidates(text):
                assert _statically_checkable(candidate), (
                    f"{workflow.name}: Dockerfile reference '{candidate}' "
                    "is not statically checkable; pin it to a concrete path"
                )
                dockerfile = REPO_ROOT / candidate
                assert dockerfile.is_file(), (
                    f"{workflow.name} builds an image from '{candidate}' "
                    "but that Dockerfile does not exist in the tree "
                    "(the bf05f21 failure mode: a workflow whose job "
                    "can never succeed)"
                )


class TestDockerfileCandidateExtraction:
    def test_bare_docker_build_defaults_to_root_dockerfile(self):
        text = "- name: Build image\n  run: docker build -t app:latest .\n"
        assert _dockerfile_candidates(text) == ["Dockerfile"]

    def test_docker_build_file_flag_uses_referenced_path(self):
        text = "run: docker build --file deploy/Dockerfile -t app:latest .\n"
        assert _dockerfile_candidates(text) == ["deploy/Dockerfile"]

    def test_build_push_action_file_input_context_and_default(self):
        with_file = (
            "uses: docker/build-push-action@v5\n"
            "with:\n"
            "  context: ./app\n"
            "  file: ./app/Dockerfile.ci\n"
        )
        assert _dockerfile_candidates(with_file) == ["./app/Dockerfile.ci"]

        context_only = (
            "uses: docker/build-push-action@v5\n"
            "with:\n"
            "  context: ./app\n"
        )
        assert _dockerfile_candidates(context_only) == ["./app/Dockerfile"]

        no_context = "uses: docker/build-push-action@v5\n"
        assert _dockerfile_candidates(no_context) == ["Dockerfile"]

    def test_non_docker_workflows_yield_no_candidates(self):
        text = (
            "name: Tests\n"
            "on: [push]\n"
            "jobs:\n"
            "  test:\n"
            "    steps:\n"
            "      - run: pytest tests/ -v\n"
        )
        assert _dockerfile_candidates(text) == []


@pytest.mark.parametrize(
    "path,checkable",
    [
        ("Dockerfile", True),
        ("deploy/Dockerfile", True),
        ("${{ inputs.dockerfile }}", False),
        ("https://example.com/Dockerfile", False),
        ("$DOCKERFILE_PATH", False),
    ],
)
def test_unresolvable_references_are_reported(path, checkable):
    assert _statically_checkable(path) is checkable
