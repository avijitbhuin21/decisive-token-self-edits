import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from datasets import load_dataset

# Reference solution fails its own hidden suite, so the task is unscorable.
BROKEN_TASKS = {590}


@dataclass
class Task:
    task_id: int
    prompt: str
    visible_tests: list[str]
    hidden_test: str
    test_imports: list[str]
    reference: str

    def user_message(self) -> str:
        """Render the task as shown to the model: description plus visible tests only."""
        tests = "\n".join(self.visible_tests)
        return (
            f"{self.prompt}\nYour code must pass these tests:\n{tests}\n"
            "Reply with a single ```python code block containing only the function. "
            "No comments, no docstrings, no tests, no example usage."
        )


@dataclass
class RunResult:
    passed: bool
    error: str


def load_tasks(limit: int | None = None) -> list[Task]:
    """Load MBPP+ tasks with the original 3 asserts as visible tests and the plus suite as hidden."""
    ds = load_dataset("evalplus/mbppplus", split="test")
    tasks = [
        Task(r["task_id"], r["prompt"], r["test_list"], r["test"], r["test_imports"], r["code"])
        for r in ds
        if r["task_id"] not in BROKEN_TASKS
    ]
    return tasks[:limit] if limit else tasks


def extract_code(text: str) -> str:
    """Pull the last python code block out of a model reply, dropping any thinking section."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return blocks[-1]
    return text.split("```")[0]


def run_code(program: str, timeout: float = 10.0) -> RunResult:
    """Execute a program in a fresh Python subprocess and report pass/fail with a short error."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            p = subprocess.run(
                [sys.executable, path], cwd=d, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return RunResult(False, f"Timeout after {timeout}s")
    if p.returncode == 0:
        return RunResult(True, "")
    lines = p.stderr.strip().splitlines()
    return RunResult(False, "\n".join(lines[-6:]))


def run_visible(task: Task, code: str) -> RunResult:
    """Run candidate code against the visible asserts, stopping at the first failure."""
    header = "\n".join(task.test_imports)
    return run_code(f"{header}\n{code}\n\n" + "\n".join(task.visible_tests) + "\n")


def run_hidden(task: Task, code: str) -> RunResult:
    """Run candidate code against the hidden MBPP+ suite (used only for scoring)."""
    header = "\n".join(task.test_imports)
    return run_code(f"{header}\n{code}\n\n{task.hidden_test}\n", timeout=30.0)
