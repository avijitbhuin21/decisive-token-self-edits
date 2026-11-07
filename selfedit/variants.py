import random
import string
from dataclasses import dataclass
from typing import Callable

from selfedit.tasks import Task

N_VARIANTS = 15
N_HIDDEN = 25


@dataclass
class Template:
    name: str
    fn: str
    prompt: str
    params: Callable[[random.Random], dict]
    ref: Callable
    inputs: Callable[[random.Random, dict], tuple]


def ints(r, n_lo=0, n_hi=10, lo=-20, hi=50):
    """Random integer list."""
    return [r.randint(lo, hi) for _ in range(r.randint(n_lo, n_hi))]


def word(r, lo=1, hi=9):
    """Random lowercase word."""
    return "".join(r.choice(string.ascii_lowercase) for _ in range(r.randint(lo, hi)))


def sentence(r, lo=0, hi=8):
    """Random space-separated lowercase words."""
    return " ".join(word(r) for _ in range(r.randint(lo, hi)))


def text(r, lo=0, hi=20):
    """Random string of letters, digits, spaces and punctuation."""
    alphabet = string.ascii_letters + string.digits + "  .,!"
    return "".join(r.choice(alphabet) for _ in range(r.randint(lo, hi)))


def to_base(n, b):
    """Base-b representation of a non-negative integer."""
    if n == 0:
        return "0"
    s = ""
    while n:
        s = str(n % b) + s
        n //= b
    return s


def chunks(lst, k):
    """Consecutive slices of length k (last may be shorter)."""
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def caesar(s, k):
    """Shift lowercase letters forward by k, leaving other characters unchanged."""
    return "".join(chr((ord(c) - 97 + k) % 26 + 97) if c.islower() else c for c in s)


def keep_at_most(lst, k):
    """Keep at most k occurrences of each value, preserving order."""
    seen, out = {}, []
    for x in lst:
        seen[x] = seen.get(x, 0) + 1
        if seen[x] <= k:
            out.append(x)
    return out


def group_right(s, k, c):
    """Insert c between groups of k characters, counting from the right."""
    out = ""
    for i, ch in enumerate(reversed(s)):
        if i and i % k == 0:
            out = c + out
        out = ch + out
    return out


def lin_seq(n, x, y):
    """n-th term (0-indexed) of a(0)=x, a(1)=y, a(i)=a(i-1)+a(i-2)."""
    a, b = x, y
    for _ in range(n):
        a, b = b, a + b
    return a


def round_up_ties(x, k):
    """Round x to the nearest multiple of k, with ties going up."""
    q, rem = divmod(x, k)
    return (q + 1) * k if 2 * rem >= k else q * k


def k_param(lo, hi, name="k"):
    """Parameter sampler for a single integer."""
    return lambda r: {name: r.randint(lo, hi)}


def char_param(r):
    """Parameter sampler for a single punctuation-like character."""
    return {"c": r.choice("*#@$%&-_+=~")}


TEMPLATES = [
    Template("sum_div", "sum_div", "Write a function sum_div(lst) that returns the sum of the elements of lst that are divisible by {k}.",
             k_param(2, 17), lambda p, l: sum(x for x in l if x % p["k"] == 0), lambda r, p: (ints(r),)),
    Template("count_gt", "count_gt", "Write a function count_gt(lst) that returns how many elements of lst are strictly greater than {k}.",
             k_param(-10, 40), lambda p, l: sum(x > p["k"] for x in l), lambda r, p: (ints(r),)),
    Template("kth_smallest", "kth_smallest", "Write a function kth_smallest(lst) that returns the {k}-th smallest element of lst, counting from 1 (duplicates count separately).",
             k_param(2, 16), lambda p, l: sorted(l)[p["k"] - 1], lambda r, p: (ints(r, p["k"], p["k"] + 8),)),
    Template("rotate_left", "rotate_left", "Write a function rotate_left(lst) that returns lst rotated to the left by {k} positions (wrapping around if {k} exceeds the length; return [] for an empty list).",
             k_param(1, 13), lambda p, l: l[p["k"] % len(l):] + l[:p["k"] % len(l)] if l else [], lambda r, p: (ints(r),)),
    Template("every_kth", "every_kth", "Write a function every_kth(lst) that returns every {k}-th element of lst, starting with the {k}-th element (positions counted from 1).",
             k_param(2, 14), lambda p, l: l[p["k"] - 1::p["k"]], lambda r, p: (ints(r, 0, 30),)),
    Template("chunk_sums", "chunk_sums", "Write a function chunk_sums(lst) that splits lst into consecutive chunks of {k} elements (the last chunk may be shorter) and returns the list of chunk sums.",
             k_param(2, 14), lambda p, l: [sum(c) for c in chunks(l, p["k"])], lambda r, p: (ints(r, 0, 30),)),
    Template("long_words", "long_words", "Write a function long_words(s) that returns the list of words in the string s (separated by spaces) that have at least {k} characters, in their original order.",
             k_param(2, 14), lambda p, s: [w for w in s.split() if len(w) >= p["k"]], lambda r, p: (" ".join(word(r, 1, 16) for _ in range(r.randint(0, 9))),)),
    Template("caesar", "caesar", "Write a function caesar(s) that shifts every lowercase letter in s forward by {k} places in the alphabet, wrapping from z back to a, and leaves all other characters unchanged.",
             k_param(1, 25), lambda p, s: caesar(s, p["k"]), lambda r, p: (text(r),)),
    Template("repeat_chars", "repeat_chars", "Write a function repeat_chars(s) that returns s with every character repeated {k} times.",
             k_param(2, 16), lambda p, s: "".join(c * p["k"] for c in s), lambda r, p: (text(r, 0, 10),)),
    Template("pad_left", "pad_left", "Write a function pad_left(s) that pads the string s on the left with the character '{c}' until it is {k} characters long. Strings that are already at least {k} characters long are returned unchanged.",
             lambda r: {**char_param(r), "k": r.randint(4, 14)}, lambda p, s: s.rjust(p["k"], p["c"]), lambda r, p: (word(r, 0, 16),)),
    Template("sum_multiples", "sum_multiples", "Write a function sum_multiples(n) that returns the sum of all positive integers strictly less than n that are multiples of {k}.",
             k_param(2, 19), lambda p, n: sum(x for x in range(1, n) if x % p["k"] == 0), lambda r, p: (r.randint(0, 300),)),
    Template("digit_sum_base", "digit_sum_base", "Write a function digit_sum_base(n) that returns the sum of the digits of the non-negative integer n when written in base {k}.",
             k_param(2, 9), lambda p, n: sum(int(d) for d in to_base(n, p["k"])), lambda r, p: (r.randint(0, 5000),)),
    Template("to_base", "to_base", "Write a function to_base(n) that returns the base-{k} representation of the non-negative integer n as a string of digits (return '0' for 0).",
             k_param(2, 9), lambda p, n: to_base(n, p["k"]), lambda r, p: (r.randint(0, 5000),)),
    Template("replace_kth", "replace_kth", "Write a function replace_kth(s) that returns s with every {k}-th character (positions counted from 1) replaced by '{c}'.",
             lambda r: {**char_param(r), "k": r.randint(2, 6)},
             lambda p, s: "".join(p["c"] if (i + 1) % p["k"] == 0 else ch for i, ch in enumerate(s)), lambda r, p: (word(r, 0, 18),)),
    Template("clamp", "clamp", "Write a function clamp(lst) that returns a new list where every element of lst below {lo} is replaced by {lo} and every element above {hi} is replaced by {hi}.",
             lambda r: (lambda lo: {"lo": lo, "hi": lo + r.randint(3, 30)})(r.randint(-15, 20)),
             lambda p, l: [min(max(x, p["lo"]), p["hi"]) for x in l], lambda r, p: (ints(r),)),
    Template("window_sums", "window_sums", "Write a function window_sums(lst) that returns the list of sums of every contiguous window of {k} elements of lst, from left to right (an empty list if lst has fewer than {k} elements).",
             k_param(2, 14), lambda p, l: [sum(l[i:i + p["k"]]) for i in range(len(l) - p["k"] + 1)], lambda r, p: (ints(r, 0, 22),)),
    Template("lin_seq", "lin_seq", "Write a function lin_seq(n) for the sequence a(0) = {x}, a(1) = {y}, a(i) = a(i-1) + a(i-2), returning a(n).",
             lambda r: {"x": r.randint(-5, 9), "y": r.randint(1, 12)}, lambda p, n: lin_seq(n, p["x"], p["y"]), lambda r, p: (r.randint(0, 25),)),
    Template("drop_kth", "drop_kth", "Write a function drop_kth(lst) that returns lst with every {k}-th element removed (positions counted from 1).",
             k_param(2, 14), lambda p, l: [x for i, x in enumerate(l) if (i + 1) % p["k"]], lambda r, p: (ints(r, 0, 30),)),
    Template("group_digits", "group_digits", "Write a function group_digits(n) that returns the decimal digits of the non-negative integer n as a string with '{c}' inserted between groups of {k} digits, counting the groups from the right.",
             lambda r: {**char_param(r), "k": r.randint(2, 5)}, lambda p, n: group_right(str(n), p["k"], p["c"]), lambda r, p: (r.randint(0, 10 ** r.randint(1, 12)),)),
    Template("sum_last_digits", "sum_last_digits", "Write a function sum_last_digits(n) that returns the sum of the last {k} decimal digits of the non-negative integer n (all its digits if it has fewer than {k}).",
             k_param(2, 16), lambda p, n: sum(int(d) for d in str(n)[-p["k"]:]), lambda r, p: (r.randint(0, 10 ** r.randint(1, 20)),)),
    Template("count_div_or", "count_div_or", "Write a function count_div_or(n) that returns how many integers from 1 to n inclusive are divisible by {a} or by {b}.",
             lambda r: (lambda a: {"a": a, "b": r.choice([x for x in range(2, 14) if x != a])})(r.randint(2, 13)),
             lambda p, n: sum(1 for x in range(1, n + 1) if x % p["a"] == 0 or x % p["b"] == 0), lambda r, p: (r.randint(0, 400),)),
    Template("first_words", "first_words", "Write a function first_words(s) that returns a string made of the first {k} words of s (words are separated by spaces), joined by single spaces.",
             k_param(2, 16), lambda p, s: " ".join(s.split()[:p["k"]]), lambda r, p: (sentence(r, 0, 20),)),
    Template("keep_at_most", "keep_at_most", "Write a function keep_at_most(lst) that returns lst keeping only the first {k} occurrences of each value, in the original order.",
             k_param(1, 15), lambda p, l: keep_at_most(l, p["k"]), lambda r, p: ([r.randint(0, 3) for _ in range(r.randint(0, 40))],)),
    Template("top_k", "top_k", "Write a function top_k(lst) that returns the {k} largest distinct values of lst in descending order (all distinct values, descending, if there are fewer than {k}).",
             k_param(2, 16), lambda p, l: sorted(set(l), reverse=True)[:p["k"]], lambda r, p: (ints(r, 0, 24, -15, 25),)),
    Template("pair_sum", "pair_sum", "Write a function pair_sum(lst) that returns the number of index pairs (i, j) with i < j such that lst[i] + lst[j] == {t}.",
             k_param(-5, 20, "t"), lambda p, l: sum(l[i] + l[j] == p["t"] for i in range(len(l)) for j in range(i + 1, len(l))),
             lambda r, p: (ints(r, 0, 12, -5, 15),)),
    Template("power_sum", "power_sum", "Write a function power_sum(lst) that returns the sum of each element of lst raised to the power {k}.",
             k_param(2, 16), lambda p, l: sum(x ** p["k"] for x in l), lambda r, p: (ints(r, 0, 6, -4, 4),)),
    Template("round_to", "round_to", "Write a function round_to(lst) that rounds every integer in lst to the nearest multiple of {k}, rounding ties up, and returns the new list.",
             k_param(2, 16), lambda p, l: [round_up_ties(x, p["k"]) for x in l], lambda r, p: (ints(r),)),
    Template("vowel_replace", "vowel_replace", "Write a function vowel_replace(s) that replaces every vowel (a, e, i, o, u, upper or lower case) in s with '{c}'.",
             char_param, lambda p, s: "".join(p["c"] if ch in "aeiouAEIOU" else ch for ch in s), lambda r, p: (text(r),)),
    Template("bits_k", "bits_k", "Write a function bits_k(lst) that returns how many non-negative integers in lst have exactly {k} bits set to 1 in binary.",
             k_param(1, 12), lambda p, l: sum(bin(x).count("1") == p["k"] for x in l), lambda r, p: (ints(r, 0, 12, 0, 8191),)),
    Template("move_to_end", "move_to_end", "Write a function move_to_end(lst) that returns lst with every occurrence of {v} moved to the end, keeping the order of the other elements.",
             k_param(0, 19, "v"), lambda p, l: [x for x in l if x != p["v"]] + [x for x in l if x == p["v"]],
             lambda r, p: ([r.randint(0, 19) for _ in range(r.randint(0, 14))],)),
    Template("coprime_k", "coprime_k", "Write a function coprime_k(lst) that returns the elements of lst whose greatest common divisor with {k} is 1, in their original order.",
             k_param(4, 30), lambda p, l: [x for x in l if __import__("math").gcd(x, p["k"]) == 1], lambda r, p: (ints(r, 0, 12, 1, 60),)),
    Template("div_a_not_b", "div_a_not_b", "Write a function div_a_not_b(n) that returns how many integers from 1 to n inclusive are divisible by {a} but not by {b}.",
             lambda r: (lambda a: {"a": a, "b": a * r.randint(2, 4)})(r.randint(2, 9)),
             lambda p, n: sum(1 for x in range(1, n + 1) if x % p["a"] == 0 and x % p["b"]), lambda r, p: (r.randint(0, 400),)),
]


def _assert(t: Template, p: dict, args: tuple) -> str:
    """One assert line comparing the function call to the reference output."""
    call = f"{t.fn}({', '.join(repr(a) for a in args)})"
    return f"assert {call} == {t.ref(p, *args)!r}"


def variant_params(t: Template, seed: int = 0) -> list[dict]:
    """Up to N_VARIANTS distinct parameter settings for a template, deterministic in seed."""
    r = random.Random(f"{t.name}-{seed}")
    out = []
    for _ in range(500):
        p = t.params(r)
        if p not in out:
            out.append(p)
        if len(out) == N_VARIANTS:
            break
    return out


def build_task(ti: int, vi: int, t: Template, p: dict) -> Task:
    """Concrete Task for template ti, variant vi with params p (3 visible asserts, hidden suite)."""
    r = random.Random(f"{t.name}-{vi}-inputs")
    seen, inputs = set(), []
    for _ in range(2000):
        args = t.inputs(r, p)
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        inputs.append(args)
        if len(inputs) == 3 + N_HIDDEN:
            break
    visible = sorted(inputs[:12], key=lambda a: -len(repr(a)))[:3]
    hidden = [a for a in inputs if a not in visible]
    return Task(task_id=ti * 100 + vi, prompt=t.prompt.format(**p),
                visible_tests=[_assert(t, p, a) for a in visible],
                hidden_test="\n".join(_assert(t, p, a) for a in visible + hidden),
                test_imports=[], reference="")


def load_variants(names: list[str] | None = None) -> dict[str, list[Task]]:
    """All variant Tasks grouped by template name (optionally only the named templates)."""
    out = {}
    for ti, t in enumerate(TEMPLATES):
        if names and t.name not in names:
            continue
        out[t.name] = [build_task(ti, vi, t, p) for vi, p in enumerate(variant_params(t))]
    return out
