"""
evaluator plugin — Multi-choice benchmark utility.

Loads JSONL benchmark files from plugins/evaluator/tests/, formats each item
as a multi-choice prompt, sends it to the current chat endpoint with
deterministic sampling, parses the model's reply for a letter answer, and
tallies a score. Results land as JSON in plugins/evaluator/results/.

Designed to be a full-service, expandable benchmarking utility:
  - Drop any number of .jsonl test files into tests/, no registration step
  - Run one or all of them with a single slash command
  - Category-level breakdowns when items carry a "category" field
  - Partial results saved on Ctrl-C
  - Optional metadata header per test (one-line _meta record)

Usage in chat:
  /evaluator                          show help (and tests dir location)
  /evaluator list                     list test files in tests/
  /evaluator info <test>              show metadata + sample question
  /evaluator run <test> [flags]       run one test
  /evaluator run-all [flags]          run every .jsonl in tests/
  /evaluator results                  list recent result files
  /evaluator results <test>           show the latest result for a test

Run flags:
  --limit N         stop after N questions
  --temp F          temperature override (default 0.0 = greedy)
  --max-new N       max_new_tokens override (default 20)
  --no-save         do not write a result file
  --verbose         print each question's outcome inline

Test file format (JSONL, one record per line):
  Optional header (first line, ignored for scoring):
    {"_meta": true, "name": "My Eval", "description": "..."}

  Each question:
    {"id": "q1", "question": "What is 7 x 8?",
     "choices": ["54", "55", "56", "63"],
     "answer": "C",
     "category": "math"}

  - choices: a list of 2-26 strings
  - answer:  a letter "A"..."Z" or a 0-indexed integer
  - id / category: optional but recommended
"""

import os
import re
import json
import time
import argparse
import urllib.request


# ──────────────────────────────────────────────────────────
#  Paths — resolved relative to this plugin file so the layout
#  follows the user's --plugins-dir wherever it is.
# ──────────────────────────────────────────────────────────

_PLUGIN_FILE = os.path.abspath(__file__)
_PLUGIN_DIR = os.path.dirname(_PLUGIN_FILE)
TESTS_DIR = os.path.join(_PLUGIN_DIR, "evaluator", "tests")
RESULTS_DIR = os.path.join(_PLUGIN_DIR, "evaluator", "results")


# ──────────────────────────────────────────────────────────
#  Example test seeded on first launch (only when tests/ is empty)
# ──────────────────────────────────────────────────────────

EXAMPLE_TEST_NAME = "example.jsonl"
EXAMPLE_TEST_RECORDS = [
    {"_meta": True, "name": "Example Mini-Eval",
     "description": "Five questions across categories. Replace or delete this file once you have your own."},
    {"id": "geo1", "category": "geography",
     "question": "What is the capital of France?",
     "choices": ["Berlin", "Paris", "Madrid", "Rome"],
     "answer": "B"},
    {"id": "math1", "category": "math",
     "question": "What is 7 multiplied by 8?",
     "choices": ["54", "55", "56", "63"],
     "answer": "C"},
    {"id": "sci1", "category": "science",
     "question": "Which planet is closest to the sun?",
     "choices": ["Venus", "Earth", "Mercury", "Mars"],
     "answer": "C"},
    {"id": "lang1", "category": "language",
     "question": "Which of the following is a verb?",
     "choices": ["happy", "quickly", "run", "blue"],
     "answer": "C"},
    {"id": "hist1", "category": "history",
     "question": "In what year did World War II end in Europe?",
     "choices": ["1942", "1945", "1948", "1950"],
     "answer": "B"},
]


# ──────────────────────────────────────────────────────────
#  Prompt template — kept simple. Edit here to experiment.
# ──────────────────────────────────────────────────────────

DEFAULT_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "{choices_block}\n\n"
    "Answer with just the letter of the correct choice."
)


# ──────────────────────────────────────────────────────────
#  Plugin
# ──────────────────────────────────────────────────────────


class EvaluatorPlugin(Plugin):  # noqa: F821 — Plugin is injected by the loader
    name = "evaluator"
    description = (
        "Multi-choice benchmark suite. Loads JSONL tests from "
        "plugins/evaluator/tests/ and scores the current model. "
        "Type /evaluator for usage."
    )
    commands = ["/evaluator", "/eval"]

    # ──────────────────────────────────────────────────────
    #  Lifecycle
    # ──────────────────────────────────────────────────────

    def on_load(self, ctx) -> None:
        """Create tests/ and results/ on first launch. Seed an example test
        ONLY if no .jsonl tests exist yet (so we never overwrite the user's
        data on /plugin reload).
        """
        try:
            os.makedirs(TESTS_DIR, exist_ok=True)
            os.makedirs(RESULTS_DIR, exist_ok=True)
            existing = [f for f in os.listdir(TESTS_DIR) if f.endswith(".jsonl")]
            if not existing:
                example_path = os.path.join(TESTS_DIR, EXAMPLE_TEST_NAME)
                with open(example_path, "w", encoding="utf-8") as f:
                    for rec in EXAMPLE_TEST_RECORDS:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            # Never break the host on a setup error
            pass

    def help_text(self) -> str:
        return (
            "Run multi-choice benchmark tests against the current model.\n"
            "\n"
            "Usage:\n"
            "  /evaluator                         show this help\n"
            "  /evaluator list                    list test files in tests/\n"
            "  /evaluator info <test>             show metadata + sample question\n"
            "  /evaluator run <test> [flags]      run one test\n"
            "  /evaluator run-all [flags]         run every test in tests/\n"
            "  /evaluator results                 list recent result files\n"
            "  /evaluator results <test>          show latest result for one test\n"
            "\n"
            "Run flags:\n"
            "  --limit N          stop after N questions\n"
            "  --temp F           temperature override (default 0.0 = greedy)\n"
            "  --max-new N        max_new_tokens override (default 20)\n"
            "  --no-save          skip writing a result file\n"
            "  --verbose          print each question's outcome\n"
            "\n"
            f"  Tests directory:   {TESTS_DIR}\n"
            f"  Results directory: {RESULTS_DIR}\n"
            "\n"
            "Test file format (JSONL, one question per line):\n"
            "  {\"id\":\"q1\", \"question\":\"...\", \"choices\":[...],\n"
            "   \"answer\":\"B\", \"category\":\"optional\"}\n"
            "\n"
            "Optional first line is a metadata header:\n"
            "  {\"_meta\":true, \"name\":\"My Eval\", \"description\":\"...\"}\n"
        )

    def handle(self, cmd, args, ctx) -> None:
        if not args or args[0] in ("help", "-h", "--help"):
            ctx.print(self.help_text(), color=Color.DIM)  # noqa: F821
            return

        action = args[0]
        rest = args[1:]

        if action == "list":
            self._list_tests(ctx)
        elif action == "info":
            self._info(rest, ctx)
        elif action == "run":
            self._run_one(rest, ctx)
        elif action == "run-all":
            self._run_all(rest, ctx)
        elif action == "results":
            self._results(rest, ctx)
        else:
            ctx.print(f"  Unknown subcommand: {action!r}. Try /evaluator help.",
                      color=Color.YELLOW)  # noqa: F821

    # ──────────────────────────────────────────────────────
    #  Subcommands
    # ──────────────────────────────────────────────────────

    def _list_tests(self, ctx) -> None:
        files = self._discover_tests()
        if not files:
            ctx.print(f"  No tests in {TESTS_DIR}", color=Color.YELLOW)  # noqa: F821
            ctx.print(f"  Drop a .jsonl file there to register it.",
                      color=Color.DIM)  # noqa: F821
            return
        ctx.print(f"  Found {len(files)} test(s) in {TESTS_DIR}:",
                  color=Color.DIM)  # noqa: F821
        for f in files:
            path = os.path.join(TESTS_DIR, f)
            meta, items, err = self._load_test(path)
            if err:
                ctx.print(f"    {f}  [load error: {err}]",
                          color=Color.YELLOW)  # noqa: F821
                continue
            display_name = meta.get("name") or f
            ctx.print(f"    {f}", color=Color.DIM)  # noqa: F821
            ctx.print(f"      {display_name}  ({len(items)} questions)",
                      color=Color.DIM)  # noqa: F821
            if meta.get("description"):
                ctx.print(f"      {meta['description']}",
                          color=Color.DIM)  # noqa: F821

    def _info(self, args, ctx) -> None:
        if not args:
            ctx.print("  Usage: /evaluator info <test.jsonl>",
                      color=Color.YELLOW)  # noqa: F821
            return
        path = self._resolve_test_path(args[0])
        if path is None:
            ctx.print(f"  Test not found: {args[0]}",
                      color=Color.RED)  # noqa: F821
            return
        meta, items, err = self._load_test(path)
        if err:
            ctx.print(f"  Load error: {err}", color=Color.RED)  # noqa: F821
            return
        ctx.print(f"  File:        {os.path.basename(path)}",
                  color=Color.DIM)  # noqa: F821
        if meta.get("name"):
            ctx.print(f"  Name:        {meta['name']}",
                      color=Color.DIM)  # noqa: F821
        if meta.get("description"):
            ctx.print(f"  Description: {meta['description']}",
                      color=Color.DIM)  # noqa: F821
        ctx.print(f"  Questions:   {len(items)}",
                  color=Color.DIM)  # noqa: F821

        cats = {}
        for it in items:
            c = it.get("category", "(none)")
            cats[c] = cats.get(c, 0) + 1
        if len(cats) > 1 or (cats and "(none)" not in cats):
            ctx.print("  Categories:", color=Color.DIM)  # noqa: F821
            for c, n in sorted(cats.items()):
                ctx.print(f"    {c}: {n}", color=Color.DIM)  # noqa: F821

        if items:
            sample = items[0]
            ctx.print("  Sample:", color=Color.DIM)  # noqa: F821
            ctx.print(f"    Q: {sample['question']}",
                      color=Color.DIM)  # noqa: F821
            for i, c in enumerate(sample["choices"]):
                ctx.print(f"    {chr(ord('A') + i)}) {c}",
                          color=Color.DIM)  # noqa: F821
            ctx.print(
                f"    Answer: "
                f"{self._normalize_answer(sample['answer'], len(sample['choices']))}",
                color=Color.DIM,  # noqa: F821
            )

    def _run_one(self, args, ctx) -> None:
        ns = self._parse_run_args(args, multi=False)
        if ns is None:
            return
        path = self._resolve_test_path(ns.test)
        if path is None:
            ctx.print(f"  Test not found: {ns.test}",
                      color=Color.RED)  # noqa: F821
            return
        self._execute_test(path, ns, ctx)

    def _run_all(self, args, ctx) -> None:
        ns = self._parse_run_args(args, multi=True)
        if ns is None:
            return
        files = self._discover_tests()
        if not files:
            ctx.print(f"  No tests in {TESTS_DIR}",
                      color=Color.YELLOW)  # noqa: F821
            return
        ctx.print(f"  Running {len(files)} test(s)...\n",
                  color=Color.DIM)  # noqa: F821

        summaries = []
        try:
            for i, f in enumerate(files, 1):
                path = os.path.join(TESTS_DIR, f)
                ctx.print(f"  [test {i}/{len(files)}] {f}",
                          color=Color.CYAN)  # noqa: F821
                res = self._execute_test(path, ns, ctx)
                if res:
                    summaries.append(res)
                ctx.print("", color=Color.DIM)  # noqa: F821
        except KeyboardInterrupt:
            ctx.print("  Suite interrupted.",
                      color=Color.YELLOW)  # noqa: F821

        if summaries:
            ctx.print("  === Suite summary ===",
                      color=Color.GREEN)  # noqa: F821
            for s in summaries:
                if s["total"] == 0:
                    continue
                ctx.print(
                    f"    {s['test']:<32}  {s['correct']:>3}/{s['total']:<3}  "
                    f"({s['accuracy'] * 100:5.1f}%)",
                    color=Color.DIM,  # noqa: F821
                )
            total_q = sum(s["total"] for s in summaries)
            total_c = sum(s["correct"] for s in summaries)
            if total_q > 0:
                ctx.print(
                    f"    {'TOTAL':<32}  {total_c:>3}/{total_q:<3}  "
                    f"({total_c / total_q * 100:5.1f}%)",
                    color=Color.GREEN,  # noqa: F821
                )

    def _results(self, args, ctx) -> None:
        try:
            files = sorted(
                (f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")),
                reverse=True,
            )
        except OSError:
            files = []
        if not files:
            ctx.print(f"  No results yet in {RESULTS_DIR}",
                      color=Color.YELLOW)  # noqa: F821
            return

        if args:
            test_base = args[0]
            if test_base.endswith(".jsonl"):
                test_base = test_base[:-len(".jsonl")]
            match = next((f for f in files if f.startswith(test_base + "_")), None)
            if match is None:
                ctx.print(f"  No results found for {args[0]!r}",
                          color=Color.YELLOW)  # noqa: F821
                return
            self._show_result_file(os.path.join(RESULTS_DIR, match), ctx)
            return

        ctx.print(f"  Recent results in {RESULTS_DIR}:",
                  color=Color.DIM)  # noqa: F821
        for f in files[:15]:
            try:
                with open(os.path.join(RESULTS_DIR, f), "r", encoding="utf-8") as h:
                    data = json.load(h)
                acc = data.get("accuracy", 0) * 100
                ctx.print(
                    f"    {f}  {data.get('correct', 0)}/{data.get('total', 0)}  "
                    f"({acc:.1f}%)  model={data.get('model', '?')}",
                    color=Color.DIM,  # noqa: F821
                )
            except Exception:
                ctx.print(f"    {f}  (unreadable)",
                          color=Color.YELLOW)  # noqa: F821

    def _show_result_file(self, path, ctx) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            ctx.print(f"  Couldn't read {path}: {e}",
                      color=Color.RED)  # noqa: F821
            return
        ctx.print(f"  File:      {os.path.basename(path)}",
                  color=Color.DIM)  # noqa: F821
        ctx.print(f"  Test:      {data.get('test', '?')}",
                  color=Color.DIM)  # noqa: F821
        ctx.print(f"  Model:     {data.get('model', '?')}",
                  color=Color.DIM)  # noqa: F821
        ctx.print(f"  Endpoint:  {data.get('endpoint', '?')}",
                  color=Color.DIM)  # noqa: F821
        ctx.print(f"  When:      {data.get('timestamp', '?')}",
                  color=Color.DIM)  # noqa: F821
        ctx.print(
            f"  Score:     {data.get('correct', 0)}/{data.get('total', 0)}  "
            f"({data.get('accuracy', 0) * 100:.1f}%)",
            color=Color.GREEN,  # noqa: F821
        )
        if data.get("parse_errors"):
            ctx.print(f"  Parse err: {data['parse_errors']}",
                      color=Color.YELLOW)  # noqa: F821
        by_cat = data.get("by_category") or {}
        if len(by_cat) > 1:
            ctx.print("  By category:", color=Color.DIM)  # noqa: F821
            for c, stats in sorted(by_cat.items()):
                ctx.print(
                    f"    {c:<20} {stats['correct']}/{stats['total']} "
                    f"({stats['accuracy'] * 100:.1f}%)",
                    color=Color.DIM,  # noqa: F821
                )

    # ──────────────────────────────────────────────────────
    #  Core execution
    # ──────────────────────────────────────────────────────

    def _execute_test(self, path, ns, ctx):
        meta, items, err = self._load_test(path)
        test_name = os.path.basename(path)

        if err:
            ctx.print(f"  Load error ({test_name}): {err}",
                      color=Color.RED)  # noqa: F821
            return None
        if not items:
            ctx.print(f"  {test_name}: no questions to run.",
                      color=Color.YELLOW)  # noqa: F821
            return None

        work = items if ns.limit is None else items[:ns.limit]

        # Eval-specific sampling override. We use top_k=1 alongside temp=0
        # so we get hard-greedy decoding even if the server treats 0.0 as
        # a small epsilon.
        override = {
            "temperature": ns.temp,
            "top_p": 1.0,
            "top_k": 1 if ns.temp == 0.0 else 0,
            "max_new_tokens": ns.max_new,
            "repetition_penalty": 1.0,
            "no_repeat_ngram": 0,
        }

        model_name = self._fetch_model_name(ctx)

        header = test_name
        if meta.get("name"):
            header += f" ({meta['name']})"
        ctx.print(f"  Test:     {header}", color=Color.DIM)  # noqa: F821
        ctx.print(f"  Model:    {model_name}", color=Color.DIM)  # noqa: F821
        ctx.print(f"  N:        {len(work)}", color=Color.DIM)  # noqa: F821
        ctx.print(f"  Sampling: temp={ns.temp} max_new={ns.max_new}",
                  color=Color.DIM)  # noqa: F821
        ctx.print(f"  Press Ctrl-C to stop. Partial results are saved.\n",
                  color=Color.DIM)  # noqa: F821

        details = []
        correct = 0
        parse_errors = 0
        by_cat = {}
        t_start = time.time()
        interrupted = False
        last_i = 0

        try:
            for i, q in enumerate(work, 1):
                last_i = i
                num_choices = len(q["choices"])
                expected = self._normalize_answer(q["answer"], num_choices)
                prompt = self._format_prompt(q)

                reply = ctx.chat(prompt, history=[], sampling_override=override)
                predicted = self._extract_letter(reply or "", num_choices, q["choices"])
                is_correct = (predicted is not None and predicted == expected)

                if is_correct:
                    correct += 1
                if predicted is None:
                    parse_errors += 1

                cat = q.get("category", "(none)")
                slot = by_cat.setdefault(cat, {"total": 0, "correct": 0})
                slot["total"] += 1
                if is_correct:
                    slot["correct"] += 1

                details.append({
                    "id": q.get("id", f"q{i}"),
                    "category": q.get("category"),
                    "question": q["question"],
                    "choices": q["choices"],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct,
                    "raw_response": reply or "",
                })

                if ns.verbose:
                    mark = "[+]" if is_correct else ("[?]" if predicted is None else "[-]")
                    color = (Color.GREEN if is_correct  # noqa: F821
                             else (Color.YELLOW if predicted is None  # noqa: F821
                                   else Color.RED))  # noqa: F821
                    q_preview = q["question"][:60]
                    if len(q["question"]) > 60:
                        q_preview += "..."
                    ctx.print(
                        f"  [{i:>3}/{len(work)}] {mark} "
                        f"expected={expected} predicted={str(predicted):>4}  "
                        f"{q_preview}",
                        color=color,
                    )
                else:
                    # Rolling single-line progress. \r jumps to column 0,
                    # \x1b[K clears to end of line so shorter lines don't
                    # leave trailing chars from a previous update.
                    elapsed = time.time() - t_start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_sec = (len(work) - i) / rate if rate > 0 else 0
                    eta_str = f"{int(eta_sec // 60)}:{int(eta_sec % 60):02d}"
                    acc_so_far = correct / i
                    line = (f"  {i}/{len(work)}  "
                            f"{correct} correct ({acc_so_far * 100:.1f}%)  "
                            f"{rate:.1f} q/s  ETA {eta_str}")
                    print("\r" + Color.DIM + line + Color.RESET + "\x1b[K",
                          end="", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            ctx.print(f"\n  Interrupted at question {last_i}/{len(work)}. "
                      "Saving partial results.",
                      color=Color.YELLOW)  # noqa: F821

        elapsed = time.time() - t_start
        total = len(details)
        accuracy = (correct / total) if total > 0 else 0.0
        answered = total - parse_errors
        accuracy_parsed = (correct / answered) if answered > 0 else 0.0

        for cat, slot in by_cat.items():
            slot["accuracy"] = slot["correct"] / slot["total"] if slot["total"] > 0 else 0.0

        result = {
            "test": test_name,
            "test_name": meta.get("name"),
            "model": model_name,
            "endpoint": ctx.server_url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": override,
            "prompt_template": DEFAULT_PROMPT_TEMPLATE,
            "total": total,
            "answered": answered,
            "correct": correct,
            "incorrect": total - correct - parse_errors,
            "parse_errors": parse_errors,
            "accuracy": accuracy,
            "accuracy_parsed_only": accuracy_parsed,
            "elapsed_seconds": round(elapsed, 2),
            "by_category": by_cat,
            "interrupted": interrupted,
            "details": details,
        }

        # On-screen summary
        ctx.print("", color=Color.DIM)  # noqa: F821
        ctx.print(f"  === {test_name} complete ===",
                  color=Color.GREEN)  # noqa: F821
        ctx.print(f"  Score:     {correct}/{total}  ({accuracy * 100:.1f}%)",
                  color=Color.GREEN)  # noqa: F821
        if parse_errors:
            ctx.print(
                f"  Parse err: {parse_errors}  "
                f"(accuracy on parsed only: {accuracy_parsed * 100:.1f}%)",
                color=Color.YELLOW,  # noqa: F821
            )
        if len(by_cat) > 1:
            for cat, slot in sorted(by_cat.items()):
                ctx.print(
                    f"    {cat:<20} {slot['correct']}/{slot['total']} "
                    f"({slot['accuracy'] * 100:.1f}%)",
                    color=Color.DIM,  # noqa: F821
                )
        rate = len(work) / max(0.01, elapsed)
        ctx.print(f"  Wall time: {elapsed:.1f}s  ({rate:.1f} q/s)",
                  color=Color.DIM)  # noqa: F821

        # Persist
        if not ns.no_save and total > 0:
            out_path = self._save_result(test_name, result, ctx)
            if out_path:
                ctx.print(f"  Saved:     {out_path}",
                          color=Color.DIM)  # noqa: F821

        return {
            "test": test_name,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
        }

    # ──────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────

    def _discover_tests(self):
        try:
            return sorted(f for f in os.listdir(TESTS_DIR) if f.endswith(".jsonl"))
        except OSError:
            return []

    def _resolve_test_path(self, name):
        """Accept 'foo', 'foo.jsonl', or an absolute path."""
        if os.path.isabs(name) and os.path.exists(name):
            return name
        if not name.endswith(".jsonl"):
            name = name + ".jsonl"
        path = os.path.join(TESTS_DIR, name)
        return path if os.path.exists(path) else None

    def _parse_run_args(self, args, multi=False):
        parser = argparse.ArgumentParser(
            prog="/evaluator run" + ("-all" if multi else ""),
            add_help=False,
        )
        if not multi:
            parser.add_argument("test")
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--temp", type=float, default=0.0)
        parser.add_argument("--max-new", type=int, default=20, dest="max_new")
        parser.add_argument("--no-save", action="store_true", dest="no_save")
        parser.add_argument("--verbose", action="store_true")
        try:
            return parser.parse_args(args)
        except SystemExit:
            # argparse exits on bad args; keep the chat alive
            return None

    def _load_test(self, path):
        """Returns (meta_dict, items_list, error_str_or_None)."""
        if not os.path.exists(path):
            return {}, [], "file not found"
        meta = {}
        items = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_no, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError as e:
                        return {}, [], f"line {line_no} JSON: {e}"
                    if obj.get("_meta"):
                        meta = obj
                        continue
                    q = obj.get("question")
                    choices = obj.get("choices")
                    ans = obj.get("answer")
                    if not isinstance(q, str) or not q.strip():
                        return {}, [], f"line {line_no} missing/invalid 'question'"
                    if not isinstance(choices, list) or len(choices) < 2:
                        return {}, [], f"line {line_no} 'choices' must be a list of 2+"
                    if len(choices) > 26:
                        return {}, [], f"line {line_no} too many choices (max 26)"
                    if ans is None:
                        return {}, [], f"line {line_no} missing 'answer'"
                    try:
                        self._normalize_answer(ans, len(choices))
                    except ValueError as e:
                        return {}, [], f"line {line_no} {e}"
                    items.append(obj)
        except OSError as e:
            return {}, [], str(e)
        return meta, items, None

    def _normalize_answer(self, ans, num_choices):
        """Return canonical uppercase letter ('A'..'Z')."""
        max_letter = chr(ord('A') + num_choices - 1)
        if isinstance(ans, bool):
            # bool is a subclass of int; reject before isinstance(ans, int)
            raise ValueError(f"answer cannot be a bool ({ans!r})")
        if isinstance(ans, int):
            if 0 <= ans < num_choices:
                return chr(ord('A') + ans)
            raise ValueError(f"answer index {ans} out of range 0..{num_choices - 1}")
        if isinstance(ans, str):
            s = ans.strip().upper()
            if len(s) == 1 and 'A' <= s <= max_letter:
                return s
            m = re.fullmatch(r'\(?([A-Z])\)?\.?', s)
            if m and 'A' <= m.group(1) <= max_letter:
                return m.group(1)
        raise ValueError(f"invalid answer {ans!r} for {num_choices} choices")

    def _format_prompt(self, q):
        choices_block = "\n".join(
            f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(q["choices"])
        )
        return DEFAULT_PROMPT_TEMPLATE.format(
            question=q["question"],
            choices_block=choices_block,
        )

    def _extract_letter(self, text, num_choices, choices=None):
        """Pull a letter (A..max) from the model reply. Returns None if none found.

        Strategy (in order):
          1-3. Strong letter patterns ("(A)", "Answer: A", reply starts with "A)")
          4.   Choice-TEXT matching — if the reply mentions one of the choice
               strings (e.g. "Paris" for choice B), return that letter.
               When multiple choice texts appear, pick the one mentioned first.
          5-6. Weaker letter patterns (any standalone letter, then first char)

        The choice-text step is what lets us score chat-trained models that
        answer with the choice's text instead of its letter.
        """
        if not text:
            return None
        max_letter = chr(ord('A') + num_choices - 1)
        pat = f'[A-{max_letter}]'
        flags = re.IGNORECASE

        # 1. Parenthesized: "(A)"
        m = re.search(r'\((' + pat + r')\)', text, flags)
        if m:
            return m.group(1).upper()
        # 2. "answer: A" / "answer is A"
        m = re.search(r'answer\s*(?:is)?\s*[:\-]?\s*(' + pat + r')\b', text, flags)
        if m:
            return m.group(1).upper()
        # 3. Reply starts with the letter: "A)", "A.", "A,"
        m = re.match(r'^\s*(' + pat + r')[\s.,):]', text, flags)
        if m:
            return m.group(1).upper()

        # 4. Choice-text fallback. Match each choice string against the reply
        #    (case-insensitive, word-boundary aware for alphanumeric edges).
        #    Earliest mentioned choice wins.
        if choices:
            matches = []
            for i, choice in enumerate(choices):
                pos = self._find_choice_position(text, str(choice))
                if pos is not None:
                    matches.append((pos, chr(ord('A') + i)))
            if matches:
                matches.sort()
                return matches[0][1]

        # 5. Any standalone letter (word-boundary)
        m = re.search(r'\b(' + pat + r')\b', text, flags)
        if m:
            return m.group(1).upper()
        # 6. Last-ditch: first non-whitespace char
        stripped = text.strip()
        if stripped:
            first = stripped[0].upper()
            if 'A' <= first <= max_letter:
                return first
        return None

    def _find_choice_position(self, text, choice_text):
        """Find earliest case-insensitive occurrence of choice_text in text.
        Word boundaries are required on alphanumeric edges to avoid "194"
        matching inside "1945" or "run" matching inside "running".
        Returns the match position or None."""
        if not choice_text or not text:
            return None
        choice_text = choice_text.strip()
        if not choice_text:
            return None
        pattern = re.escape(choice_text)
        # Anchor with \b on edges that are word characters so substring
        # collisions don't fire (numeric choices like "194" vs "1945").
        if choice_text[0].isalnum():
            pattern = r'\b' + pattern
        if choice_text[-1].isalnum():
            pattern = pattern + r'\b'
        m = re.search(pattern, text, re.IGNORECASE)
        return m.start() if m else None

    def _fetch_model_name(self, ctx):
        """Pull the model name from /api/health for result provenance."""
        try:
            url = ctx.server_url.rstrip("/") + "/api/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict):
                return data.get("model", "unknown")
        except Exception:
            pass
        return "unknown"

    def _save_result(self, test_name, result, ctx):
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            base = (test_name[:-len(".jsonl")]
                    if test_name.endswith(".jsonl") else test_name)
            stamp = time.strftime("%Y%m%dT%H%M%S")
            out_name = f"{base}_{stamp}.json"
            out_path = os.path.join(RESULTS_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return out_path
        except OSError as e:
            ctx.print(f"  Couldn't save result: {e}",
                      color=Color.YELLOW)  # noqa: F821
            return None
