"""
evaluator plugin — format-aware benchmark suite for the current model.

Loads JSONL benchmark files from plugins/evaluator/tests/, formats each item,
sends it to the current chat endpoint with deterministic sampling, scores the
model reply, and saves JSON results in plugins/evaluator/results/.

Supported test formats:
  - mcqa         Multiple-choice QA. This is the original format and remains
                 the default when a test file has no explicit format flag.
  - open_ended  Plain open-ended QA. The model's reply is marked correct when
                 it contains one of the expected answer strings.

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
    {"_meta": true, "name": "My Eval", "test_format": "mcqa"}

  Legacy/default MCQA question:
    {"id": "q1", "question": "What is 7 x 8?",
     "choices": ["54", "55", "56", "63"],
     "answer": "C",
     "category": "math"}

  Open-ended question:
    {"id": "q1", "question": "What is 7 x 8?",
     "expected_answer": "56",
     "category": "math"}

Backwards compatibility:
  Any test file that does not declare test_format is treated as the original
  MCQA format.
"""

import os
import re
import json
import time
import argparse


# ──────────────────────────────────────────────────────────
#  Paths — resolved relative to this plugin file so the layout
#  follows the user's --plugins-dir wherever it is.
# ──────────────────────────────────────────────────────────

_PLUGIN_FILE = os.path.abspath(__file__)
_PLUGIN_DIR = os.path.dirname(_PLUGIN_FILE)
TESTS_DIR = os.path.join(_PLUGIN_DIR, "evaluator", "tests")
RESULTS_DIR = os.path.join(_PLUGIN_DIR, "evaluator", "results")


# ──────────────────────────────────────────────────────────
#  Test formats
# ──────────────────────────────────────────────────────────

FORMAT_MCQA = "mcqa"
FORMAT_OPEN_ENDED = "open_ended"
FORMAT_CHECKS = "checks"      # ChatbotGym: deterministic rule-based scoring
FORMAT_JUDGE = "judge"        # ChatbotGym: LLM-as-judge rubric scoring

# Check types understood by the deterministic `checks` scorer.
_CHECK_TYPES = {
    "contains", "contains_any", "contains_all", "not_contains",
    "regex", "equals", "starts_with", "ends_with",
    "max_words", "min_words", "max_chars", "min_chars",
    "line_count", "is_json", "json_has_key",
}

_FORMAT_ALIASES = {
    "mcqa": FORMAT_MCQA,
    "multiple_choice": FORMAT_MCQA,
    "multiplechoice": FORMAT_MCQA,
    "multiple_choice_qa": FORMAT_MCQA,
    "multi_choice": FORMAT_MCQA,
    "multichoice": FORMAT_MCQA,
    "open_ended": FORMAT_OPEN_ENDED,
    "openended": FORMAT_OPEN_ENDED,
    "open_qa": FORMAT_OPEN_ENDED,
    "free_response": FORMAT_OPEN_ENDED,
    "freeform": FORMAT_OPEN_ENDED,
    "qa": FORMAT_OPEN_ENDED,
    "checks": FORMAT_CHECKS,
    "check": FORMAT_CHECKS,
    "rubric_checks": FORMAT_CHECKS,
    "deterministic": FORMAT_CHECKS,
    "judge": FORMAT_JUDGE,
    "llm_judge": FORMAT_JUDGE,
    "model_graded": FORMAT_JUDGE,
    "graded": FORMAT_JUDGE,
}


# ──────────────────────────────────────────────────────────
#  Example test seeded on first launch (only when tests/ is empty)
# ──────────────────────────────────────────────────────────

EXAMPLE_TEST_NAME = "example.jsonl"
EXAMPLE_TEST_RECORDS = [
    {"_meta": True, "name": "Example Mini-Eval", "test_format": FORMAT_MCQA,
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
#  Prompt templates — kept simple. Edit here to experiment.
# ──────────────────────────────────────────────────────────

DEFAULT_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "{choices_block}\n\n"
    "Answer with just the letter of the correct choice."
)

OPEN_ENDED_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Answer directly and concisely."
)


# ──────────────────────────────────────────────────────────
#  Plugin
# ──────────────────────────────────────────────────────────


class EvaluatorPlugin(Plugin):  # noqa: F821 — Plugin is injected by the loader
    name = "evaluator"
    description = (
        "Format-aware benchmark suite. Loads JSONL tests from "
        "plugins/evaluator/tests/ and scores the current model. "
        "Supports legacy MCQA and flagged open-ended tests. "
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
            "Run benchmark tests against the current model.\n"
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
            "Supported test formats:\n"
            "  mcqa         original multiple-choice format; default if omitted\n"
            "  open_ended   plain question + expected_answer containment scoring\n"
            "  checks       chatbot prompt + deterministic rule checks (no judge)\n"
            "  judge        chatbot prompt + rubric graded by a separate judge model\n"
            "\n"
            "Optional first line is a metadata header:\n"
            "  {\"_meta\":true, \"name\":\"My Eval\", \"test_format\":\"mcqa\"}\n"
            "\n"
            "MCQA question:\n"
            "  {\"id\":\"q1\", \"question\":\"...\", \"choices\":[...],\n"
            "   \"answer\":\"B\", \"category\":\"optional\"}\n"
            "\n"
            "Open-ended question:\n"
            "  {\"id\":\"q1\", \"question\":\"...\",\n"
            "   \"expected_answer\":\"Paris\", \"category\":\"optional\"}\n"
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
            test_format = self._display_test_format(meta)
            ctx.print(f"    {f}", color=Color.DIM)  # noqa: F821
            ctx.print(f"      {display_name}  ({len(items)} questions, format={test_format})",
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
        test_format = meta.get("_test_format", FORMAT_MCQA)
        ctx.print(f"  File:        {os.path.basename(path)}",
                  color=Color.DIM)  # noqa: F821
        ctx.print(f"  Format:      {test_format}",
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
            ctx.print(f"    Q: {sample.get('question') or sample.get('prompt') or ''}",
                      color=Color.DIM)  # noqa: F821
            if test_format == FORMAT_MCQA:
                for i, c in enumerate(sample["choices"]):
                    ctx.print(f"    {chr(ord('A') + i)}) {c}",
                              color=Color.DIM)  # noqa: F821
                ctx.print(
                    f"    Answer: "
                    f"{self._normalize_answer(sample['answer'], len(sample['choices']))}",
                    color=Color.DIM,  # noqa: F821
                )
            elif test_format == FORMAT_OPEN_ENDED:
                expected = self._normalize_expected_answers(
                    self._get_expected_answer_value(sample)
                )
                ctx.print(f"    Expected: {self._format_expected_display(expected)}",
                          color=Color.DIM)  # noqa: F821
                if isinstance(sample.get("prompt"), str) and sample["prompt"].strip():
                    ctx.print("    Prompt override: yes",
                              color=Color.DIM)  # noqa: F821
            else:
                if test_format == FORMAT_CHECKS and isinstance(sample.get("checks"), list):
                    ctx.print(f"    Checks: {len(sample['checks'])} rule(s)",
                              color=Color.DIM)  # noqa: F821
                if test_format == FORMAT_JUDGE and isinstance(sample.get("rubric"), str):
                    rub = sample["rubric"]
                    ctx.print(f"    Rubric: {rub[:100]}{'...' if len(rub) > 100 else ''}",
                              color=Color.DIM)  # noqa: F821

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
                    f"    {s['test']:<32}  {s.get('test_format', '?'):<11}  "
                    f"{s['correct']:>3}/{s['total']:<3}  "
                    f"({s['accuracy'] * 100:5.1f}%)",
                    color=Color.DIM,  # noqa: F821
                )
            total_q = sum(s["total"] for s in summaries)
            total_c = sum(s["correct"] for s in summaries)
            if total_q > 0:
                ctx.print(
                    f"    {'TOTAL':<32}  {'':<11}  {total_c:>3}/{total_q:<3}  "
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
                test_format = data.get("test_format", FORMAT_MCQA)
                ctx.print(
                    f"    {f}  {data.get('correct', 0)}/{data.get('total', 0)}  "
                    f"({acc:.1f}%)  format={test_format}  model={data.get('model', '?')}",
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
        ctx.print(f"  Format:    {data.get('test_format', FORMAT_MCQA)}",
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
        test_format = meta.get("_test_format", FORMAT_MCQA)

        if err:
            ctx.print(f"  Load error ({test_name}): {err}",
                      color=Color.RED)  # noqa: F821
            return None
        if not items:
            ctx.print(f"  {test_name}: no questions to run.",
                      color=Color.YELLOW)  # noqa: F821
            return None

        work = items if ns.limit is None else items[:ns.limit]

        # Resolve max_new_tokens. Legacy formats keep the historical 20;
        # chatbot formats need real room to reply. A test may set its own
        # default via a `max_new_tokens` field in its _meta header.
        if ns.max_new is not None:
            resolved_max_new = ns.max_new
        elif isinstance(meta.get("max_new_tokens"), int):
            resolved_max_new = meta["max_new_tokens"]
        elif test_format in (FORMAT_CHECKS, FORMAT_JUDGE):
            resolved_max_new = 384
        else:
            resolved_max_new = 20

        # Eval-specific sampling override. We use top_k=1 alongside temp=0
        # so we get hard-greedy decoding even if the server treats 0.0 as
        # a small epsilon.
        override = {
            "temperature": ns.temp,
            "top_p": 1.0,
            "top_k": 1 if ns.temp == 0.0 else 0,
            "max_new_tokens": resolved_max_new,
            "repetition_penalty": 1.0,
            "no_repeat_ngram": 0,
        }

        # Judge backend config (only used by FORMAT_JUDGE tests).
        judge_cfg = self._judge_cfg_from_meta(meta)

        model_name = self._fetch_model_name(ctx)

        header = test_name
        if meta.get("name"):
            header += f" ({meta['name']})"
        ctx.print(f"  Test:     {header}", color=Color.DIM)  # noqa: F821
        ctx.print(f"  Format:   {test_format}", color=Color.DIM)  # noqa: F821
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
        used_prompt_override = False

        try:
            for i, q in enumerate(work, 1):
                last_i = i
                prompt = self._format_prompt(q, test_format)
                if test_format == FORMAT_OPEN_ENDED and isinstance(q.get("prompt"), str) and q["prompt"].strip():
                    used_prompt_override = True

                reply = ctx.chat(prompt, history=[], sampling_override=override)

                if test_format == FORMAT_MCQA:
                    num_choices = len(q["choices"])
                    expected = self._normalize_answer(q["answer"], num_choices)
                    predicted = self._extract_letter(reply or "", num_choices, q["choices"])
                    is_correct = (predicted is not None and predicted == expected)
                    item_parse_error = predicted is None
                    detail = {
                        "id": q.get("id", f"q{i}"),
                        "category": q.get("category"),
                        "test_format": test_format,
                        "question": q["question"],
                        "choices": q["choices"],
                        "expected": expected,
                        "predicted": predicted,
                        "correct": is_correct,
                        "raw_response": reply or "",
                    }
                elif test_format == FORMAT_OPEN_ENDED:
                    expected_answers = self._normalize_expected_answers(
                        self._get_expected_answer_value(q)
                    )
                    predicted = self._match_expected_answer(reply or "", expected_answers)
                    is_correct = predicted is not None
                    item_parse_error = False
                    detail = {
                        "id": q.get("id", f"q{i}"),
                        "category": q.get("category"),
                        "test_format": test_format,
                        "question": q["question"],
                        "expected": (expected_answers[0]
                                     if len(expected_answers) == 1
                                     else expected_answers),
                        "expected_answers": expected_answers,
                        "predicted": predicted,
                        "correct": is_correct,
                        "raw_response": reply or "",
                    }
                    if isinstance(q.get("prompt"), str) and q["prompt"].strip():
                        detail["prompt"] = q["prompt"]

                elif test_format == FORMAT_CHECKS:
                    is_correct, checks_passed, checks_total, check_results = \
                        self._score_checks(reply, q)
                    item_parse_error = False
                    predicted = f"{checks_passed}/{checks_total} checks"
                    detail = {
                        "id": q.get("id", f"q{i}"),
                        "category": q.get("category"),
                        "test_format": test_format,
                        "question": q.get("question") or q.get("prompt"),
                        "prompt": q.get("prompt"),
                        "checks_passed": checks_passed,
                        "checks_total": checks_total,
                        "check_results": check_results,
                        "predicted": predicted,
                        "correct": is_correct,
                        "raw_response": reply or "",
                    }
                elif test_format == FORMAT_JUDGE:
                    max_score = q.get("max_score", 5)
                    threshold = q.get("pass_threshold", (max_score + 1) / 2.0)
                    j_score, j_raw, j_err = self._run_judge(prompt, reply, q, judge_cfg, ctx)
                    if j_err or j_score is None:
                        is_correct = False
                        item_parse_error = True   # judge failed to return a score
                    else:
                        is_correct = j_score >= threshold
                        item_parse_error = False
                    predicted = (f"{j_score}/{max_score}" if j_score is not None else None)
                    detail = {
                        "id": q.get("id", f"q{i}"),
                        "category": q.get("category"),
                        "test_format": test_format,
                        "question": q.get("question") or q.get("prompt"),
                        "prompt": q.get("prompt"),
                        "rubric": q.get("rubric"),
                        "judge_score": j_score,
                        "judge_max": max_score,
                        "pass_threshold": threshold,
                        "judge_output": j_raw,
                        "judge_error": j_err,
                        "predicted": predicted,
                        "correct": is_correct,
                        "raw_response": reply or "",
                    }
                if is_correct:
                    correct += 1
                if item_parse_error:
                    parse_errors += 1

                cat = q.get("category", "(none)")
                slot = by_cat.setdefault(cat, {"total": 0, "correct": 0})
                slot["total"] += 1
                if is_correct:
                    slot["correct"] += 1

                details.append(detail)

                if ns.verbose:
                    mark = "[+]" if is_correct else ("[?]" if item_parse_error else "[-]")
                    color = (Color.GREEN if is_correct  # noqa: F821
                             else (Color.YELLOW if item_parse_error  # noqa: F821
                                   else Color.RED))  # noqa: F821
                    q_text = q.get("question") or q.get("prompt") or ""
                    q_preview = q_text[:60]
                    if len(q_text) > 60:
                        q_preview += "..."
                    expected_display = self._format_expected_display(
                        detail.get("expected_answers", detail.get("expected"))
                    )
                    predicted_display = str(predicted) if predicted is not None else "None"
                    if len(expected_display) > 28:
                        expected_display = expected_display[:25] + "..."
                    if len(predicted_display) > 18:
                        predicted_display = predicted_display[:15] + "..."
                    ctx.print(
                        f"  [{i:>3}/{len(work)}] {mark} "
                        f"expected={expected_display} predicted={predicted_display:>8}  "
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
            "test_format": test_format,
            "model": model_name,
            "endpoint": ctx.server_url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": override,
            "prompt_template": (DEFAULT_PROMPT_TEMPLATE
                                if test_format == FORMAT_MCQA
                                else OPEN_ENDED_PROMPT_TEMPLATE),
            "used_prompt_override": used_prompt_override,
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
            "test_format": test_format,
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
        # Default is None (a sentinel). It resolves to 20 for the legacy
        # mcqa/open_ended paths (unchanged behavior) and to a larger value
        # for the chatbot formats, which need room to actually reply.
        parser.add_argument("--max-new", type=int, default=None, dest="max_new")
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
        raw_items = []
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
                        meta.update(obj)
                        continue
                    raw_items.append((line_no, obj))
        except OSError as e:
            return {}, [], str(e)

        try:
            test_format = self._normalize_test_format(meta)
        except ValueError as e:
            return meta, [], str(e)
        meta["_test_format"] = test_format

        items = []
        for line_no, obj in raw_items:
            if test_format == FORMAT_MCQA:
                err = self._validate_mcqa_item(obj, line_no)
            elif test_format == FORMAT_OPEN_ENDED:
                err = self._validate_open_ended_item(obj, line_no)
            elif test_format == FORMAT_CHECKS:
                err = self._validate_checks_item(obj, line_no)
            elif test_format == FORMAT_JUDGE:
                err = self._validate_judge_item(obj, line_no)
            else:
                err = f"line {line_no} unhandled format {test_format!r}"
            if err:
                return meta, [], err
            items.append(obj)

        return meta, items, None

    def _normalize_test_format(self, meta):
        """Return canonical test format.

        Backwards compatibility rule: missing format means original MCQA.
        """
        raw = None
        if isinstance(meta, dict):
            for key in ("test_format", "format", "test_type"):
                if key in meta:
                    raw = meta.get(key)
                    break
        if raw is None or raw == "":
            return FORMAT_MCQA
        if not isinstance(raw, str):
            raise ValueError("metadata test_format must be a string")
        key = raw.strip().lower().replace("-", "_").replace(" ", "_")
        key = re.sub(r"_+", "_", key)
        fmt = _FORMAT_ALIASES.get(key)
        if fmt:
            return fmt
        valid = ", ".join(sorted({FORMAT_MCQA, FORMAT_OPEN_ENDED, FORMAT_CHECKS, FORMAT_JUDGE}))
        raise ValueError(f"unsupported test_format {raw!r}; expected one of: {valid}")

    def _display_test_format(self, meta):
        return meta.get("_test_format", FORMAT_MCQA) if isinstance(meta, dict) else FORMAT_MCQA

    def _validate_mcqa_item(self, obj, line_no):
        q = obj.get("question")
        choices = obj.get("choices")
        ans = obj.get("answer")
        if not isinstance(q, str) or not q.strip():
            return f"line {line_no} missing/invalid 'question'"
        if not isinstance(choices, list) or len(choices) < 2:
            return f"line {line_no} 'choices' must be a list of 2+"
        if len(choices) > 26:
            return f"line {line_no} too many choices (max 26)"
        for idx, choice in enumerate(choices):
            if not isinstance(choice, str) or not choice.strip():
                letter = chr(ord('A') + idx)
                return f"line {line_no} choice {letter} must be a non-empty string"
        if ans is None:
            return f"line {line_no} missing 'answer'"
        try:
            self._normalize_answer(ans, len(choices))
        except ValueError as e:
            return f"line {line_no} {e}"
        return None

    def _validate_open_ended_item(self, obj, line_no):
        q = obj.get("question")
        if not isinstance(q, str) or not q.strip():
            return f"line {line_no} missing/invalid 'question'"
        value = self._get_expected_answer_value(obj)
        if value is None:
            return f"line {line_no} missing 'expected_answer'"
        try:
            self._normalize_expected_answers(value)
        except ValueError as e:
            return f"line {line_no} {e}"
        if "prompt" in obj and not isinstance(obj.get("prompt"), str):
            return f"line {line_no} optional 'prompt' must be a string"
        return None

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

    def _get_expected_answer_value(self, q):
        for key in ("expected_answer", "expected_answers", "answers", "answer"):
            if key in q:
                return q.get(key)
        return None

    def _normalize_expected_answers(self, value):
        """Return a non-empty list of accepted answer strings."""
        if isinstance(value, bool) or value is None:
            raise ValueError("expected_answer must be a non-empty string or list of strings")
        if isinstance(value, (int, float)):
            value = str(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                raise ValueError("expected_answer must not be empty")
            return [s]
        if isinstance(value, list):
            answers = []
            for item in value:
                if isinstance(item, bool) or item is None:
                    raise ValueError("expected_answer list must contain only non-empty strings")
                if isinstance(item, (int, float)):
                    item = str(item)
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("expected_answer list must contain only non-empty strings")
                answers.append(item.strip())
            if not answers:
                raise ValueError("expected_answer list must not be empty")
            return answers
        raise ValueError("expected_answer must be a non-empty string or list of strings")

    def _format_expected_display(self, expected):
        if isinstance(expected, list):
            return " | ".join(str(x) for x in expected)
        return str(expected)

    def _format_prompt(self, q, test_format=FORMAT_MCQA):
        if test_format in (FORMAT_CHECKS, FORMAT_JUDGE):
            # Send exactly what an end user would type. No answer-format
            # instructions are appended; following the prompt IS the test.
            return q["prompt"]
        if test_format == FORMAT_OPEN_ENDED:
            prompt = q.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                return prompt
            return OPEN_ENDED_PROMPT_TEMPLATE.format(question=q["question"])

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

    def _match_expected_answer(self, text, expected_answers):
        """Return the matched expected answer string, or None.

        Open-ended scoring is intentionally simple: the reply is correct when
        the normalized response contains one accepted answer string as a whole
        token/phrase. This keeps the plugin lightweight and deterministic.
        """
        if not text:
            return None
        normalized_text = self._normalize_open_text(text)
        if not normalized_text:
            return None
        matches = []
        for answer in expected_answers:
            normalized_answer = self._normalize_open_text(answer)
            if not normalized_answer:
                continue
            pattern = r'(?<!\w)' + re.escape(normalized_answer).replace(r'\ ', r'\s+') + r'(?!\w)'
            m = re.search(pattern, normalized_text)
            if m:
                matches.append((m.start(), answer))
        if not matches:
            return None
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    def _normalize_open_text(self, text):
        text = re.sub(r'<think>.*?</think>', ' ', str(text), flags=re.IGNORECASE | re.DOTALL)
        text = text.lower()
        text = text.replace("’", "'").replace("‘", "'")
        text = re.sub(r"[^a-z0-9']+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _fetch_model_name(self, ctx):
        """Pull the model name from the host's cached health state."""
        try:
            if hasattr(ctx, "refresh_health"):
                ctx.refresh_health()
            model_name = getattr(ctx, "model_name", "unknown")
            if isinstance(model_name, str) and model_name.strip():
                return model_name.strip()
        except Exception:
            pass
        return "unknown"


    # ══════════════════════════════════════════════════════════════════════
    #  ChatbotGym: deterministic `checks` scoring + LLM-as-judge `judge`
    #  scoring. Everything below is additive; the mcqa / open_ended paths
    #  above are untouched, so featherweight scores stay comparable.
    # ══════════════════════════════════════════════════════════════════════

    # ---- validation -------------------------------------------------------

    def _validate_checks_item(self, obj, line_no):
        p = obj.get("prompt")
        if not isinstance(p, str) or not p.strip():
            return f"line {line_no} 'checks' item needs a non-empty 'prompt'"
        checks = obj.get("checks")
        if not isinstance(checks, list) or not checks:
            return f"line {line_no} 'checks' must be a non-empty list"
        for j, c in enumerate(checks):
            if not isinstance(c, dict) or "type" not in c:
                return f"line {line_no} check {j} needs a 'type'"
            if c["type"] not in _CHECK_TYPES:
                return f"line {line_no} unknown check type {c['type']!r}"
        mode = obj.get("scoring", "all")
        if mode not in ("all", "fraction"):
            return f"line {line_no} 'scoring' must be 'all' or 'fraction'"
        if mode == "fraction":
            pf = obj.get("pass_fraction", 1.0)
            if not (isinstance(pf, (int, float)) and 0.0 < pf <= 1.0):
                return f"line {line_no} 'pass_fraction' must be in (0, 1]"
        return None

    def _validate_judge_item(self, obj, line_no):
        p = obj.get("prompt")
        if not isinstance(p, str) or not p.strip():
            return f"line {line_no} 'judge' item needs a non-empty 'prompt'"
        r = obj.get("rubric")
        if not isinstance(r, str) or not r.strip():
            return f"line {line_no} 'judge' item needs a non-empty 'rubric'"
        ms = obj.get("max_score", 5)
        if not isinstance(ms, int) or isinstance(ms, bool) or ms < 2:
            return f"line {line_no} 'max_score' must be an int >= 2"
        pt = obj.get("pass_threshold")
        if pt is not None and not (isinstance(pt, (int, float))
                                   and not isinstance(pt, bool)
                                   and 1 <= pt <= ms):
            return f"line {line_no} 'pass_threshold' must be within 1..max_score"
        if "reference" in obj and not isinstance(obj["reference"], str):
            return f"line {line_no} optional 'reference' must be a string"
        return None

    # ---- deterministic checks ---------------------------------------------

    def _score_checks(self, reply, q):
        """Return (is_correct, passed, total, per_check_results)."""
        reply = reply or ""
        results = []
        for c in q["checks"]:
            ok, note = self._run_single_check(reply, c)
            results.append({"type": c.get("type"), "passed": bool(ok), "note": note})
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        if q.get("scoring", "all") == "fraction":
            frac = (passed / total) if total else 0.0
            is_correct = frac >= q.get("pass_fraction", 1.0)
        else:
            is_correct = (total > 0 and passed == total)
        return is_correct, passed, total, results

    def _run_single_check(self, reply, c):
        t = c.get("type")
        ic = c.get("ignore_case", True)
        def hay(s):
            return s.lower() if ic else s

        if t in ("contains", "contains_any", "contains_all", "not_contains"):
            vals = c.get("values")
            if vals is None and "value" in c:
                vals = [c["value"]]
            vals = [str(v) for v in (vals or [])]
            h = hay(reply)
            present = [v for v in vals if hay(v) in h]
            if t in ("contains", "contains_all"):
                ok = len(vals) > 0 and len(present) == len(vals)
            elif t == "contains_any":
                ok = len(present) > 0
            else:  # not_contains
                ok = len(present) == 0
            return ok, f"{len(present)}/{len(vals)} present"

        if t == "regex":
            # regex defaults to case-SENSITIVE: the pattern author controls
            # case explicitly, and IGNORECASE silently breaks patterns like
            # ^[^a-z]*$ (it would fold A-Z into the class too).
            flags = re.IGNORECASE if c.get("ignore_case", False) else 0
            if c.get("dotall"):
                flags |= re.DOTALL
            if c.get("multiline"):
                flags |= re.MULTILINE
            ok = re.search(c.get("pattern", ""), reply, flags) is not None
            return ok, ("match" if ok else "no match")

        if t == "equals":
            if c.get("normalize", True):
                a = self._normalize_open_text(reply)
                b = self._normalize_open_text(str(c.get("value", "")))
            else:
                a, b = reply.strip(), str(c.get("value", "")).strip()
            return a == b, f"got={a[:40]!r}"

        if t in ("starts_with", "ends_with"):
            s = hay(reply.strip())
            v = hay(str(c.get("value", "")))
            ok = s.startswith(v) if t == "starts_with" else s.endswith(v)
            return ok, ""

        if t in ("max_words", "min_words"):
            n = len(re.findall(r"\S+", reply))
            lim = c.get("value", c.get("equals"))
            ok = (n <= lim) if t == "max_words" else (n >= lim)
            return ok, f"{n} words (limit {lim})"

        if t in ("max_chars", "min_chars"):
            n = len(reply.strip())
            lim = c.get("value")
            ok = (n <= lim) if t == "max_chars" else (n >= lim)
            return ok, f"{n} chars (limit {lim})"

        if t == "line_count":
            lines = [ln for ln in reply.splitlines() if ln.strip()]
            n = len(lines)
            if "equals" in c:
                ok = n == c["equals"]
            else:
                ok = True
                if "min" in c:
                    ok = ok and n >= c["min"]
                if "max" in c:
                    ok = ok and n <= c["max"]
            return ok, f"{n} non-empty lines"

        if t in ("is_json", "json_has_key"):
            data, err = self._try_parse_json(reply)
            if err:
                return False, "invalid json"
            if t == "is_json":
                return True, "valid json"
            key = c.get("key")
            ok = isinstance(data, dict) and key in data
            return ok, f"key {key!r} {'found' if ok else 'missing'}"

        return False, f"unimplemented check {t!r}"

    def _try_parse_json(self, text):
        s = (text or "").strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
        if fence:
            s = fence.group(1).strip()
        try:
            return json.loads(s), None
        except Exception:
            m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1)), None
                except Exception as e:
                    return None, str(e)
            return None, "no json object found"

    # ---- LLM-as-judge -----------------------------------------------------

    def _judge_cfg_from_meta(self, meta):
        cfg = {}
        if isinstance(meta, dict) and isinstance(meta.get("judge"), dict):
            cfg = dict(meta["judge"])
        cfg.setdefault("backend", "openai")
        return cfg

    def _build_judge_prompt(self, user_prompt, reply, rubric, reference, max_score):
        reply = (reply or "").strip()
        parts = [
            "You are a strict grader assessing an AI assistant's reply to a user, "
            "for everyday chatbot quality. Judge ONLY against the rubric. Do not "
            "reward length, verbosity, or confident tone on their own.",
            "",
            "[USER MESSAGE]",
            user_prompt.strip(),
            "",
            "[ASSISTANT REPLY]",
            reply if reply else "(the assistant produced no usable reply)",
            "",
            "[RUBRIC]",
            rubric.strip(),
        ]
        if reference and reference.strip():
            parts += ["", "[REFERENCE ANSWER — GUIDANCE ONLY, NOT THE ONLY VALID ANSWER]",
                      reference.strip()]
        parts += [
            "",
            f"Score the reply as an integer from 1 to {max_score} "
            f"(1 = fails the rubric, {max_score} = fully satisfies it).",
            "Respond in exactly two lines and nothing else:",
            "SCORE: <integer>",
            "REASON: <one short sentence>",
        ]
        return "\n".join(parts)

    def _run_judge(self, user_prompt, reply, q, judge_cfg, ctx):
        """Return (score_or_None, raw_judge_text_or_None, error_or_None)."""
        max_score = q.get("max_score", 5)
        judge_prompt = self._build_judge_prompt(
            user_prompt, reply or "", q["rubric"], q.get("reference"), max_score
        )
        backend = judge_cfg.get("backend", "openai")
        try:
            if backend == "ctx_chat":
                # Discouraged: uses the model under test as its own judge.
                raw = ctx.chat(
                    judge_prompt, history=[],
                    sampling_override={"temperature": 0.0, "top_p": 1.0,
                                       "top_k": 1, "max_new_tokens": 128},
                )
            elif backend == "openai":
                raw = self._judge_call_openai(judge_prompt, judge_cfg)
            else:
                return None, None, f"unknown judge backend {backend!r}"
        except Exception as e:
            return None, None, f"judge call failed: {e}"
        score = self._parse_judge_score(raw or "", max_score)
        if score is None:
            return None, (raw or ""), "could not parse judge score"
        return score, (raw or ""), None

    def _judge_call_openai(self, judge_prompt, cfg):
        """Minimal OpenAI-compatible /v1/chat/completions call via stdlib.

        Works with vLLM, llama.cpp server, LM Studio, Ollama's OpenAI shim,
        or the real OpenAI/Anthropic-compatible gateways. Configure via the
        test's _meta.judge block: {"backend":"openai","url":...,"model":...}.
        """
        import urllib.request
        url = cfg.get("url") or cfg.get("endpoint")
        if not url:
            raise RuntimeError("judge.url is not set in the test's _meta.judge block")
        body = {
            "model": cfg.get("model", "judge"),
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": cfg.get("temperature", 0.0),
            "max_tokens": cfg.get("max_tokens", 128),
        }
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        key_env = cfg.get("api_key_env")
        if key_env and os.environ.get(key_env):
            headers["Authorization"] = f"Bearer {os.environ[key_env]}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=cfg.get("timeout", 60)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]

    def _parse_judge_score(self, text, max_score):
        if not text:
            return None
        m = re.search(r"score\s*[:=]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if not m:
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*\d+", text)
        if not m:
            m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if not m:
            return None
        try:
            val = float(m.group(1))
        except ValueError:
            return None
        if val < 1:
            val = 1.0
        if val > max_score:
            val = float(max_score)
        # Return an int when it is one, for cleaner reports.
        return int(val) if val == int(val) else val

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
