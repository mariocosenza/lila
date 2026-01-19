from __future__ import annotations

import logging
import multiprocessing as mp
from io import StringIO
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Optional, Tuple, Any, Dict
from collections.abc import Iterator

from lark import Lark, UnexpectedInput

from .semantic.ast_builder import ASTBuilder
from .semantic.semantic_analyzer import SemanticAnalyzer, SemanticError
from .codegen.code_generator import CodeGenerator
from .codegen.optimizer import GrammoOptimizer
from .codegen.execution import JITExecutor


_EXEC_TIMEOUT_SECONDS: int = 300

_PARSER: Optional[Lark] = None


def load_parser() -> Lark:
    base_dir = Path(__file__).parent
    grammar_path = base_dir / "lex_syntax" / "grammo.lark"

    if not grammar_path.exists():
        grammar_path = Path("grammo.lark")

    if not grammar_path.exists():
        raise FileNotFoundError(f"Could not find grammar at {grammar_path.absolute()}")

    grammar = grammar_path.read_text(encoding="utf-8")
    return Lark(
        grammar,
        start="start",
        parser="lalr",
        propagate_positions=True,
        maybe_placeholders=False,
    )


def _get_parser() -> Lark:
    global _PARSER
    if _PARSER is None:
        _PARSER = load_parser()
    return _PARSER


class _LevelSplitHandler(logging.Handler):
    """
    Routes log records into one of three buffers based on level:
      - < WARNING -> info buffer
      - WARNING.. < ERROR -> warning buffer
      - >= ERROR -> errors buffer
    """

    def __init__(self, info_buf: StringIO, warn_buf: StringIO, err_buf: StringIO):
        super().__init__()
        self._info = info_buf
        self._warn = warn_buf
        self._err = err_buf

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()

        if record.levelno >= logging.ERROR:
            self._err.write(msg + "\n")
        elif record.levelno >= logging.WARNING:
            self._warn.write(msg + "\n")
        else:
            self._info.write(msg + "\n")


class _StdoutStderrLogHandler(logging.Handler):
    """
    Routes log records into stdout/stderr buffers:
      - < WARNING -> stdout buffer
      - >= WARNING -> stderr buffer
    """

    def __init__(self, stdout_buf: StringIO, stderr_buf: StringIO):
        super().__init__()
        self._out = stdout_buf
        self._err = stderr_buf

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()

        if record.levelno >= logging.WARNING:
            self._err.write(msg + "\n")
        else:
            self._out.write(msg + "\n")


@contextmanager
def _capture_output(log_level: int = logging.INFO) -> Iterator[Tuple[StringIO, StringIO, StringIO, logging.Logger]]:
    """
    Captures:
      - stdout/stderr into info buffer (prints, tracebacks, etc.)
      - INFO logs into info buffer
      - WARNING logs into warning buffer
      - ERROR/CRITICAL logs into errors buffer

    This manipulates the root logger handlers.
    """
    info_buf = StringIO()
    warn_buf = StringIO()
    err_buf = StringIO()

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level

    handler = _LevelSplitHandler(info_buf, warn_buf, err_buf)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root.handlers = [handler]
    root.setLevel(log_level)

    try:
        with redirect_stdout(info_buf), redirect_stderr(info_buf):
            yield info_buf, warn_buf, err_buf, root
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)
        handler.close()


@contextmanager
def _capture_test_io(log_level: int = logging.INFO) -> Iterator[Tuple[StringIO, StringIO, logging.Logger]]:
    """
    Captures:
      - stdout -> stdout buffer
      - stderr -> stderr buffer
      - INFO/DEBUG logs -> stdout buffer
      - WARNING/ERROR/CRITICAL logs -> stderr buffer

    This manipulates the root logger handlers.
    """
    stdout_buf = StringIO()
    stderr_buf = StringIO()

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level

    handler = _StdoutStderrLogHandler(stdout_buf, stderr_buf)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root.handlers = [handler]
    root.setLevel(log_level)

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            yield stdout_buf, stderr_buf, root
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)
        handler.close()


def _compile_to_module(source: str, opt_level: int):
    """
    Internal compilation pipeline:
      parse -> AST -> semantic -> codegen -> optimize
    Returns optimized LLVM module reference.
    """
    def _run_pipeline(src_text: str):
        parser = _get_parser()
        tree = parser.parse(src_text)

        builder = ASTBuilder()
        ast_root = builder.transform(tree)

        analyzer = SemanticAnalyzer()
        analyzer.analyze(ast_root)

        codegen = CodeGenerator()
        llvm_module = codegen.visit(ast_root)

        optimizer = GrammoOptimizer()
        return optimizer.optimize(llvm_module, speed_level=opt_level)

    try:
        return _run_pipeline(source)
    except SemanticError as e:
        # Check if the error is due to missing 'main'
        err_msg = str(e).lower()
        if "main" in err_msg and ("missing" in err_msg or "not found" in err_msg or "required" in err_msg):
            logging.info("Missing 'main' detected. Implicitly injecting empty main function.")
            new_source = source + "\n\n// [Implicit] Auto-generated empty main\nfunc void -> main() {}\n"
            return _run_pipeline(new_source)
        raise e


def _worker_execute_ir(ir_text: str, q) -> None:
    """
    Worker process: parse LLVM IR text into a module and execute via JITExecutor.
    Returns dict with keys: info, warning, errors.
    """
    with _capture_output(log_level=logging.INFO) as (info, warn, err, logger):
        try:
            import llvmlite.binding as llvm  # type: ignore

            # Safe to call multiple times in-process
            try:
                llvm.initialize()
                llvm.initialize_native_target()
                llvm.initialize_native_asmprinter()
            except Exception:
                pass

            mod = llvm.parse_assembly(ir_text)
            mod.verify()

            executor = JITExecutor()
            executor.run(mod)

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            q.put(
                {
                    "info": info.getvalue(),
                    "warning": warn.getvalue(),
                    "errors": err.getvalue(),
                }
            )


def _worker_execute_ir_test(ir_text: str, q) -> None:
    """
    Worker process: execute LLVM IR and capture stdout/stderr for tests.
    Returns dict with keys: stdout, stderr.
    """
    with _capture_test_io(log_level=logging.INFO) as (out, err, logger):
        try:
            import llvmlite.binding as llvm  # type: ignore

            try:
                llvm.initialize()
                llvm.initialize_native_target()
                llvm.initialize_native_asmprinter()
            except Exception:
                pass

            mod = llvm.parse_assembly(ir_text)
            mod.verify()

            executor = JITExecutor()
            executor.run(mod)

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            q.put({"stdout": out.getvalue(), "stderr": err.getvalue()})


def _run_with_timeout(worker_fn, args: tuple, timeout_seconds: int) -> tuple[bool, Optional[dict]]:
    """
    Runs worker_fn(*args, q) in a subprocess and enforces a hard timeout.
    Returns (timed_out, payload_dict_or_none).
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=worker_fn, args=(*args, q))
    p.start()
    p.join(timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join(5)
        return True, None

    try:
        payload = q.get_nowait()
    except Exception:
        payload = None

    return False, payload


def compile_text(source: str, opt_level: int = 3) -> dict:
    """
    Compile ONLY (no execution).

    Returns dict:
      {
        "compiled": bool,
        "info": str,
        "warning": str,
        "errors": str
      }

    On success, "info" includes the optimized LLVM IR appended at the end.
    """
    with _capture_output(log_level=logging.INFO) as (info, warn, err, logger):
        result = {"compiled": False, "info": "", "warning": "", "errors": ""}

        logger.info("Parsing/Compiling...")
        try:
            optimized_mod_ref = _compile_to_module(source, opt_level=opt_level)
            result["compiled"] = True

            logger.info("Compilation successful.")
            ir_text = str(optimized_mod_ref)

            info.write("\n=== LLVM IR (Optimized) ===\n")
            info.write(ir_text)
            if not ir_text.endswith("\n"):
                info.write("\n")

        except UnexpectedInput as e:
            logger.error(f"Syntax Error at line {e.line}, column {e.column}:")
            try:
                logger.error(e.get_context(source))
            except Exception:
                pass
            logger.error(str(e))

        except SemanticError as e:
            logger.error(f"Semantic Error:\n{e}")

        except Exception as e:
            import traceback
            logger.critical(f"Unexpected Error: {e}")
            logger.error(traceback.format_exc())

        finally:
            result["info"] = info.getvalue()
            result["warning"] = warn.getvalue()
            result["errors"] = err.getvalue()

        return result


def run_text(source: str, opt_level: int = 3) -> dict:
    """
    Compile + Execute via JITExecutor, enforcing a hard 5-minute execution limit.

    Returns dict:
      {
        "compiled": bool,
        "info": str,
        "warning": str,
        "errors": str
      }
    """
    with _capture_output(log_level=logging.INFO) as (info, warn, err, logger):
        result = {"compiled": False, "info": "", "warning": "", "errors": ""}

        logger.info("Compiling...")
        try:
            optimized_mod_ref = _compile_to_module(source, opt_level=opt_level)
            result["compiled"] = True

            ir_text = str(optimized_mod_ref)

            logger.info("Executing (timeout: 300s)...")
            timed_out, payload = _run_with_timeout(
                _worker_execute_ir,
                args=(ir_text,),
                timeout_seconds=_EXEC_TIMEOUT_SECONDS,
            )

            if timed_out:
                err.write(f"ERROR: Execution timed out after {_EXEC_TIMEOUT_SECONDS} seconds.\n")
            elif payload is None:
                err.write("ERROR: Execution process finished but returned no output payload.\n")
            else:
                if payload.get("info"):
                    info.write(payload["info"])
                if payload.get("warning"):
                    warn.write(payload["warning"])
                if payload.get("errors"):
                    err.write(payload["errors"])

            logger.info("Execution finished.")

        except UnexpectedInput as e:
            logger.error(f"Syntax Error at line {e.line}, column {e.column}:")
            try:
                logger.error(e.get_context(source))
            except Exception:
                pass
            logger.error(str(e))

        except SemanticError as e:
            logger.error(f"Semantic Error:\n{e}")

        except Exception as e:
            import traceback
            logger.critical(f"Unexpected Error: {e}")
            logger.error(traceback.format_exc())

        finally:
            result["info"] = info.getvalue()
            result["warning"] = warn.getvalue()
            result["errors"] = err.getvalue()

        return result


def run_tests(code: str, tests: str) -> Dict[str, Any]:
    """
    Run Grammo tests against provided source code.

    Parameters
    ----------
    code : str
        Source code to be tested (e.g., module source or program text).
    tests : str
        Test definitions or test-suite content (e.g., test cases, assertions, or
        commands) that the runner will execute against `code`.

    Returns
    -------
    Dict[str, Any]
        Result dictionary containing at minimum:
        - "passed" (bool): True if all tests passed, False otherwise.
        - "stdout" (str): Captured standard output from the test run.
        - "stderr" (str): Captured standard error from the test run.

    Raises
    ------
    ValueError
        If `code` or `tests` are empty or otherwise invalid.
    RuntimeError
        If the test execution environment cannot be initialized or the runner
        encounters an unexpected internal error.
    """
    if code is None or not code.strip():
        raise ValueError("`code` must be a non-empty string.")
    if tests is None or not tests.strip():
        raise ValueError("`tests` must be a non-empty string.")

    bundle = code.rstrip() + "\n\n" + tests.lstrip()
    opt_level = 3


    with _capture_test_io(log_level=logging.INFO) as (compile_out, compile_err, logger):
        try:
            logger.info("Compiling test bundle...")
            optimized_mod_ref = _compile_to_module(bundle, opt_level=opt_level)
            ir_text = str(optimized_mod_ref)
        except UnexpectedInput as e:
            logger.error(f"Syntax Error at line {e.line}, column {e.column}:")
            try:
                logger.error(e.get_context(bundle))
            except Exception:
                pass
            logger.error(str(e))
            return {"passed": False, "stdout": compile_out.getvalue(), "stderr": compile_err.getvalue()}
        except SemanticError as e:
            logger.error(f"Semantic Error:\n{e}")
            return {"passed": False, "stdout": compile_out.getvalue(), "stderr": compile_err.getvalue()}
        except Exception as e:
            import traceback
            logger.error(f"Unexpected compile error: {e}")
            logger.error(traceback.format_exc())
            return {"passed": False, "stdout": compile_out.getvalue(), "stderr": compile_err.getvalue()}

        # Execute in subprocess with hard timeout
        logger.info("Running tests (timeout: 300s)...")
        timed_out, payload = _run_with_timeout(
            _worker_execute_ir_test,
            args=(ir_text,),
            timeout_seconds=_EXEC_TIMEOUT_SECONDS,
        )

        stdout_run = ""
        stderr_run = ""

        if timed_out:
            stderr_run = f"ERROR: Test execution timed out after {_EXEC_TIMEOUT_SECONDS} seconds.\n"
        elif payload is None:
            stderr_run = "ERROR: Test execution process finished but returned no output payload.\n"
        else:
            stdout_run = payload.get("stdout", "")
            stderr_run = payload.get("stderr", "")

        # Combine compile and run outputs
        stdout_all = compile_out.getvalue() + stdout_run
        stderr_all = compile_err.getvalue() + stderr_run

        # Pass/fail rules:
        # - Any timeout or execution error logged to stderr typically implies fail.
        # - Optional explicit markers:
        combined = stdout_all + "\n" + stderr_all
        passed = True
        if "GRAMMO_TEST_RESULT:FAIL" in combined:
            passed = False
        elif "GRAMMO_TEST_RESULT:PASS" in combined:
            passed = True
        else:
            # If stderr contains "ERROR:" from timeout/payload issues, fail
            if "ERROR:" in stderr_all:
                passed = False

        return {"passed": passed, "stdout": stdout_all, "stderr": stderr_all}