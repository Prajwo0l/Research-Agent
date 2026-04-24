from __future__ import annotations

import json 
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any,Dict,List,Optional


RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"
DIM     = "\033[2m"
ORANGE  = "\033[38;5;208m"

_BAR = "═" * 66


@dataclass 
class HumanResponse:
    """Result of a HITL checkpoint interaction."""
    choice : str  # approve edits or abors
    payload :Optional[Any] = None # extra inpit(Json patch , note,etc)


def format_guardrail_report(results:list)->str:
    """Format a list of Guardrail Result objects for display in a HITL prompt"""
    if not results:
        return " (no guardrail results)"
    lines : list[str] =[]
    for r in results:
        icon  = {"pass": "✅", "warn": "⚠️ ", "block": "🔴"}.get(r.status, "❓")
        lines.append(f"  {icon} [{r.guardrail}] {r.message}")

    return "\n".join(lines)

def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Render a simple ASCII table.

    Example:
      format_table(["Query", "Domain"], [["llm reasoning", "academic_papers"], ...])
    """
    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    sep  = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    def fmt_row(cells: List[str]) -> str:
        parts = []
        for i, w in enumerate(col_widths):
            val = str(cells[i]) if i < len(cells) else ""
            parts.append(f" {val:<{w}} ")
        return "|" + "|".join(parts) + "|"

    lines = [sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)



def collect_hitl_response(
        options:List[str],
        extra_prompts : Optional[Dict[str,str]]=None,
        max_attempts : int = 3,
)-> HumanResponse:
    """Read a response from stdin and validate it"""

    from.exceptions import PipelineAbortedError
    extra_prompts = extra_prompts or {}
    valid =[o.lower() for o in options]

    for attempt in range(1,max_attempts + 1):
        try:
            raw = input(f"{BOLD}{CYAN} Your choice [{'/'.join(valid)}]: {RESET}").strip().lower
        except(EOFError,KeyboardInterrupt):
            print(f"\n{RED} Interrupted--aborting pipeline.{RESET}")
            raise PipelineAbortedError("stdin","Keyboard interrupt")
        if raw not in valid:
            remaining = max_attempts -attempt
            if remaining > 0:
                print(f"{YELLOW} Invalid choice '{raw}'."
                      f"Please enter one of : { ','.join(valid)}."
                      f"({remaining} attempt(s) remaining){RESET}")
            else:
                print(f"{RED} Too many invalid attempts --auto -aborting.{RESET}")
                raise PipelineAbortedError ("stdin", "Too many invalid attempts")
            continue
        
        #vakid choice collect extra input if needed
        payload =None
        if raw in extra_prompts:
            prompt_text = extra_prompts[raw]
            print(f"\n{BOLD}{WHITE} {prompt_text}{RESET}")
            print(f"{DIM} (Enter your input below . For JSON ,paste and press Enter twice.){RESET}")
            lines:list[str]=[]
            try:
                while True:
                    line = input(" ")
                    if line =="" and lines:
                        break
                    lines.append(line)
            except (EOFError,KeyboardInterrupt):
                pass

            raw_input = "\n".join(lines).strip()

            if raw_input.startswith("{") or raw_input.startswith("["):
                try:
                    payload = json.loads(raw_input)
                except json.JSONDecodeError:
                    print(f"{YELLOW} Warning: could not parse as JSON -using as plain text . {RESET}")
                    payload = raw_input
            else:
                payload = raw_input
        print()
        return HumanResponse(choice= raw,payload=payload)
    

    raise PipelineAbortedError("stdin","Max attempts exceeded")

def _print_bar() -> None:
    print(f"{BOLD}{CYAN}{_BAR}{RESET}")


def _print_header(checkpoint_name: str, thread_id: str = "", stage: str = "") -> None:
    _print_bar()
    print(f"{BOLD}{YELLOW}  ⏸  HUMAN REVIEW REQUIRED — {checkpoint_name}{RESET}")
    if stage:
        print(f"{DIM}  Stage   : {stage}{RESET}")
    if thread_id:
        print(f"{DIM}  Thread  : {thread_id}{RESET}")
    print(f"{DIM}  Time    : {datetime.now().strftime('%H:%M:%S')}{RESET}")
    _print_bar()

def display_hitl_prompt(payload:dict) -> None:
    """
    Print a formatted HITL Checkpoint prompt to the terminal
    """
    checkpoint = payload.get("checkpoint","Unknown")
    stage = payload.get("stage", "")
    thread_id = payload.get("thread_id","")
    content = payload.get("content","")
    options = payload.get("options",[])
    eta = payload.get("estimated_time_remaining","")
    gr_report = payload.get("guardrail_report", "")
    print()
    _print_header(checkpoint,thread_id=thread_id,stage=stage)
    #main content
    if content:
        print()
        for line in content.splitlines():
            print(f"{line}")
    #guardrail warnings
    if gr_report:
        print()
        print(f"{BOLD}{WHITE} Guardrail Report: {RESET}")
        for line in gr_report.splitlines():
            print(f"{line}")
    #eta
    if eta:
        print()
        print(f"{DIM} Estimated time if approved : {eta}{RESET}")

    # Options
    print()
    print(f"{BOLD}{WHITE}  Options:{RESET}")
    for opt in options:
        key   = opt.get("key",   "?")
        label = opt.get("label", "")
        desc  = opt.get("description", "")
        print(f"    {BOLD}{CYAN}[{key}]{RESET}  {BOLD}{label}{RESET}")
        if desc:
            print(f"         {DIM}{desc}{RESET}")

    _print_bar()
    print()





def build_hitl1_payload(
        decomposition,
        guardrail_results:list,
        thread_id:str,
)-> dict:
    """Format the HITL 1(query plan review ) prompt payload ."""
    queries = decomposition.search_queries
    n = len(queries)

    #Build query table
    rows = [
        [str(i+1),sq.query [:55],sq.domain,sq.rationale[:45]]
        for i , sq in enumerate(queries)
    ]
    table = format_table(["#", "Query","Domain", "Rationale"],rows)

    content = (
        f'Topic : {decomposition.topic}\n'
        f"Scope : {decomposition.scope_statement}\n\n"
        f"Generated {n} search queries : \n\n"
        f"{table}\n\n"
        f"Domain covered : {', '.join(sorted({sq.domain for sq in queries}))}\n"
        f"Estimated runtime: ~{n* 3}#{n*5}s  for parallel search"

    )
    return {
        "checkpoint":  "HITL-1: Query Plan Review",
        "stage":       "PhD Research Agent — Phase 1",
        "thread_id":   thread_id,
        "content":     content,
        "estimated_time_remaining": f"~{n * 4}s search + ~3 min synthesis",
        "guardrail_report": format_guardrail_report(guardrail_results),
        "options": [
            {"key": "a", "label": "Approve",
             "description": "Continue with these queries unchanged."},
            {"key": "e", "label": "Approve with edits",
             "description": (
                 'Provide a JSON patch: {"add":[...], "remove":[0,3], '
                 '"modify":{"2":"new query text"}}'
             )},
            {"key": "x", "label": "Abort",
             "description": "Stop the pipeline. Provide optional reason."},
        ],
    }