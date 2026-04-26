from __future__ import annotations

class PipelineAbotedError(Exception):
    """
    Raised when a human chooses to abort at a HITL checkpoint
    """

    def __init__(self,checkpoint:str,reason:str ="") -> None:
        self.checkpoint=checkpoint
        self.reason = reason
        msg = f"Pipeline Aborted at [{checkpoint}]"
        if reason:
            msg += f":{reason}"
        super().__init__(msg)
        
class GuardrailError(Exception):
    """
    Raised when a guardrail BLOCKS pipeline execution.

    Attributes

    guardrail:str
        Name of the guardrail that triggered.
        e.g "Guardrail-1: Query Validator"
    message : str
        Human-readable explanation of why the block was triggered.
    details :dict
        Structured data for logging and debugging(counts,offending values,etc.)
    """
    def __init__(self,guardrail:str,message:str,details:dict | None=None)-> None:
        self.guardrail = guardrail
        self.message = message
        self.details = details or {}
        super(). __init__(f"[{guardrail}] BLOCKED:{message}")