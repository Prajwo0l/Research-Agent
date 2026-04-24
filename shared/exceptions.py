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
        