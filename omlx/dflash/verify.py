from __future__ import annotations

from typing import Any

from .interfaces import AcceptanceResult, DraftBlock


def greedy_exact_accept(target_token_ids: list[int], draft_token_ids: list[int]) -> AcceptanceResult:
    """Lossless exact-match verifier described in the DFlash paper.

    Accepts the longest exact prefix of the draft block, then leaves room for a
    bonus token from the target path.
    """
    accepted = 0
    for target_id, draft_id in zip(target_token_ids, draft_token_ids):
        if target_id != draft_id:
            break
        accepted += 1

    bonus_token_id = None
    if len(target_token_ids) > accepted:
        bonus_token_id = target_token_ids[accepted]

    return AcceptanceResult(
        accepted_token_ids=draft_token_ids[:accepted],
        rejected_from_index=accepted,
        bonus_token_id=bonus_token_id,
        target_token_ids=target_token_ids,
        accepted_count=accepted,
        drafted_count=len(draft_token_ids),
        exact_match=accepted == len(draft_token_ids),
    )


class DFlashVerifier:
    """Target-side verification boundary.

    Future responsibilities:
    - call the custom Metal M=16 verify kernel
    - run one target forward pass over prefix + draft block
    - return exact-match acceptance result
    - expose instrumentation for tok/s and acceptance rate
    """

    def __init__(self, target_model: Any, kernel: Any | None = None):
        self.target_model = target_model
        self.kernel = kernel

    def verify_block(self, prefix_state: Any, draft_block: DraftBlock) -> AcceptanceResult:
        raise NotImplementedError(
            "DFlash target verification is not wired yet. "
            "Implement target forward + kernel-backed verify path."
        )
