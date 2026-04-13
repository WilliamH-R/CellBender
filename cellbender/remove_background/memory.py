"""Helpers for handling CUDA memory pressure."""

import cellbender.remove_background.consts as consts

from typing import Optional


def is_cuda_oom_error(error: BaseException) -> bool:
    """Return True when an exception looks like a CUDA out-of-memory error."""

    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return ('cuda' in message) and ('out of memory' in message)


def get_reduced_batch_size(batch_size: int) -> Optional[int]:
    """Halve a batch size, respecting the smallest supported minibatch."""

    if batch_size <= consts.SMALLEST_ALLOWED_BATCH:
        return None
    reduced = max(consts.SMALLEST_ALLOWED_BATCH, batch_size // 2)
    return reduced if reduced < batch_size else None
