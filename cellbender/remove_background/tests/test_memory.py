"""Test CUDA memory helper functions."""

from cellbender.remove_background.memory import is_cuda_oom_error, get_reduced_batch_size


def test_is_cuda_oom_error():
    assert is_cuda_oom_error(RuntimeError('CUDA out of memory. Tried to allocate 16.00 MiB'))
    assert not is_cuda_oom_error(RuntimeError('some unrelated runtime error'))
    assert not is_cuda_oom_error(ValueError('CUDA out of memory'))


def test_get_reduced_batch_size():
    assert get_reduced_batch_size(128) == 64
    assert get_reduced_batch_size(5) == 4
    assert get_reduced_batch_size(4) is None
