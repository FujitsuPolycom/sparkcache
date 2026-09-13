"""Prefix digest batching honors declared persistent chunk geometry."""
import pytest
from sparkcache.spark_context_cache_codec import CodecError, chunk_prefix_digests, context_prefix_digest


def test_profile_chunk_boundaries_hash_the_exact_prefix():
    tokens = list(range(6000))
    for size, boundaries in [(32, [32, 2848, 5696]), (256, [256, 512])]:
        result = chunk_prefix_digests(tokens, 'test-identity', boundaries=boundaries, chunk_tokens=size)
        assert result == tuple((end, context_prefix_digest(tokens, 'test-identity', token_count=end)) for end in boundaries)


def test_default_digest_geometry_stays_256():
    tokens = list(range(512))
    assert chunk_prefix_digests(tokens, 'salt', boundaries=[256]) == chunk_prefix_digests(tokens, 'salt', boundaries=[256], chunk_tokens=256)
    with pytest.raises(CodecError, match='multiples of 256'):
        chunk_prefix_digests(tokens, 'salt', boundaries=[32])


@pytest.mark.parametrize('size', [0, -1, True, 32.0, '32'])
def test_invalid_chunk_geometry_is_rejected(size):
    with pytest.raises(CodecError, match='chunk_tokens'):
        chunk_prefix_digests([1] * 512, 'salt', boundaries=[256], chunk_tokens=size)
