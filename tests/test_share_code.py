from deck_vision.share_code import decode_share_code, encode_share_ids, generate_share_code


def test_encode_decode_round_trip() -> None:
    share_ids = list(range(1, 34))
    payload = encode_share_ids(share_ids, 37)

    import base64

    decoded, seed = decode_share_code(base64.b64encode(payload).decode("ascii"))
    assert seed == 37
    assert decoded == share_ids


def test_generate_share_code_uses_first_unblocked_seed() -> None:
    share_ids = list(range(1, 34))
    first = generate_share_code(share_ids, block_words=[])
    second = generate_share_code(share_ids, block_words=[first[:5].lower()])

    assert first != second
    assert first[:5].lower() not in second.lower()
