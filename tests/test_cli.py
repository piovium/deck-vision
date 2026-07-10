from deck_vision.cli import main


def test_cli_missing_image_reports_structured_error(capsys) -> None:
    code = main(["recognize", "missing.png", "--json"])

    captured = capsys.readouterr()
    assert code == 2
    assert "image_not_found" in captured.err
