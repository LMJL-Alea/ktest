import pytest

from ktest.utils import verbosity


def test_verbosity(capsys, recwarn):
    """Test function managing verbosity output."""

    # no output
    msg = "no output"
    verbosity(msg, msg_type="print", verbose=0)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
    assert len(recwarn) == 0

    msg = "no output"
    verbosity(msg, msg_type="warning", verbose=0)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
    assert len(recwarn) == 0

    # print only output
    msg = "print only"
    verbosity(msg, msg_type="print", verbose=1)
    captured = capsys.readouterr()
    assert captured.out == f"{msg}\n"
    assert captured.err == ''
    assert len(recwarn) == 0

    msg = "print only"
    verbosity(msg, msg_type="warning", verbose=1)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
    assert len(recwarn) == 0

    # print or warning output
    msg = "this is a print"
    verbosity(msg, msg_type="print", verbose=2)
    captured = capsys.readouterr()
    assert captured.out == f"{msg}\n"
    assert captured.err == ''
    assert len(recwarn) == 0

    msg = "this is a warning"
    with pytest.warns(UserWarning, match=msg):
        verbosity(msg, msg_type="warning", verbose=2)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''

    # boolean verbosity level
    msg = "no output"
    verbosity(msg, msg_type="print", verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
    assert len(recwarn) == 0

    msg = "print only"
    verbosity(msg, msg_type="print", verbose=True)
    captured = capsys.readouterr()
    assert captured.out == f"{msg}\n"
    assert captured.err == ''
    assert len(recwarn) == 0

    # check bad input
    err_msg = "`msg` input should be a character string"
    with pytest.raises(AssertionError) as excinfo:
        verbosity(msg=111, msg_type="print", verbose=0)
    assert str(excinfo.value) == err_msg

    err_msg = "`msg_type` input should be either \"print\" or \"warning\""
    with pytest.raises(AssertionError) as excinfo:
        verbosity(msg="this is a message", msg_type="error", verbose=0)
    assert str(excinfo.value) == err_msg

    err_msg = "`verbosity` input should be a boolean or an integer value"
    with pytest.raises(AssertionError) as excinfo:
        verbosity(msg="this is a message", msg_type="print", verbose="0")
    assert str(excinfo.value) == err_msg

