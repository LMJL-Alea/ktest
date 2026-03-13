from numbers import Number
import warnings


def verbosity(msg: str, msg_type: str = "print", verbose: int | bool = 0):
    """
    Verbosity output function.

    Input:
        msg (str): message to print.
        msg_type (str): type of message among `"print"`, `"warning"`.
        verbose (int | bool): verbosity level, `0` means no output,
            `1` means only print, `>=2` means print and warning.
            Note: `verbose=False` is equivalent to `verbose=0` and
            `verbose=True` is equivalent to `verbose=1`

    Output: no output
    """

    # check input
    assert isinstance(msg, str), "`msg` input should be a character string"
    assert isinstance(msg_type, str) and msg_type in ["print", "warning"], \
        "`msg_type` input should be either \"print\" or \"warning\""
    assert isinstance(verbose, bool) or \
        isinstance(verbose, Number) and verbose.is_integer(), \
        "`verbosity` input should be a boolean or an integer value"

    # manage verbosity
    if verbose == 1 and msg_type == "print":
        print(msg)
    elif verbose > 1:
        if msg_type == "print":
            print(msg)
        elif msg_type == "warning":
            warnings.warn(msg, UserWarning)
