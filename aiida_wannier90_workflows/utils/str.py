"""Functions for str."""


def removesuffix(text: str, suffix: str) -> str:
    """For python <= 3.8 compatibility.

    :param text: _description_
    :type text: str
    :param suffix: _description_
    :type suffix: str
    :return: _description_
    :rtype: str
    """
    return text[: -len(suffix)] if text.endswith(suffix) and len(suffix) != 0 else text


def removeprefix(text: str, prefix: str) -> str:
    """For python <= 3.8 compatibility.

    :param text: _description_
    :type text: str
    :param prefix: _description_
    :type prefix: str
    :return: _description_
    :rtype: str
    """
    return text[len(prefix) :] if text.startswith(prefix) and len(prefix) != 0 else text
