class TODO:
    def __init__(self, msg: str = None):
        error_msg = "This is a placeholder value"
        if msg is not None:
            error_msg = f'{error_msg}: "{msg}"'
        raise NotImplementedError(error_msg)
