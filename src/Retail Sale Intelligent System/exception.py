import sys
from src.mlproject.logger import logging


def error_message_details(error, error_detail: sys):
    """Build a detailed error message with filename and line number.

    error: the exception instance
    error_detail: usually the sys module so we can call sys.exc_info()
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        "Error occurred in python script name [{0}] line number [{1}] error message [{2}]"
        .format(file_name, exc_tb.tb_lineno, str(error))
    )

    return error_message



class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # create full message first
        self.error_message = error_message_details(error_message, error_detail)
        # initialize base Exception with the detailed message
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message



