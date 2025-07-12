import sys
import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        super().__init__(str(error_message))
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    def get_detailed_error_message(self, error_message, error_detail):
        try:
            _, _, exc_tb = sys.exc_info() if error_detail is None else (None, None, error_detail)
            if exc_tb is not None:
                filename = exc_tb.tb_frame.f_code.co_filename
                line_no = exc_tb.tb_lineno
                return f"Error occurred in script [{filename}] at line [{line_no}]: {error_message}"
            return f"Error: {error_message}"
        except Exception:
            return f"Error: {error_message}"

    def __str__(self):
        return self.error_message
