import logging
import os
from datetime import datetime

# Sanitize filename to avoid invalid characters
log_file = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

log_file_path = os.path.join(log_path, log_file)

# Logging format
formats = '%(asctime)s - %(filename)s - line %(lineno)d - %(levelname)s - %(message)s'

# Logging config
logging.basicConfig(
    filename=log_file_path,
    format=formats,
    level=logging.INFO
)
