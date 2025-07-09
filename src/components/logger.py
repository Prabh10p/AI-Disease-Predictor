from datetime import datetime
import logging
import os
log_file = f"{datetime.now().strftime('%d:%m/%Y, %H:%M:%S')}.log"
log_path = os.path.join(os.getcwd(),"logs")
os.makedirs(log_path,exist_ok=True)


log_file_path = os.path.join(log_path,log_file)
formats = '%(asctime)s - %(filename)s - line %(lineno)d - %(levelname)s - %(message)s'



logging.basicConfig(filename=log_file_path,
                    format=formats,
                    level=logging.info)
