import unicodedata
import re
import os
from datetime import datetime


class Logger:
    """A simple logger that prints to console and writes to a file."""
    
    def __init__(self, log_dir: str, filename: str = 'main_log.txt'):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory where log file will be created
            filename: Name of the log file
        """
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)
    
    def log(self, msg: str) -> None:
        """Log a message to console and file."""
        print(msg)
        with open(self.log_path, 'a') as f:
            f.write(str(msg) + '\n')
    
    def __call__(self, msg: str) -> None:
        """Allow logger to be called directly like a function."""
        self.log(msg)


def create_timestamped_log_dir(base_dir: str, prefix: str = '') -> tuple[str, str]:
    """
    Create a timestamped log directory.
    
    Args:
        base_dir: Base directory for logs
        prefix: Optional prefix for the directory name
        
    Returns:
        Tuple of (log_dir path, timestamp string)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}" if prefix else timestamp
    log_dir = os.path.join(base_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, timestamp


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')