from datetime import datetime

class TimestampGenerator:

    def __init__(self, format_string= "%Y%m%d_%H%M%S"):
        """
        Initialize the TimestampGenerator with a specific format.

        Args:
            format_string: Format string for datetime.strftime
        """
        self.format_string = format_string
    def generate_timestamp(self):
        """
        Generate a timestamp string based on the current time.

        Returns:
            Formatted timestamp string
        """
        return datetime.now().strftime(self.format_string)
    def generate_custom_timestamp(self, format_string):
        """
        Generate a timestamp string based on the current time with a custom format.

        Args:
            format_string: Custom format string for datetime.strftime

        Returns:
            Formatted timestamp string
        """
        return datetime.now().strftime(format_string)
    def generate_unix_epoch(self):
        """
        Generate the current time as a Unix epoch timestamp.

        Returns:
            Unix epoch timestamp (float)
        """
        return int(datetime.now().timestamp())
    
    def get_current_datetime(self):
        """
        Get the current datetime object.

        Returns:
            Current datetime object
        """
        return datetime.now()

class TimestampFormats:
    """Common timestamp format strings."""
    COMPACT = "%Y%m%d_%H%M%S"
    READABLE = "%Y-%m-%d_%H:%M:%S"
    DATE_ONLY = "%Y-%m-%d"
    TIME_ONLY = "%H-%M-%S"
