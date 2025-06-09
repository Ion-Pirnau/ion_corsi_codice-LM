import time
from functools import wraps

def measure_time(method):
    """
    Decorator for measuring the execution method's time.
    Add the time as the fourth element of the return's values

    :param method: Any type of method, that use this function as a decorator
    """

    @wraps(method)
    def timed_method(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        elapsed = (end - start)

        if elapsed < 1e-6:
            formatted_time = f"{elapsed * 1_000_000_000:.2f} ns"
        elif elapsed < 1e-3:
            formatted_time = f"{elapsed * 1_000_000:.2f} µs"
        elif elapsed < 1:
            formatted_time = f"{elapsed * 1000:.2f} ms"
        else:
            formatted_time = f"{elapsed:.2f} s"

        if isinstance(result, tuple):
            return(*result, formatted_time)
        else:
            return result, formatted_time
    
    return timed_method

def parse_time_string(time_str: str) -> float:
    """
    Parse the time-string into float values

    :param time_str: string to parse into float

    :return: float value of the time
    """

    val, unit = time_str.split()
    val = float(val)

    if unit in ['s', 'sec', 'seconds']:
        return val
    elif unit in ['ms', 'milliseconds']:
        return val / 1000
    elif unit in ['µs', 'us', 'microsecondi']:
        return val / 1_000_000
    elif unit in ['ns', 'nanoseconds']:
        return val / 1_000_000_000
    else:
        raise ValueError(f"Time unit not acceptable!: {unit}" )