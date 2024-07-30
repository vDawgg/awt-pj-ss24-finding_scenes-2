def time_string_to_milliseconds(time_str):
    """
    Converts a time string in the format 'hh:mm:ss.sss' to milliseconds.
    Parameters:
    time_str (str): The time string to be converted.
    Returns:
    int: The total number of milliseconds.
    Example:
    >>> time_string_to_milliseconds('01:23:45.678')
    5025678
    """
    # Split the time string into hours, minutes, seconds, and milliseconds
    hours, minutes, seconds_milliseconds = time_str.split(':')
    seconds, milliseconds = seconds_milliseconds.split('.')

    # Convert each component to milliseconds and sum them up
    total_milliseconds = (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)

    return total_milliseconds


def milliseconds_to_time_string(milliseconds):
    """
    Converts milliseconds to a formatted time string in the format hh:mm:ss.SSS.
    Args:
        milliseconds (int): The number of milliseconds to convert.
    Returns:
        str: The formatted time string in the format hh:mm:ss.SSS.
    """
    # Convert milliseconds to seconds
    total_seconds = milliseconds / 1000

    # Extract hours, minutes, seconds, and milliseconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)

    # Format the components into hh:mm:ss.SSS format
    time_string = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds * 1000):03d}"

    return time_string