def validate_temperature_range(min_temp: float | None, max_temp: float | None) -> bool:
    """Validate min and max temperature values.

    Args:
        min_temp: Minimum temperature value or None
        max_temp: Maximum temperature value or None

    Returns:
        bool: True if both values are provided and valid, False otherwise

    Raises:
        ValueError: If both values are provided but min_temp >= max_temp
    """
    if min_temp is not None and max_temp is not None:
        if min_temp >= max_temp:
            raise ValueError("min_temp must be less than max_temp")
        return True
    return False
