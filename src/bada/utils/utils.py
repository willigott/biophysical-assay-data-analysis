import itertools


def get_row_labels(plate_size: int) -> list[str]:
    if plate_size == 384:
        return list("ABCDEFGHIJKLMNOP")
    elif plate_size == 96:
        return list("ABCDEFGH")

    raise ValueError(f"Unsupported plate size: {plate_size}")


def get_column_labels(plate_size: int, as_str: bool = True) -> list[str | int]:
    if plate_size == 384:
        if as_str:
            return [str(i) for i in range(1, 25)]
        else:
            return list(range(1, 25))
    elif plate_size == 96:
        if as_str:
            return [str(i) for i in range(1, 13)]
        else:
            return list(range(1, 13))

    raise ValueError(f"Unsupported plate size: {plate_size}")


def get_well_ids(plate_size: int) -> list[str]:
    """Get well IDs for a given plate size."""
    rows = get_row_labels(plate_size)
    cols = get_column_labels(plate_size, as_str=True)

    all_combinations = itertools.product(rows, cols)

    return [f"{row}{col}" for row, col in all_combinations]
