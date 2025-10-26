def season_of_year(month):
    """
    Assigns a season number (1-4) based on the month.

    Seasons:
    1 = Winter     (January, February, December)
    2 = Spring     (March, April, May)
    3 = Summer     (June, July, August)
    4 = Autumn     (September, October, November)

    Parameters:
        month (int): The month as an integer (1-12).

    Returns:
        int: Integer representing the season (1-4).
    """

    if month in [1, 2, 12]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4