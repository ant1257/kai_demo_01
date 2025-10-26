def part_of_day(start_date_full):
    """
    Assigns a part-of-day category (1-8) based on the hour and minute of a datetime object.

    Parts of day:
    1 = late night      (00:00 - 02:59)
    2 = early morning   (03:00 - 05:59)
    3 = morning         (06:00 - 08:59)
    4 = late morning    (09:00 - 11:59)
    5 = midday          (12:00 - 14:59)
    6 = afternoon       (15:00 - 17:59)
    7 = evening         (18:00 - 20:59)
    8 = late evening    (21:00 - 23:59)

    Parameters:
        start_date_full (datetime): A datetime object.

    Returns:
        int: Integer (1-8) indicating the part of day.
    """
    total_minutes = start_date_full.hour * 60 + start_date_full.minute
    if total_minutes >= 0 and total_minutes < 3*60:
        # late_night
        return 1
    elif total_minutes >= 3*60 and total_minutes < 6*60:
        # early_morning
        return 2
    elif total_minutes >= 6*60 and total_minutes < 9*60:
        # moring
        return 3
    elif total_minutes >= 9*60 and total_minutes < 12*60:
        # late_morning
        return 4
    elif total_minutes >= 12*60 and total_minutes < 15*60:
        # midday
        return 5
    elif total_minutes >= 15*60 and total_minutes < 18*60:
        # afternoon
        return 6
    elif total_minutes >= 18*60 and total_minutes < 21*60:
        # evening
        return 7
    else:
        # late_evening
        return 8