import random
from datetime import timedelta

def make_diff_to_str(diff, target_aware=False):
    days = diff.days

    if not target_aware:
        if days == 0:
            # interval_str = "on the same day"
            seconds = diff.seconds
            if seconds < 3600:
                interval_str = f"after less than an hour"
            else:
                hours = seconds // 3600
                interval_str = f"after {hours} hour{'s' if hours > 1 else ''}"
        elif days == 1:
            interval_str = "after 1 day"
        elif days < 30:
            interval_str = f"after {days} days"
        elif days < 365:
            months = days // 30
            interval_str = f"after {months} month{'s' if months > 1 else ''}"
        else:
            years = days // 365
            interval_str = f"after {years} year{'s' if years > 1 else ''}"
    else:
        if days == 0:
            # interval_str = "on the same day"
            seconds = diff.seconds
            if seconds < 3600:
                interval_str = f"less than an hour ago"
            else:
                hours = seconds // 3600
                interval_str = f"before {hours} hour{'s' if hours > 1 else ''} ago"
        elif days == 1:
            interval_str = "1 day ago"
        elif days < 30:
            interval_str = f"{days} days ago"
        elif days < 365:
            months = days // 30
            interval_str = f"{months} month{'s' if months > 1 else ''} ago"
        else:
            years = days // 365
            interval_str = f"{years} year{'s' if years > 1 else ''} ago"
    return interval_str


def make_diff_to_str_flexible(diff, target_aware=False):
    days = diff.days

    if not target_aware:
        if days == 0:
            seconds = diff.seconds
            if seconds < 3600:
                return random.choice([
                    "after less than an hour",
                    "shortly after",
                    "a few minutes later"
                ])
            else:
                hours = seconds // 3600
                return random.choice([
                    f"after {hours} hour{'s' if hours > 1 else ''}",
                    f"{hours}h later",
                    f"around {hours} hours after"
                ])
        elif days == 1:
            return random.choice([
                "after 1 day",
                "the next day",
                "a day later"
            ])
        elif days < 30:
            return random.choice([
                f"after {days} days",
                f"{days} days later",
                "a few days after"
            ])
        elif days < 365:
            months = days // 30
            return random.choice([
                f"after {months} month{'s' if months > 1 else ''}",
                f"{months} months later",
                "a couple of months after"
            ])
        else:
            years = days // 365
            return random.choice([
                f"after {years} year{'s' if years > 1 else ''}",
                f"{years}y later",
                "some years after"
            ])
    else:
        if days == 0:
            seconds = diff.seconds
            if seconds < 3600:
                return random.choice([
                    "less than an hour ago",
                    "just now",
                    "a few minutes ago"
                ])
            else:
                hours = seconds // 3600
                return random.choice([
                    f"{hours} hour{'s' if hours > 1 else ''} ago",
                    f"{hours}h ago",
                    f"about {hours} hours ago"
                ])
        elif days == 1:
            return random.choice([
                "1 day ago",
                "yesterday",
                "a day ago"
            ])
        elif days < 30:
            return random.choice([
                f"{days} days ago",
                f"{days}d ago",
                "a few days ago"
            ])
        elif days < 365:
            months = days // 30
            return random.choice([
                f"{months} month{'s' if months > 1 else ''} ago",
                f"{months}m ago",
                "a couple of months ago"
            ])
        else:
            years = days // 365
            return random.choice([
                f"{years} year{'s' if years > 1 else ''} ago",
                f"{years}y ago",
                "some years ago"
            ])


def make_datetime_to_str(datetime_str):
    # 2011-10-18 -> October 18, 2011
    if not datetime_str or len(datetime_str) < 10:
        return datetime_str
        
    try:
        year = datetime_str[:4]
        month = datetime_str[5:7]
        day = datetime_str[8:10]
        
        month_names = {
            '01': 'Jan',
            '02': 'Feb',
            '03': 'Mar',
            '04': 'Apr',
            '05': 'May',  
            '06': 'Jun',
            '07': 'Jul',
            '08': 'Aug',
            '09': 'Sep',
            '10': 'Oct',
            '11': 'Nov',
            '12': 'Dec'
        }
        month_name = month_names.get(month, month)
        
        if day.startswith('0'):
            day = day[1:]
            
        date_str = f"{month_name} {day}, {year}"
        return date_str
    except:
        return datetime_str


# v1 ---------------------------------------------
# RECENCY_TOKENS = {
#     '<GAP_VERY_SHORT>': '<= 1 day',
#     '<GAP_SHORT>': '1 day < x <= 7 days',
#     '<GAP_MEDIUM>': '7 days < x <= 30 days',
#     '<GAP_LONG>': '30 days < x <= 365 days',
#     '<GAP_VERY_LONG>': 'x > 365 days'
# }

# def diff_to_recency_token(diff):
#     token = None
#     days = diff.days
#     if days <= 1:
#         token = '<GAP_VERY_SHORT>'
#     elif days <= 7:
#         token = '<GAP_SHORT>'
#     elif days <= 30:
#         token = '<GAP_MEDIUM>'
#     elif days <= 365:
#         token = '<GAP_LONG>'
#     else:
#         token = '<GAP_VERY_LONG>'
#     return token

RECENCY_TOKENS = {
    '<GAP_VERY_SHORT>': '<= 1 day',
    '<GAP_SHORT>': '1 day < x <= 30 days',
    '<GAP_MEDIUM>': '30 days < x <= 180 days',
    '<GAP_LONG>': 'x > 180 days'
}

def diff_to_recency_token(diff):
    token = None
    days = diff.days
    if days <= 1:
        token = '<GAP_VERY_SHORT>'
    elif days <= 30:
        token = '<GAP_SHORT>'
    elif days <= 180:
        token = '<GAP_MEDIUM>'
    else:
        token = '<GAP_LONG>'
    return token


def diff_to_bucket(diff):
    idx = None
    days = diff.days
    if days <= 1:  # within a day
        idx = 0
    elif days <= 30:  # within a month
        idx = 1
    elif days <= 365:  # within a year
        idx = 2
    else:  # over a year
        idx = 3
    return idx


TIME_BUCKETS = {
    0: 'within 1 day',            # x ≤ 1 day
    1: '2 to 30 days',   # 1 < x ≤ 30 days
    2: '1 to 12 months', # 30 < x ≤ 365 days
    3: 'more than 1 year',        # x > 365 days
}