from datetime import datetime


# Note that the makeDateReadable() function only works on a single value, not
# and entire array/series.
#
# A good way to convert the entire date column to something readable is:
#
# df.date = df.date.astype('int').astype("datetime64[s]")
#
def makeDateReadable(timestamp, longform=False):
    if type(timestamp) is datetime:
        timestamp = timestamp.timestamp()
    if not longform:
        return datetime.fromtimestamp(timestamp).strftime("%m/%d/%y %I:%M%p")
    else:
        return datetime.fromtimestamp(timestamp).strftime("%m/%d/%y %I:%M:%S%p")


def calculatePrevMonth(timestamp):
    last_month_num = datetime.fromtimestamp(int(timestamp)).month
    last_year = datetime.fromtimestamp(int(timestamp)).year
    if last_month_num > 1:
        return datetime(year=last_year, month=last_month_num-1, day=1)
    else:
        return datetime(year=last_year-1, month=12, day=1)
