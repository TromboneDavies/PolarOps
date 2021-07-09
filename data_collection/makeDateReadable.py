import datetime as dt


# Note that the makeDateReadable() function only works on a single value, not
# and entire array/series.
#
# A good way to convert the entire date column to something readable is:
#
# df.date = df.date.astype('int').astype("datetime64[s]")
#
def makeDateReadable(timestamp, longform=False):
    if not longform:
        return dt.datetime.fromtimestamp(timestamp).strftime("%m/%d/%y %I:%M%p")
    else:
        return dt.datetime.fromtimestamp(timestamp).strftime("%m/%d/%y %I:%M:%S%p")

