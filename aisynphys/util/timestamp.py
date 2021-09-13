import time, datetime


def datetime_to_timestamp(d):
    return time.mktime(d.timetuple()) + d.microsecond * 1e-6
    

def timestamp_to_datetime(ts):
    return datetime.datetime.fromtimestamp(ts)
