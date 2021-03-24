from datetime import datetime

from pytz import timezone
from pytz import utc


def get_pst_time():
    date_format = "%m_%d_%Y_%H_%M_%S_%Z"
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone("US/Pacific"))
    pstDateTime = date.strftime(date_format)
    return pstDateTime
