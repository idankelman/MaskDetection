import datetime
import time
import json
def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    s = s.split('.')[0]
    return int(h) * 3600 + int(m) * 60 + int(s)
start = datetime.datetime.now()
formated = start.strftime("%H:%M:%S")
print(json.dumps(formated, indent=4, default=str))
time.sleep(3)
end = datetime.datetime.now()
duration = end - start
print(get_sec(str(duration)))