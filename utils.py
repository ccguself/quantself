import datetime

def trans_ExTime(t):
    t1 = int(int(t) // 1e9)
    t = (int(t) - t1 * 1e9)
    hour = int(t // 1e7)
    minute = int((t - hour * 1e7) // 1e5)
    second = int((t - hour * 1e7 - minute * 1e5) // 1e3)
    microsecond = int((t - hour * 1e7 - minute * 1e5 - second * 1e3) // 1e1)
    
    # 部分数据存在60s的情况
    if second == 60:
        minute += 1
        second = 0
        microsecond = 0
    if minute == 60:
        hour += 1
        minute = 0
    if hour == 24:
        hour = 0
    t = datetime.time(hour, minute, second, microsecond)
    return t