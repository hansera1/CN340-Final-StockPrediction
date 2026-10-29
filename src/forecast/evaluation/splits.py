"""แบ่งตามเวลาเท่านั้น — ห้ามสุ่มแถวข้ามเวลา"""
import numpy as np


def walk_forward_splits(dates, test_block_days, min_train_days, holdout_days=0):
    """ฝึกบนทุกวันก่อนหน้า แล้วทดสอบบล็อกถัดไป (ช่วง holdout ท้ายสุดไม่ถูกแตะ)"""
    ud = np.unique(dates)
    usable = ud[:len(ud) - holdout_days] if holdout_days else ud
    for start in range(min_train_days, len(usable), test_block_days):
        yield ud[:start], usable[start:start + test_block_days]


def holdout_split(dates, holdout_days):
    """ใช้หลังล็อกวิธีแล้วเท่านั้น: ฝึกทุกวันก่อน holdout แล้วทดสอบ holdout ครั้งเดียว"""
    ud = np.unique(dates)
    yield ud[:-holdout_days], ud[-holdout_days:]
