import collections
import numpy as np

class SCIP:
    class Entry:
        def __init__(self, value, pos="M", freq=1, time=0, is_new=True):
            self.value = value
            self.freq = freq
            self.time = time
            self.evicted_time = None
            self.is_demoted = False
            self.is_new = is_new
            self.pos = pos

    class Learning_Rate:
        def __init__(self, period_length):
            self.learning_rate = np.sqrt((2.0 * np.log(2)) / period_length)
            self.learning_rate_reset = min(max(self.learning_rate, 0.001), 1)
            self.learning_rate_curr = self.learning_rate
            self.learning_rate_prev = 0.0
            self.learning_rates = []
            self.period_len = period_length
            self.hitrate = 0
            self.hitrate_prev = 0.0
            self.hitrate_diff_prev = 0.0
            self.hitrate_zero_count = 0
            self.hitrate_nega_count = 0
        
        def __mul__(self, other):
            return self.learning_rate * other

        def update(self, time):
            if time % self.period_len == 0:
                hitrate_curr = round(self.hitrate / float(self.period_len), 3)
                hitrate_diff = round(hitrate_curr - self.hitrate_prev, 3)

                delta_LR = round(self.learning_rate_curr, 3) - round(
                    self.learning_rate_prev, 3)
                delta, delta_HR = self.updateInDeltaDirection(
                    delta_LR, hitrate_diff)

                if delta > 0:
                    self.learning_rate = min(
                        self.learning_rate +
                        abs(self.learning_rate * delta_LR), 1)
                    print(self.learning_rate)
                    self.hitrate_nega_count = 0
                    self.hitrate_zero_count = 0
                elif delta < 0:
                    self.learning_rate = max(
                        self.learning_rate -
                        abs(self.learning_rate * delta_LR), 0.001)
                    print(self.learning_rate)
                    self.hitrate_nega_count = 0
                    self.hitrate_zero_count = 0
                elif delta == 0 and hitrate_diff <= 0:
                    # print(self.learning_rate)
                    if (hitrate_curr <= 0 and hitrate_diff == 0):
                        self.hitrate_zero_count += 1
                    if hitrate_diff < 0:
                        self.hitrate_nega_count += 1
                        self.hitrate_zero_count += 1
                    if self.hitrate_zero_count >= 10:
                        self.learning_rate = self.learning_rate_reset
                        self.hitrate_zero_count = 0
                    elif hitrate_diff < 0:
                        if self.hitrate_nega_count >= 10:
                            self.learning_rate = self.learning_rate_reset
                            self.hitrate_nega_count = 0
                        else:
                            self.updateInRandomDirection()
                self.learning_rate_prev = self.learning_rate_curr
                self.learning_rate_curr = self.learning_rate
                self.hitrate_prev = hitrate_curr
                self.hitrate_diff_prev = hitrate_diff
                self.hitrate = 0

            self.learning_rates.append(self.learning_rate)

        # Update the learning rate according to the change in learning_rate and hitrate
        def updateInDeltaDirection(self, learning_rate_diff, hitrate_diff):
            delta = learning_rate_diff * hitrate_diff
            # Get delta = 1 if learning_rate_diff and hitrate_diff are both positive or negative
            # Get delta =-1 if learning_rate_diff and hitrate_diff have different signs
            # Get delta = 0 if either learning_rate_diff or hitrate_diff == 0
            delta = int(delta / abs(delta)) if delta != 0 else 0
            delta_HR = 0 if delta == 0 and learning_rate_diff != 0 else 1
            return delta, delta_HR

        # Update the learning rate in a random direction or correct it from extremes
        def updateInRandomDirection(self):
            if self.learning_rate >= 1:
                self.learning_rate = 0.9
            elif self.learning_rate <= 0.001:
                self.learning_rate = 0.005
            elif np.random.choice(['Increase', 'Decrease']) == 'Increase':
                self.learning_rate = min(self.learning_rate * 1.25, 1)
            else:
                self.learning_rate = max(self.learning_rate * 0.75, 0.001)

    def __init__(self, c):
        # Randomness and Time
        np.random.seed(123)
        self.time = 0

        # Cache
        self.capacity = c
        self.cache = collections.OrderedDict()
        self.used_size = 0

        self.m_hist = collections.OrderedDict()
        self.l_hist = collections.OrderedDict()
        self.history_capacity = c // 2
        self.m_hist_used_size = 0
        self.l_hist_used_size = 0

        # Decision Weights Initilized
        self.initial_weight = 0.5

        # Learning Rate
        self.learning_rate = self.Learning_Rate(100000)

        # Decision Weights
        self.w = np.array([self.initial_weight, 1 - self.initial_weight], dtype=np.float32)

        self.hit_number = 0
        self.total_number = 0
        self.hit_byte = 0
        self.total_byte = 0

        self.nor_count = 0

    # Adjust the weights based on the given rewards for MRU and LRU
    def adjustWeights(self, rewardMRU, rewardLRU):
        reward = np.array([rewardMRU, rewardLRU], dtype=np.float32)
        self.w = self.w * np.exp(self.learning_rate * reward)
        self.w = self.w / np.sum(self.w)

        if self.w[0] >= 0.99:
            self.w = np.array([0.99, 0.01], dtype=np.float32)
        elif self.w[1] >= 0.99:
            self.w = np.array([0.01, 0.99], dtype=np.float32)

    def request(self, key, value):
        self.total_byte += value
        self.total_number += 1
        self.time += 1

        self.learning_rate.update(self.time)

        if key in self.cache:
            self.hit_byte += value
            self.hit_number += 1
            self.promote(key)
        elif key in self.m_hist:
            history_entry = self.m_hist.pop(key)
            self.m_hist_used_size -= history_entry.value
            freq = history_entry.freq + 1
            if history_entry.is_new:
                self.nor_count -= 1
                self.is_new = False
            self.adjustWeights(-1, 0)
        elif key in self.l_hist:
            history_entry = self.l_hist.pop(key)
            self.l_hist_used_size -= history_entry.value
            freq = history_entry.freq + 1
            self.adjustWeights(0, -1)
        else:
            self.evict(value)
            self.insert(key, value)

        self.Learning_Rate.hitrate = self.hit_byte/self.total_byte
    
    def promote(self, key):
        self.cache[key].freq += 1
        self.cache[key].time = self.time
        policy = "M" if np.random.rand() < self.w[0] else "L"
        if policy == "M":
            self.cache.move_to_end(key)
        else:
            self.cache[key].pos = "L"
            self.cache.move_to_end(key, last=False)
    
    def evict(self, v):
        while self.used_size + v > self.capacity:
            self.is_full = True
            if len(self.cache) > 0:
                del_entry = self.cache.popitem(last=False)
                del_key = del_entry[0]
                del_value = del_entry[1].value
                self.used_size -= del_value
                del_pos = del_entry[1].pos
                del_entry[1].evicted_time = self.time

                if del_pos == "M":
                    while del_value + self.m_hist_used_size > self.history_capacity:
                        if len(self.m_hist) > 0:
                            del_histoty_entry = self.m_hist.popitem(last=False)
                            del_histoty_value = del_histoty_entry[1].value
                            self.m_hist_used_size -= del_histoty_value
                        else:
                            break
                    self.m_hist[del_key] = del_entry[1]
                    self.m_hist_used_size += del_value
                else:
                    while del_value + self.l_hist_used_size > self.history_capacity:
                        if len(self.l_hist) > 0:
                            del_histoty_entry = self.l_hist.popitem(last=False)
                            del_histoty_value = del_histoty_entry[1].value
                            self.l_hist_used_size -= del_histoty_value
                        else:
                            break
                    self.l_hist[del_key] = del_entry[1]
                    self.l_hist_used_size += del_value
            else:
                break
    
    def insert(self, key, value):
        policy = "M" if np.random.rand() < self.w[0] else "L"
        if policy == "M":
            entry = self.Entry(value, 1, self.time, "M")
            self.cache[key] = entry
            self.used_size += value
        else:
            entry = self.Entry(value, 1, self.time, "L")
            self.cache[key] = entry
            self.used_size += value
            self.cache.move_to_end(key, last=False)
    
    def get_missrate(self):
        return round((self.total_number-self.hit_number)*100/self.total_number, 4), round((self.total_byte-self.hit_byte)*100/self.total_byte, 4)