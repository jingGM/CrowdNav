# Copyright 2017 The DRLCA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


#!/usr/bin/python
import math
import random

class VelocitySmoother:
    decel_factor = 1.0
    aecel_lim_x = 1.5
    aecel_lim_w = 2.0

    decel_lim_x = aecel_lim_x * decel_factor
    decel_lim_w = aecel_lim_w * decel_factor
    speed_lim_v = 1.0
    speed_lim_w = 1.0

    def __init__(self):
        self.last_vx = 0.
        self.last_vw = 0.

    def step(self,  vx , vw, excu_time):
        # bound the input first
        if vx > 0:
            target_vx = min(self.speed_lim_v, vx)
        else:
            target_vx = 0.
            #target_vx = max(-self.speed_lim_v, vx)
        if vw > 0:
            target_vw = min(self.speed_lim_w, vw)
        else:
            target_vw = max(-self.speed_lim_w, vw)

        if (target_vx != self.last_vx) and (target_vw != self.last_vw):
            v_inc = target_vx - self.last_vx
            w_inc = target_vw - self.last_vw

            if v_inc > 0:
                max_v_inc = excu_time * self.aecel_lim_x
            else:
                max_v_inc = excu_time * self.decel_lim_x

            if w_inc > 0:
                max_w_inc = excu_time * self.aecel_lim_w
            else:
                max_w_inc = excu_time * self.decel_lim_w

            MA = math.sqrt(v_inc * v_inc + w_inc * w_inc)
            MB = math.sqrt(max_v_inc * max_v_inc + max_w_inc * max_w_inc)

            Av = abs(v_inc) / MA
            Aw = abs(w_inc) / MA
            Bv = max_v_inc / MB
            Bw = max_w_inc / MB
            theta = math.atan2(Bw, Bv) - math.atan2(Aw, Av)

            if (theta < 0):
                max_v_inc = (max_w_inc * abs(v_inc)) / abs(w_inc)
            else:
                max_w_inc = (max_v_inc * abs(w_inc)) / abs(v_inc)

            final_cmd_vx = 0.
            final_cmd_vw = 0.
            if (abs(v_inc) > max_v_inc):
                final_cmd_vx = self.last_vx + math.copysign(max_v_inc, v_inc)
            else :
                final_cmd_vx = target_vx

            if (abs(w_inc) > max_w_inc):
                final_cmd_vw = self.last_vw + math.copysign(max_w_inc, w_inc)
            else :
                final_cmd_vw = target_vw

            self.last_vx = final_cmd_vx
            self.last_vw = final_cmd_vw
            return [final_cmd_vx, final_cmd_vw]
        else:
            return [target_vx, target_vw]

if __name__ == "__main__":
    smoother = VelocitySmoother()

    for i in range(10):
        cmd_x = random.uniform(-0.5, 0.5)
        cmd_w = random.uniform(-1.5, 1.5)
        print 0.3 + cmd_x, 0.7 + cmd_w
        print smoother.step(0.3 + cmd_x, 0.7 + cmd_w, 0.1)
        print "\n"
