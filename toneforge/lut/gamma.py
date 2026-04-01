import numpy as np
import matplotlib.pyplot as plt

class DualGammaLut:
    def interp(self, input):
        raise NotImplementedError

    def invers_interp(self, input):
        raise NotImplementedError



class GammaLut:
    def __init__(self, func, input_bit, output_bit, lut_bit):
        self.rshift = input_bit - lut_bit
        self.resi_max = (1<<self.rshift) - 1
        self.rounding = 1<<(self.rshift-1) if self.rshift else 0

        self.Lut = np.minimum(
            func(np.array(range((1<<lut_bit)+1))/(1<<lut_bit))*(1<<output_bit)+0.5, (1<<output_bit)-1
        ).astype(np.uint32)

    def interp(self, input):
        index = input >> self.rshift

        y_pos0 = self.Lut[index]
        y_pos1 = self.Lut[index+1]

        y_value = y_pos1 - y_pos0
        x_value = input & self.resi_max

        output = np.zeros_like(input)

        mask = index < len(self.Lut)-2
        output[mask] = y_pos0[mask] + ((y_value[mask] * x_value[mask] + self.rounding) >> self.rshift)

        mask = ~mask
        if self.resi_max:
            output[mask] = y_pos0[mask] + ((y_value[mask] * x_value[mask] * (65535 // self.resi_max) + 32768) >> 16)
        else:
            output[mask] = y_pos0[mask]

        return output

    def invers_interp(self, input):
        def get_lut_index(ipt):
            return np.array([np.sum(self.Lut <= x) for x in ipt])
        
        index = get_lut_index(input) # np.array([get_lut_index(x1) for x1 in input])

        lut_min = self.Lut[index-1]
        lut_max = self.Lut[index]

        lut_base = np.where(index==0, 0, (index-1) << self.rshift)

        step = lut_max - lut_min
        resi = (input - lut_min) << (self.rshift + 1)

        resi_carry = (resi + step - 1)
        resi_carry[step==0] = 0
    

        level = np.zeros_like(input)
        level[step!=0] = resi_carry[step!=0] / step[step!=0]
        
        inc = level >> 1

        y = lut_base + inc
        return y



def getLut2(func, input_bit = 12, lut_bit = 24, step_bit1 = 4, step_bit2 = 6, bvalue = 128):
    # input_bit = 12
    lut_x1 = np.array(range(0, bvalue+(1<<step_bit1), 1<<step_bit1), dtype=np.float64)/((1<<input_bit)-1)
    lut_x2 = np.array(range(bvalue+(1<<step_bit2), (1<<input_bit)+(1<<step_bit2), 1<<step_bit2), dtype=np.float64)/((1<<input_bit)-1)

    lut1 = np.round(func(lut_x1) * ((1<<lut_bit)-1)).astype(np.uint64)
    lut2 = np.round(func(lut_x2) * ((1<<lut_bit)-1)).astype(np.uint64)
    lut2 = np.minimum(lut2, (1<<lut_bit)-1)

    return lut1, lut2

def lutEOTF(x, lut1, lut2, lut_bit:int = 24, step_bit1:int = 4, step_bit2:int = 6, bvalue:int = 128):
    x1 = x[x<=bvalue]
    x2 = x[x>bvalue]

    index1 = x1 >> step_bit1
    index2 = (x2-bvalue) >> step_bit2

    resi1 = x1 & ((1<<(step_bit1))-1)
    resi2 = x2 & ((1<<(step_bit2))-1)

    lut_min1 = lut1[index1]
    lut_min2 = np.piecewise(index2, [index2>0], [
        lambda idx: lut2[idx-1],
        lambda idx: lut1[-1],
    ])

    lut_max1 = np.piecewise(index1, [index1<(bvalue>>step_bit1)], [
        lambda idx: lut1[idx+1],
        lambda idx: lut1[-1]
    ])
    lut_max2 = lut2[index2]

    interp_val1 = resi1*lut_max1 + ((1<<step_bit1)-resi1)*lut_min1 + (1<<(step_bit1-1))
    interp_val2 = resi2*lut_max2 + ((1<<step_bit2)-resi2)*lut_min2 + (1<<(step_bit2-1))

    y1 = np.int64(interp_val1) >> step_bit1
    y2 = np.int64(interp_val2) >> step_bit2
    y2 = np.minimum(y2, (1<<lut_bit)-1)
    
    y = np.zeros_like(x, dtype=np.uint64)
    y[x<=bvalue] = y1
    y[x>bvalue] = y2

    return y

def getLutIndex(x, lut1, lut2, lut1_max):
    return np.piecewise(x, [x >= lut1_max], [
        lambda x: [np.sum(lut2<x2) for x2 in x],
        lambda x: [np.sum(lut1<x1) for x1 in x]
    ])

# def getLutIndex(x, lut):
#     return np.array([np.sum(lut<x1) for x1 in x])

def lutOETF(x, lut1, lut2, lut_bit = 24, step_bit1 = 4, step_bit2 = 6, bvalue:int = 128, output_bit = 12):

    lut1_max = lut1[-1]
    index = getLutIndex(x, lut1, lut2, lut1_max)

    index1_max = lut1.shape[0]
    index2_max = lut2.shape[0]
    lut_min = np.piecewise(index, [(x<lut1_max)&(index==0), (x<lut1_max)&(index>index1_max), 
                                   (x<lut1_max)&(index<=index1_max)&(index>0), 
                                   (x>=lut1_max)&(index==0), (x>=lut1_max)&(index>0)], [
        lambda x: 0,
        lambda x: lut1[-1],
        lambda x: lut1[x-1],
        lambda x: lut1[-1],
        lambda x: lut2[x-1]
    ])

    lut_max = np.piecewise(index, [(x<lut1_max)&(index>(index1_max-1)), (x<lut1_max)&(index<=(index1_max-1)), 
                                   (x>=lut1_max)&(index>=index2_max), (x>=lut1_max)&(index<index2_max)], [
        lambda x: (1<<lut_bit)-1,
        lambda x: lut1[x],
        lambda x: (1<<lut_bit)-1,
        lambda x: lut2[x]
    ])

    lut_base = np.piecewise(index, [(x<lut1_max)&(index==0), (x<lut1_max)&(index>0), 
                                    (x>=lut1_max)&(index>=index2_max), (x>=lut1_max)&(index<index2_max)], [
        lambda x: 0,
        lambda x: (x - 1) << step_bit1,
        lambda x: (1<<output_bit)-1,
        lambda x: (x + ((1<<output_bit)>>step_bit2) - index2_max) << step_bit2
    ])

    step = (lut_max - lut_min).astype(np.int64)
    resi = x - lut_min
    resi[x<lut1_max] <<= (step_bit1 + 1)
    resi[x>=lut1_max] <<= (step_bit2 + 1)
    
    resi_carry = (resi + step - 1).astype(np.int64)
    resi_carry[step==0] = 0

    level = np.zeros_like(x)
    level[step!=0] = np.minimum(resi_carry[step!=0] / step[step!=0], bvalue)
    

    inc = level >> 1
    inc[(x<lut1_max)&(level>((1<<(step_bit1+1))-1))] = (1<<step_bit1)
    print(step.max())
    
    y = np.minimum(lut_base + inc, (1<<output_bit)-1)
    return y

    