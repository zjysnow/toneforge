import numpy as np

class GammaLut:
    def __init__(self, func, input_bit, output_bit, lut_bit):
        self.rshift = input_bit - lut_bit
        self.resi_max = (1<<self.rshift) - 1
        self.rounding = 1<<(self.rshift-1)

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


