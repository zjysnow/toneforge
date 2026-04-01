import numpy as np

class PiecewiseModel:
    def __init__(self, break_points):
        self.R = np.r_[0, break_points[0], 1]
        self.G = np.r_[0, break_points[1], 1]
        self.B = np.r_[0, break_points[2], 1]

    def generate_colors(self):
        colors = []
        coeff = []
        for r in self.R:
            for g in self.G:
                for b in self.B:
                    colors.append(np.array([r, g, b]))
                    coeff.append(np.array([1, r*g, r*b, g*b, r*g*b]))
        self.coeff = np.asarray(coeff)  # 64×5
        return np.asarray(colors)

    def _build_subspaces(self):
        subspaces = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    corners = [
                        (i,   j,   k),
                        (i+1, j,   k),
                        (i,   j+1, k),
                        (i+1, j+1, k),
                        (i,   j,   k+1),
                        (i+1, j,   k+1),
                        (i,   j+1, k+1),
                        (i+1, j+1, k+1),
                    ]
                    subspaces.append([r*16 + g*4 + b for (r, g, b) in corners])
        return np.asarray(subspaces)

    def fit(self, deltaE, lambda_cont=1000.0):
        deltaE = np.asarray(deltaE)
        subspaces = self._build_subspaces()

        # Unknown vector x has length 27*3*5 = 405
        A_list = []
        b_list = []

        # --- 1. Data fitting equations ---
        for s, idxs in enumerate(subspaces):
            C = self.coeff[idxs].T        # 5×8
            DE = deltaE[idxs].T           # 3×8

            for corner in range(8):
                c = C[:, corner]          # 5
                de = DE[:, corner]        # 3

                for ch in range(3):
                    row = np.zeros(27*15)
                    base = s*15 + ch*5
                    row[base:base+5] = c
                    A_list.append(row)
                    b_list.append(de[ch])

        # --- 2. Continuity constraints ---
        vertex_to_subspaces = {v: [] for v in range(64)}
        for s, idxs in enumerate(subspaces):
            for v in idxs:
                vertex_to_subspaces[v].append(s)

        for v, s_list in vertex_to_subspaces.items():
            if len(s_list) <= 1:
                continue

            c = self.coeff[v]  # 5

            for i in range(len(s_list) - 1):
                s1 = s_list[i]
                s2 = s_list[i+1]

                for ch in range(3):
                    row = np.zeros(27*15)
                    base1 = s1*15 + ch*5
                    base2 = s2*15 + ch*5
                    row[base1:base1+5] = c
                    row[base2:base2+5] = -c
                    A_list.append(np.sqrt(lambda_cont) * row)
                    b_list.append(0.0)

        # --- Solve global least squares ---
        A = np.vstack(A_list)
        b = np.asarray(b_list)

        x, *_ = np.linalg.lstsq(A, b, rcond=None)

        # reshape back to Ms
        Ms = []
        for s in range(27):
            block = x[s*15:(s+1)*15]
            Ms.append(block.reshape(3, 5))
        self.Ms = np.asarray(Ms)

    def predict(self, rgb):
        r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]

        i = np.clip(np.searchsorted(self.R, r, side='right') - 1, 0, 2)
        j = np.clip(np.searchsorted(self.G, g, side='right') - 1, 0, 2)
        k = np.clip(np.searchsorted(self.B, b, side='right') - 1, 0, 2)

        subspace_id = i * 9 + j * 3 + k

        # print(subspace_id)

        Ms = self.Ms[subspace_id]  # n×3×5
        coeff = np.stack([np.ones_like(r), r*g, r*b, g*b, r*g*b], axis=-1)
        deltaE = np.einsum('nij,nj->ni', Ms, coeff)
        return deltaE


if __name__ == "__main__":
    model = PiecewiseModel(np.array([[0.25, 0.5],[0.25, 0.5], [0.25, 0.5]]))
    colors = model.generate_colors()

    # random deltaE
    deltaE = np.random.randn(64, 3) * 0.1

    model.fit(deltaE, 400)
    pred = model.predict(colors)


    print("predicted delta 21 with Ms[0]:", model.Ms[0] @ model.coeff[21])
    print("predicted delta 21 with Ms[1]:", model.Ms[1] @ model.coeff[21])
    print("predicted delta 21 with Ms[3]:", model.Ms[3] @ model.coeff[21])
    print("predicted delta 21 with Ms[4]:", model.Ms[4] @ model.coeff[21])
    print("predicted delta 21 with Ms[9]:", model.Ms[9] @ model.coeff[21])
    print("predicted delta 21 with Ms[10]:", model.Ms[10] @ model.coeff[21])
    print("predicted delta 21 with Ms[12]:", model.Ms[12] @ model.coeff[21])
    print("predicted delta 21 with Ms[13]:", model.Ms[13] @ model.coeff[21])
    print("real delta 21:", deltaE[21])

    # print(model.Ms)

    # print(deltaE)
    # print(pred)
    # print("max error:", np.max(np.abs(pred - deltaE)))

