import numpy as np

class PiecewiseModel:
    '''
    Color characterization model for OLED displays with crosstalk effects
    '''
    def __init__(self, break_points):
        '''
        break_points = np.array([
            [0.25, 0.5], # R
            [0.25, 0.5], # G
            [0.25, 0.5]  # B
        ])
        '''
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
        self.coeff = np.asarray(coeff) # 64x5
        return np.asarray(colors)

    def fit(self, deltaE):
        '''
        dletaE is 64x3 matrix for [X_CT, Y_CT, Z_CT] - [X, Y, Z]
        '''
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

        subspaces = np.asarray(subspaces) # 27x8

        deltaE = np.asarray(deltaE)
        Ms = []
        for subspace in subspaces:
            coeff = self.coeff[subspace].T # 5x8
            deltaE_subspace = deltaE[subspace].T # 3x8
            Ms_subspace = deltaE_subspace @ np.linalg.pinv(coeff)
            Ms.append(Ms_subspace)
        self.Ms = np.asarray(Ms) # 27x3x5

    def predict(self, rgb):
        r, g, b = rgb[:,0].copy(), rgb[:,1].copy(), rgb[:,2].copy()

        i = np.array([0 if x <= self.R[1] else (1 if x <= self.R[2] else 2) for x in r])
        j = np.array([0 if x <= self.G[1] else (1 if x <= self.G[2] else 2) for x in g])
        k = np.array([0 if x <= self.B[1] else (1 if x <= self.B[2] else 2) for x in b])

        subspace_id = i * 9 + j * 3 + k

        Ms = self.Ms[subspace_id] # nx3x5
        coeff = np.stack([np.ones_like(r), r*g, r*b, g*b, r*g*b], axis=-1) # nx5
        deltaE = np.einsum('nij,nj->ni', Ms, coeff)
        return deltaE
    

if __name__ == "__main__":
    model = PiecewiseModel(np.array([[0.25, 0.5],[0.25, 0.5], [0.25, 0.5]]))
    colors = np.asarray(model.generate_colors())
    # print(colors)
    
    deltaE = np.random.randn(64, 3) * 0.001

    model.fit(deltaE)

    pred = model.predict(colors)

    print(deltaE[1])
    print(model.Ms[0] @ model.coeff[1])
    print(model.Ms[1] @ model.coeff[1])



