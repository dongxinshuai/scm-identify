import torch
import torch.nn as nn

def inv_sqrtm(A, eps=1e-9):
    """Compute the inverse square-root of a positive definite matrix."""
    # Perform eigendecomposition of covariance matrix
    U, S, V = torch.svd(A)
    # Enforce positive definite by taking a torch max() with eps
    S = torch.max(S, torch.tensor(eps, device=S.device))
    # Calculate inverse square-root
    inv_sqrt_S = torch.diag_embed(torch.pow(S, -0.5))
    # Calculate inverse square-root matrix
    B = torch.matmul(torch.matmul(U, inv_sqrt_S), V.transpose(-1, -2))
    return B


class CCA_eigvals:
    """Differentiable CCA Loss. Solves the CCA problem."""

    def __init__(self, r: float = 0, eps: float = 1e-3):
        self.r = r
        self.eps = eps

    def get_eigvals(self, X, Y):
        """Calculate correlation."""
        latent_dims = X.shape[1]
        o1 = X.shape[1]
        o2 = Y.shape[1]

        SigmaHat12 = torch.cov(torch.hstack((X, Y)).T)[
            :latent_dims, latent_dims:
        ]
        SigmaHat11 = torch.cov(X.T) + self.r * torch.eye(
            o1, device=X.device
        )
        SigmaHat22 = torch.cov(Y.T) + self.r * torch.eye(
            o2, device=Y.device
        )

        SigmaHat11RootInv = inv_sqrtm(SigmaHat11, self.eps)
        SigmaHat22RootInv = inv_sqrtm(SigmaHat22, self.eps)

        Tval = SigmaHat11RootInv @ SigmaHat12 @ SigmaHat22RootInv
        trace_TT = Tval.T @ Tval
        eigvals = torch.linalg.eigvalsh(trace_TT)

        return eigvals

#    def loss(self, X, Y):
#        """Calculate loss."""
#        eigvals = self.get_eigvals(X, Y)
#        #eigvals = eigvals[torch.gt(eigvals, 0)]
#        return -eigvals.sum()


class PolynomialNN(nn.Module):
    def __init__(self, degree, init_mode):
        super(PolynomialNN, self).__init__()
        self.poly_layer = nn.Linear(degree, 1, bias=True)  # Learnable coefficients
        self.init_mode = init_mode

        if init_mode==0:
            with torch.no_grad():
                self.poly_layer.weight.copy_((torch.rand(degree)-0.5)/1000)
                self.poly_layer.bias.copy_((torch.rand(1)-0.5)/1000) 
        elif init_mode==1:
            with torch.no_grad():
                self.poly_layer.weight.copy_((torch.rand(degree)-0.5)/1000)
                self.poly_layer.bias.copy_((torch.rand(1)-0.5)/1000+1) 

    def forward(self, x):
        poly_features = torch.cat([x ** i for i in range(1, self.poly_layer.in_features + 1)], dim=1)
        return self.poly_layer(poly_features)

class PolynomialMatrix(nn.Module):
    def __init__(self, degree, dim1, dim2):
        super(PolynomialMatrix, self).__init__()
        self.degree = degree
        self.dim1 = dim1
        self.dim2 = dim2
        self.layers = nn.ModuleList()
        for i in range(dim1):
            for j in range(dim2):
                if i==j:
                    self.layers.append(PolynomialNN(degree=self.degree, init_mode=1)) 
                else:
                    self.layers.append(PolynomialNN(degree=self.degree, init_mode=0)) 

    def forward(self, x):
        batch_size = x.shape[0]
        matrix = torch.zeros((batch_size, self.dim1, self.dim2), device=x.device)
        for i in range(self.dim1):
            for j in range(self.dim2):
                matrix[:, i, j]=self.layers[i*self.dim2+j](x).reshape(-1)

        return matrix

class RationalMatrix(nn.Module):
    def __init__(self, degree, dim1, dim2):
        super(RationalMatrix, self).__init__()
        self.degree = degree
        self.dim1 = dim1
        self.dim2 = dim2
        self.layersP = nn.ModuleList()
        self.layersQ = nn.ModuleList()
        for i in range(dim1):
            for j in range(dim2):
                if i==j:
                    self.layersP.append(PolynomialNN(degree=self.degree, init_mode=1)) 
                    self.layersQ.append(PolynomialNN(degree=self.degree, init_mode=1)) 
                else:
                    self.layersP.append(PolynomialNN(degree=self.degree, init_mode=0)) 
                    self.layersQ.append(PolynomialNN(degree=self.degree, init_mode=1)) 

    def forward(self, x):
        batch_size = x.shape[0]
        matrix = torch.zeros((batch_size, self.dim1, self.dim2), device=x.device)
        for i in range(self.dim1):
            for j in range(self.dim2):
                matrix[:, i, j]=self.layersP[i*self.dim2+j](x).reshape(-1) / self.layersQ[i*self.dim2+j](x).reshape(-1)

        return matrix

class CCA_with_t(nn.Module):
    def __init__(self, degree, dimX, dimY):
        super(CCA_with_t, self).__init__()
        self.degree = degree
        self.dimX = dimX
        self.dimY = dimY

        self.A = RationalMatrix(self.degree, dimX, dimX)
        self.B = RationalMatrix(self.degree, dimY, dimY)
        self.AA = PolynomialMatrix(self.degree, dimX, dimX)
        self.BB = PolynomialMatrix(self.degree, dimY, dimY)        

        self.cca = CCA_eigvals()

    def forward(self, X, Y, t):

        #eigvals_ori = self.cca.get_eigvals(X, Y)

        '''
        X = X.reshape(X.shape[0], X.shape[1],1)
        Y = Y.reshape(Y.shape[0], Y.shape[1],1)

        try:
            X = torch.bmm(torch.linalg.inv(self.A(t)), X)
        except:
            X = torch.bmm(torch.linalg.pinv(self.A(t)), X)

        try:
            Y = torch.bmm(torch.linalg.inv(self.B(t)), Y)
        except:
            Y = torch.bmm(torch.linalg.pinv(self.B(t)), Y)
        
        #X = torch.bmm(self.A(t), X)
        #Y = torch.bmm(self.B(t), Y)
        X = X.reshape(X.shape[0], X.shape[1])
        Y = Y.reshape(Y.shape[0], Y.shape[1])
        '''

        AA = torch.zeros(t.shape[0], self.dimX)
        tempAA = self.AA(t)
        for i in range(self.dimX):
            AA[:,i] = tempAA[:,i,i]
        X = X / AA

        BB = torch.zeros(t.shape[0], self.dimY)
        tempBB = self.BB(t)
        for i in range(self.dimY):
            BB[:,i] = tempBB[:,i,i]
        Y = Y / BB

        eigvals = self.cca.get_eigvals(X, Y)

        return torch.flip(eigvals,[0])
    
    def loss_nonsingular_AB(self, t):
        A = self.A(t)
        B = self.B(t)

        def get_loss(M):
            threshold = 0.05
            U, S, V = torch.linalg.svd(M)
            return torch.sum(torch.relu(threshold - S)) 

        loss = get_loss(A)+get_loss(B)
        return loss
    
    def loss_nonsingular_AABB(self, t):
        AA = torch.zeros(t.shape[0], self.dimX)
        tempAA = self.AA(t)
        for i in range(self.dimX):
            AA[:,i] = tempAA[:,i,i]

        BB = torch.zeros(t.shape[0], self.dimY)
        tempBB = self.BB(t)
        for i in range(self.dimY):
            BB[:,i] = tempBB[:,i,i]

        epsilon = 5e-2
        return torch.sum(torch.relu(epsilon - torch.abs(AA))) + torch.sum(torch.relu(epsilon - torch.abs(BB)))


    def loss_init_AB(self, t):

        return ((self.A(t) - torch.eye(self.dimX))**2).mean() + ((self.B(t) - torch.eye(self.dimY))**2).mean()

    