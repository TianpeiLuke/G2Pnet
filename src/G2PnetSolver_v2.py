# coding: utf-8
 
import GPy 
import GPy.util as gutil
import numpy as np
import scipy as sp
import warnings
 
class G2PnetSolver:
    """
      
       Grassmannian Gaussian Process Network (G2P-net) solver
   
       :param X: input observations, with rows as samples
       :param kernel: GPy.kern.Kern type  module 
       :param U: latent variables of the observation  
       :       
 
    """
 
    def __init__(self, X, U, kernel, U_ref, eta=0.01, lambda_var = 1, tol=1e-3, max_iters=10, max_iters_kernel=100, add_regularizer=False,  normalize_X = False ):
       if len(X.shape) == 1:
           X = X.reshape(-1,1)
           warnings.warn("One dimensional observation (N,) being shaped as (N,1)")
       if len(U.shape) == 1:
           U = U.reshape(-1,1)
           warnings.warn("One dimensional input (N,) being shaped as (N,1)")
       
       self.U = U
       self._init_U = U

       assert U_ref.shape == U.shape, "The reference U should be of the same size as U_init"

       self.U_ref = U_ref
       self._init_U_ref = U_ref

       nsample, self.ndim_latent = U.shape    
 
       self.X = X
       self.nsample, self.ndim = X.shape
       assert self.nsample == nsample, "The number of samples does not match." 
       
       if normalize_X:
           self._Xoffset = np.mean(X,axis=1)
           self._Xscale  = np.std(X, axis=1)
           self.X = (X.copy() - self._Xoffset) / self._Xscale
       
       self._init_X = X
       self.XX = gutil.linalg.tdot(self.X)

       assert isinstance(kernel, GPy.kern.Kern), "Use the GPy.kern.Kern type module"  
       assert kernel.input_dim == self.ndim_latent , "The input dimension of U and kernel should match" 
       self.kern = kernel
       self._kern = self.kern
       
       self.max_iters_kernel = int(max_iters_kernel)
       print("Kernel initialization ...")
       self.kern, _, self.KiX2 = self.kernel_update(X=self.X, U=self.U, kernel=self.kern)
       self.negative_loglikelihood_update()
              
       self.lambda_var = lambda_var
       self.add_regularizer = add_regularizer

       self.eta = float(eta)
       self.max_iters = int(max_iters)
       self._max_iters = self.max_iters
       self.tol = float(tol)
       
       self.hist_kern = list()
       self.hist_nll = np.zeros([max_iters])
       self.hist_eta = np.zeros([max_iters])
          
    
    

    def optimize(self, max_iters = 10, tol=1e-3, init_mode = "fixed", optimizor="grad_descent", verbose=False):
       '''
     
        The main function of optimization procedure. Calls optimize_gradient_descent if the gradient descent is required 


       '''


       if optimizor == "grad_descent_grass":
           return self.optimize_gradient_descent_grass(max_iters, tol, verbose)
       elif optimizor == "grad_descent":
           return self.optimize_gradient_descent_Euclidean(max_iters, tol, verbose)  



 
    def optimize_gradient_descent_grass(self, max_iters=10, tol=1e-3, init_mode = "fixed", verbose=False):
       '''

         Optimization procedure using gradient descent on manifold.


       '''
       self.max_iters = max_iters
       self.tol = tol
       self.hist_kern = list()
       self.hist_nll = np.zeros([max_iters])
       self.hist_eta = np.zeros([max_iters])

       messages = verbose
    
       eta = self.eta*np.ones([max_iters])
       self.hist_eta = eta
       for i, eta_i in enumerate(eta):
           self.hist_kern.append(self.kern)
           self.hist_nll[i] = self.negative_loglikelihood
           print("[step {0:3d}] neg-log-likelihood {1:9f} stepsize {2:5.3f}".format(i, self.negative_loglikelihood, eta_i))
           # compute the grassmannian gradient
           self._gradient_U_Grassmann(mode=1)
           # update the subspace through geodesic on Grassmann manifold     
           U_temp = Grassmann_update(self.U, self.H, eta_i)
           self.U = U_temp
           # update the other kernel function
           self.kern, _ , _ = self.kernel_update(X=self.X, U=self.U, kernel=self.kern, messages= messages)

           self.negative_loglikelihood_update()
       
       return (self.hist_nll, self.hist_eta, self.hist_kern)



    def optimize_gradient_descent_Euclidean(self, max_iters=10, tol=1e-3, init_mode = "fixed", verbose=False):
       '''

         Optimization procedure using gradient descent on Eulidean space.


       '''
       self.max_iters = max_iters
       self.tol = tol
       self.hist_kern = list()
       self.hist_nll = np.zeros([max_iters])
       self.hist_eta = np.zeros([max_iters])
 
       messages = verbose
     
       eta = self.eta*np.ones([max_iters])
       self.hist_eta = eta
       for i, eta_i in enumerate(eta):
           self.hist_kern.append(self.kern)
           self.hist_nll[i] = self.negative_loglikelihood
           print("[step {0:3d}] neg-log-likelihood {1:9f} stepsize {2:5.3f}".format(i, self.negative_loglikelihood, eta_i))
           # compute the grassmannian gradient
           #self._gradient_U_Grassmann(mode=1)
           self._gradient_U_Euclidean(mode=1)
           # update the matrix U via gradient descent 
           U_temp = Euclidean_update(self.U, self.H, eta_i)
           self.U = U_temp
           # update the other kernel function
           self.kern, _ , _ = self.kernel_update(X=self.X, U=self.U, kernel=self.kern, messages= messages)
           self.negative_loglikelihood_update()
       
       return (self.hist_nll, self.hist_eta, self.hist_kern)


    def kernel_update(self, U, X, kernel, messages = True):
       '''

           Given U, X, return the self.kern from GPRegression 

       '''
       nsample , ndim = X.shape 
       nsample2 , ndim_latent = U.shape
        
       assert nsample == nsample2, "The number of rows of X and U should match"
       assert isinstance(kernel, GPy.kern.Kern), "Use the GPy.kern.Kern type module"  
       assert kernel.input_dim == self.ndim_latent , "The input dimension of U and kernel should match" 

       model = GPy.models.GPRegression(X= U, Y = X, kernel = kernel)
      
       model.optimize(max_iters = self.max_iters_kernel, messages=messages)
       kern = model.copy().kern
       
       Kernel_mat = kern.K(U, U)             
       XX = gutil.linalg.tdot(X)

       KiX2 , _ = gutil.linalg.dpotrs(np.asfortranarray(Kernel_mat), np.asfortranarray(XX), lower=1)
       return (kern, XX, KiX2)


    def _gradient_K(self, mode=0):
       '''
  
          Learn the gradient of the negative log-likelihood of Gaussian Process with respect to the kernel matrix K. Require GPy package installed. 
   
           Try:  pip install GPy
 
            .. math::
    
               \frac{dL}{dK} = 2*T - diag(diag(T)) 
 
               T = (ndim*K^{-1} - K^{-1}*X*X'*K^{-1})
         :param mode: =0 in test mode, =1 in working mode
         :param Kernel:  nsample x nsamples  positive definite matrix with i,j elemen                    being K(Zi, Zj)
         :type Kernel:  numpy.ndarray
         :param X: nsample x ndim data matrix of observations of Gaussian Process
         :type X: numpy.ndarray
         :param nsample: number of samples
         :type nsample: float
         :param ndim: dimension of observation data
         :type ndim: int
  
       '''
       Kernel = self.kern.K(self.U, self.U)
       S = self.XX
       try:
           KiX2 = self.KiX2
       except AttributeError:
           print("No KiX2 stored. Recompute")
           KiX2 , _ = gutil.linalg.dpotrs(np.asfortranarray(Kernel), np.asfortranarray(S), lower=1)
       else:
           if(mode == 1):
               KiX2 , _ = gutil.linalg.dpotrs(np.asfortranarray(Kernel), np.asfortranarray(S), lower=1)
      # KiS = K^{-1}*X*X'
       
       KiXXiK , _ = gutil.linalg.dpotrs(np.asfortranarray(Kernel), np.asfortranarray(KiX2.T), lower=1)
       # KiXXiK: K^{-1}*X*X'*K^{-1}
     
       Ki , _ = gutil.linalg.dpotri(np.asfortranarray(Kernel), lower=1)
       dL_dK_0 = 0.5*(self.ndim*Ki - KiXXiK)
       self.dL_dK =  2*dL_dK_0 - np.diag(np.diag(dL_dK_0)) 
         
       return self.dL_dK 
     
     

     
    def _rbf_kernel_gradient_U(self, mode=0):
       '''
           
          Compute the gradient of RBF kernel matrix with respect to the input matrix U.
           
         .. math::
     
               \frac{dK}{dU} = -\frac{variance}{lengthscale^2}\exp( -\frac{1}{2*lengthscale^2} \| U_{m} - U_{n} \|^2  )*2[U_{m,,l}- U_{,m,l}]
          
          :param mode: =0 in test mode, =1 in working mode
          :param U: nsample x ndim_latent matrix. Each row corresponds to a latent input of Gaussian Process sample function.
          :type U: numpy.ndarray
          :param lengthscale: corresponds to $K(r) = variance* \exp(-\frac{1}{2*lengthscale^2}r^2)$ 
          :type lengthscale: float
          :param variance: see above
          :type variance: float
       '''
     
       self.lengthscale = self.kern.lengthscale.values
       self.variance  = self.kern.variance.values     
  
       U_scaled = self.U / self.lengthscale
       Usquare = np.sum(np.square(U_scaled), 1)
       K_dist2 = -2.*gutil.linalg.tdot(U_scaled) + (Usquare[:, None] + Usquare[None, :])
        
       K_dvar = np.exp(-0.5 * K_dist2) # exp(-0.5*||zm - zn||^2 )
   
       lengthscale2 = np.square(self.lengthscale)  
       K_dist = 2*(self.U[:, None, :] - self.U[None, :, :])
       self.dK_dZ = (-self.variance / lengthscale2) * np.transpose(K_dvar[:, :, np.newaxis] * K_dist, (1, 0, 2))
     
       return self.dK_dZ
     
     
    def _gradient_U(self, mode=0):
       '''

           return dL/dU, the gradient of log-likelihood with respect to input  variables Z via chain rule
      
       '''
       # pre-compute the gradient dL/dK and dK/dU 
       try:
           self.dL_dK
       except AttributeError:
           print("The gradient of L w.r.t K not been computed. Just compute it anyway.")
           self._gradient_K() 
       else:
           if(mode == 1):
              self._gradient_K(mode)
 
       try:
           self.dK_dZ
       except AttributeError:
           print("The gradient of K w.r.t. U not been computed. Just compute it anyway")
           if self.kern.name == 'rbf':
               self._rbf_kernel_gradient_U()
       else:
           if(mode == 1):
              if self.kern.name == 'rbf':
                 self._rbf_kernel_gradient_U(mode)

       self.dL_dU = np.sum(self.dK_dZ * self.dL_dK.T[:, :, None], 0)
       return self.dL_dU


    def _gradient_U_Grassmann(self, mode=0):
       '''

          Compute DL/dU, the natural gradient of Likelihood w.r.t. U on the Grassmann manifold
          
          .. math::
             \mathbf{G} = (\mathbf{I}- \mathbf{UU}^{T})*\frac{dL}{dU}

       '''
       # pre-compute the Euclidean gradient dL/dU
       try: 
           self.dL_dU
       except AttributeError:
           print("The gradient of L w.r.t. U not been computed. Just compute it anyway.")
           self._gradient_U()
       else:
           if (mode == 1):
              self._gradient_U(mode)
       
       #projection on orthogonal direction
       U_orth = np.eye(self.nsample) - gutil.linalg.tdot(self.U)
       self.G = np.dot(U_orth, self.dL_dU)
       if self.add_regularizer:
          self.G =  self.G + self.lambda_var*np.dot((np.eye(self.nsample) - gutil.linalg.tdot(self.U_ref)), U)
       
       self.H = -self.G
       return (self.G, self.H)
        
    
    def _gradient_U_Euclidean(self, mode=0):
       '''

          Compute DL/dU, the natural gradient of Likelihood w.r.t. U on the Grassmann manifold
          
          .. math::
             \mathbf{G} = (\mathbf{I}- \mathbf{UU}^{T})*\frac{dL}{dU}

       '''
       # pre-compute the Euclidean gradient dL/dU
       try: 
           self.dL_dU
       except AttributeError:
           print("The gradient of L w.r.t. U not been computed. Just compute it anyway.")
           self._gradient_U()
       else:
           if (mode == 1):
              self._gradient_U(mode)
       
       #projection on orthogonal direction
       self.G = self.dL_dU
       if self.add_regularizer:
          self.G =  self.G + self.lambda_var*np.dot((np.eye(self.nsample) - gutil.linalg.tdot(self.U_ref)), U)
       
       self.H = -self.G
       return (self.G, self.H)



   
    def negative_loglikelihood_update(self):
       ''' 

          Compute the negative log-likelihood of Gaussian distribution
 
          .. math::
             negative-loglikelihood = \frac{p*N}{2}\log(2\pi) + \frac{p}{2}\log\det(K) + \frac{1}{2}\text{tr}(K^{-1}X*X^{T})
             
       '''
       K = self.compute_kernel_mat()
       sign , K_logdet = np.linalg.slogdet(K)
       try:
           KiX2 = self.KiX2
       except AttributeError:
           print("No KiX2 stored. Recompute")   
           KiX2 , _ = gutil.linalg.dpotrs(np.asfortranarray(K), np.asfortranarray(self.XX), lower=1)
       
         
       KiX2_trace = np.trace(KiX2)
   
       self.negative_loglikelihood = (self.ndim*self.nsample) / 2 * np.log(2*np.pi)  \
                                     + self.ndim / 2 * K_logdet  + 1/2* KiX2_trace



    def compute_kernel_mat(self):
       return self.kern.K(self.U, self.U)    
    
    
    def get_G(self):
       try: 
           self.G
       except AttributeError:
           print("The tangent direction of L w.r.t. U not been computed. Just compute it anyway.")
           self.gradient_U_Grassmann()
       return self.G

    
    def get_H(self):
       try: 
           self.H
       except AttributeError:
           print("The negative tangent direction of L w.r.t. U not been computed. Just compute it anyway.")
           self.gradient_U_Grassmann()
       return self.H

    
    def get_X(self):
       return self.X

    def get_U(self):
       return self.U
    
#    def __str__(self):
#       self.print_param()
#       self.print_solver()
    
    def print_param(self):
       print("Parameter Information: ")
       print("total samples: " + str(self.nsample))
       print("observation dimension: " + str(self.ndim))
       print("latent variable dimension: " + str(self.ndim_latent))
       print(self.kern)

    def print_solver(self):
       print("Solver settings: ")
       print("stepsize eta: " + str(self.eta))
       print("total iterations: " + str(self.iterations))
       print("tolerance threshold: " + str(self.tol))
    
       
    def get_optimize_trajectory(self):
       return (self.hist_nll, self.hist_eta, self.hist_kern)



    def restart(self):

       print("Solver restart ...")
       self.U =  self._init_U
       self.X =  self._init_X

       _ , self.ndim_latent = self.U.shape    

       self.nsample, self.ndim = self.X.shape
       
       self.XX = gutil.linalg.tdot(self.X)

       self.kern = self._kern
       
       self.negative_loglikelihood_update()
       max_iters = self._max_iters        
       self.hist_kern = list()
       self.hist_nll = np.zeros([max_iters])
       self.hist_eta = np.zeros([max_iters])
       try:
           self.G
       except AttributeError:
           print("No G.")
       else:
           self.G = np.zeros(self.G.shape)    

       try:
           self.H
       except AttributeError:
           print("No H.")
       else:
           self.H = np.zeros(self.H.shape)    


       try:
           self.KiX2
       except AttributeError:
           print("No G.")
       else:
           del self.KiX2




'''
-----------------------------------------------------------------------------------------

        Auxilary function 

-----------------------------------------------------------------------------------------
'''
    
def Grassmann_update(U, H, eta):
    '''
  
          Compute the update of U in direction of H along the geodesic of the Grassmann manifold
          .. math::
             U' = U*V*\cos(\eta*\Sigma)*V^{T} + W*\sin(\eta*\Sigma)*V^{T}
 
          :param U: the initial point of U on Grassmann manifold
          :type U: numpy.ndarray
          :param H: the tangent direction of curve on the manifold
          :type H: numpy.ndarray
          :param eta: the stepsize 
          :type eta: float

    '''
    # compute the svd of tangent vector H 
    U_sig, sig, Vh_sig =  sp.linalg.svd(H, full_matrices=False)
    ndim_latent, _ = Vh_sig.shape
    # compute the two orthogonal channels
    Sig_cos = sp.linalg.diagsvd(np.cos(eta*sig), ndim_latent, ndim_latent)
    Sig_sin = sp.linalg.diagsvd(np.sin(eta*sig), ndim_latent, ndim_latent)       
    # compute the geodesic on Grassmann manifold
    U_new = gutil.linalg.mdot(U, (Vh_sig.T, Sig_cos, Vh_sig)) + gutil.linalg.mdot(U_sig, Sig_sin, Vh_sig)
    assert (np.allclose(np.dot(U_new.T, U_new), np.eye(ndim_latent))), "Output not orthogonal"
    return U_new


def Euclidean_update(U, H, eta):
    '''

       Compute the update of U in the direction of H in Euclidean space via conventional gradient descent

       .. math::
             U' = U + eta*H
 
       :param U: the initial point of U on Grassmann manifold
       :type U: numpy.ndarray
       :param H: the tangent direction of curve on the manifold
       :type H: numpy.ndarray
       :param eta: the stepsize 
       :type eta: float

    '''
    U_new = U + eta*H
    return U_new       