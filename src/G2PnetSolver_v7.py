# coding: utf-8
 
import GPy 
import GPy.util as gutil
import numpy as np
import scipy as sp
import warnings
 
class G2PnetSolver:
    """
    ====================================================================================  
       Grassmannian Gaussian Process Network (G2P-net) solver
       We add the conjugate gradient descent algorithm for the solver and assume that the kernel is rbf + white kernel

       This is for the version 6.     
    ==================================================================================== 
    """
 
    def __init__(self, X, U, kernel, U_ref, **kwargs):
       '''

          The Grassmannian Gaussian Process Network (G2P-net) solver. 
          
          :param X: input observations, with rows as samples
          :param kernel: GPy.kern.Kern type  module. Assume the kernel is some kernel + white kernel. 
          :param U: latent variables of the observation  
          :param U_ref: the reference subspace for promity 
          :param normalize_X: True/False. if stardardize each columns. Default is False       
          :param eta: a nonnegative float, the stepsize for gradient descent and conjugate gradient descent algorithm. Default is 1e-4.
          :param lambda_var: a nonnegative float, the parameter for subspace affinity regularization. Default is 1. 
          :param max_iters: integer, maximum iterations for optimization algorithm. Default is 500.
          :param tol:= a nonnegative float, the threshold for stopping criterion. Default is 1e-3.
          :param max_iter_kernels: integer, the maximum inner iterations for update kernel parameters. Default is 100.
          :param noise_interval: = [lower, upper], the constraint region for variance of the white noise kernel. Default is [0.01, 1.]
          :param add_reg: True/False. If add the subspace affinity regularization. Default is True. 

       '''
       if len(X.shape) == 1:
           X = X.reshape(-1,1)
           warnings.warn("One dimensional observation (N,) being shaped as (N,1)")
       if len(U.shape) == 1:
           U = U.reshape(-1,1)
           warnings.warn("One dimensional input (N,) being shaped as (N,1)")

       nsample, self.ndim_latent = U.shape    
       self._ones = np.ones([nsample,])/np.sqrt(nsample)

       assert(np.allclose(np.dot(U.T, self._ones), np.zeros([self.ndim_latent,])), "U should be orthogonal to all ones") 
       self.U = U
       self._init_U = U

       assert U_ref.shape == U.shape, "The reference U should be of the same size as U_init"

       self.U_ref = U_ref
       self._init_U_ref = U_ref

 
       self.X = X
       self.nsample, self.ndim = X.shape
       assert self.nsample == nsample, "The number of samples does not match." 
       
       self.normalize_X = kwargs.get('normalize_X')
       if self.normalize_X is None:
           self.normalize_X = False 

       if self.normalize_X:
           self._Xoffset = np.mean(X,axis=1)
           self._Xscale  = np.std(X, axis=1)
           self.X = (X.copy() - self._Xoffset) / self._Xscale
       
       self._init_X = X
       self.XX = gutil.linalg.tdot(self.X)

       assert isinstance(kernel, GPy.kern.Kern), "Use the GPy.kern.Kern type module."  
       assert kernel.input_dim == self.ndim_latent , "The input dimension of U and kernel should match" 
       self.kern = kernel

       
       self.eta = kwargs.get('eta')
       if self.eta is None:
           self.eta = 1e-4

       self.lambda_var = kwargs.get('lambda_var')
       if self.lambda_var is None:
           self.lambda_var= 1
       assert self.lambda_var >= 0, "lambda_var should be nonnegative."
        
       self.tol = kwargs.get('tol')
       if self.tol is None:
           self.tol = 1e-3
       assert self.tol > 0, "tol should be positive"

       self.max_iters = kwargs.get('max_iters')
       if self.max_iters is None:
           self.max_iters = 600
       assert self.max_iters >= 0, "max_iters should be nonnegative."

       self.max_iters_kernel = kwargs.get('max_iters_kernel')
       if self.max_iters_kernel is None: 
           self.max_iters_kernel = 100

       self.noise_interval = kwargs.get('noise_interval')
       if self.noise_interval is None:
           self.noise_interval =  np.array([0.01, 1])
 
       assert len(self.noise_interval) == 2 and self.noise_interval[0] < self.noise_interval[1], "noise_interval = [lower, upper]"
       
       self.add_reg = kwargs.get('add_reg')
       if self.add_reg is None:
           self.add_reg = True
 


       self.max_iters_kernel = int(self.max_iters_kernel)
       print("kernel initialization ...")
       self.kern_param_array, self.kern_param_names, self.model_param_array, self.model_param_names,  _, self.KiX2 = self.kernel_update(X=self.X, U=self.U, noise_interval = self.noise_interval, kernel=self.kern)

       self.kern.param_array[:] = self.kern_param_array
       self.kern.parameters_changed()
      
       self._kern = self.kern.copy()
       self.Gaussian_noise_var = self.model_param_array[-1]

       print("negative log-likelihood compute")
       self.negative_loglikelihood_update()
              
       self.eta = float(self.eta)
       self.max_iters = int(self.max_iters)
       self._max_iters = self.max_iters
       self.tol = float(self.tol)
       self.cond_dK = 0 
       self.cond_K = 0

       self.hist_nll = np.zeros([self.max_iters])
       self.hist_eta = np.zeros([self.max_iters])
          
    
    

    def optimize(self, max_iters = 600, tol=1e-3, init_mode = "fixed", optimizor="gd_grass", verbose=False):
       '''
     
        The main function of optimization procedure. Calls optimize_gradient_descent if the gradient descent is required 
        Input:
           :param max_iters: integer, maximum iterations for optimization algorithm. Default is 600.
           :param tol: a nonnegative float, the threshold for stopping criterion. Default is 1e-3.
           :param verbose: if message reporting during the update of kernel parameters. Default is False.
           :param init_mode: "fixed" if the initial point is U_init, "random" if random initial point is chosen.
           :param optimizor: The type of optimizor 
                   "gd_grass": using the Grassmannian Gradient Descent algorithm. The default setting.
                   "gd": using the Euclidean Gradient Descent. The convention setting.
                   "cg_grass": using the Conjugate Gradient Descent on Grassmannian. 

        Return:
           (self.hist_nll, self.hist_eta, self.hist_kern, self.hist_Hsig)

           :param hist_nll: The trajectory of negative log-likelihood functions vs. iterations
           :param hist_eta: The trajectory of stepsize eta vs. iterations
           :param hist_kern: The trajectory of kernels vs. iterations
           :param hist_Hsig: numpy.ndarray of size [max_iters, ndim_latent] each row for all singular value of H at given iteration. 
 
       '''


       if optimizor == "gd_grass":
           return self.optimize_gradient_descent_grass(max_iters, tol, verbose)
       elif optimizor == "gd":
           return self.optimize_gradient_descent_Euclidean(max_iters, tol, verbose)  
       elif optimizor == "cg_grass":
           return self.optimize_conjugate_grad_grass(max_iters, tol, verbose)


 
    def optimize_gradient_descent_grass(self, max_iters=10, tol=1e-3, init_mode = "fixed", verbose=False):
       '''

         Optimization procedure using gradient descent on manifold.
         See optimize() method.

         Auxilary params:
             :param hist_GPmodel: records of models. Used for kernel update.
             :param hist_G_norm:  records for Frobenius norm of gradient matrix G
             :param hist_cond_dK: records of condition number of T = dL/dK 
             :param hist_cond_K:  records of condition number of K
       '''
       self.max_iters = max_iters
       self.tol = tol
       self.hist_nll = np.zeros([max_iters])
       self.hist_eta = np.zeros([max_iters])
       self.hist_kern = []
       self.hist_GPmodel = []      
       self.hist_Hsig = np.zeros([max_iters, self.ndim_latent])
       self.hist_G_norm = np.zeros([max_iters])
       self.hist_cond_dK = np.zeros([max_iters])
       self.hist_cond_K = np.zeros([max_iters])

       messages = verbose
       eta = self.eta*np.ones([max_iters])
       self.hist_eta = eta
       for i, eta_i in enumerate(eta):
           if i == 0:
              noise_var = 1.
              self.hist_kern = self.kern_param_array
              self.hist_GPmodel = self.model_param_array
           else: 
              noise_var = self.Gaussian_noise_var
              self.hist_kern = np.vstack((self.hist_kern, self.kern.param_array))
              self.hist_GPmodel = np.vstack((self.hist_GPmodel, self.model_param_array))

           self.hist_nll[i] = self.negative_loglikelihood
           # compute the grassmannian gradient
           self._gradient_U_Grassmann(mode=1)
           self.hist_cond_dK[i] = self.cond_dK
           self.hist_cond_K[i] = self.cond_K
           self.hist_G_norm[i] = self.compute_norm_G()
           self.hist_Hsig[i,:] =  sp.linalg.svdvals(self.H).T
#           while not check_step(self.H, eta_i):
#              print("stepsize decrease {0:6.5f}".format(eta_i*0.9))
#              eta_i = eta_i*0.9
           print("[step {0:3d}] neg-log-likelihood {1:9f} stepsize {2:6.4f}".format(i, self.negative_loglikelihood, eta_i))
           self.hist_eta[i] = eta_i
           if check_step(self.H, eta_i):
              # update the subspace through geodesic on Grassmann manifold     
              U_temp = Grassmann_update(self.U, self.H, eta_i)
              self.U = U_temp
              # update the other kernel function
              # self.kern, _ , _ = self.kernel_update(X=self.X, U=self.U, kernel=self.kern, noise_var = self.Gaussian_noise_var  ,messages= messages)

              self.kern_param_array, _ , self.model_param_array, _ ,  _, _ = self.kernel_update(X=self.X, U=self.U, noise_var = noise_var, kernel=self.kern, noise_interval = self.noise_interval, messages = messages)

              self.kern.param_array[:] = self.kern_param_array
              self.kern.parameters_changed()
      
      
              self.Gaussian_noise_var = self.model_param_array[-1]
              self.negative_loglikelihood_update()


       return (self.hist_nll, self.hist_eta, self.hist_kern, self.hist_Hsig)



    def optimize_conjugate_grad_grass(self, max_iters=10, tol=1e-3, init_mode = "fixed", verbose=False):
       '''

         Optimization procedure using conjugate gradient descent on manifold.
         See optimize() method.

         Auxilary params:
             :param hist_GPmodel: records of models. Used for kernel update.
             :param hist_G_norm:  records for Frobenius norm of gradient matrix G
             :param hist_cond_dK: records of condition number of T = dL/dK 
             :param hist_cond_K:  records of condition number of K


       '''
       self.max_iters = max_iters
       self.tol = tol
       self.hist_nll = np.zeros([max_iters])
       self.hist_eta = np.zeros([max_iters])
       self.hist_kern = []
       self.hist_GPmodel = []
       self.U_pre = self.U 
       self.G_pre = np.zeros([self.nsample, self.ndim_latent])      
       self.H_pre = np.zeros(self.G_pre.shape)
       self.hist_Hsig = np.zeros([max_iters, self.ndim_latent])
       self.hist_Hsig_pre = np.zeros([max_iters, self.ndim_latent])
       self.hist_G_norm = np.zeros([max_iters])
       self.hist_gamma  = np.zeros([max_iters])
       self.hist_delta_G = np.zeros([max_iters])
       self.hist_cond_dK = np.zeros([max_iters])
       self.hist_cond_K = np.zeros([max_iters])

       messages = verbose
       eta = self.eta*np.ones([max_iters])
       self.hist_eta = eta
       for i, eta_i in enumerate(eta):
           if i == 0:
              noise_var = 1.
              self.hist_kern = self.kern_param_array
              self.hist_GPmodel = self.model_param_array
           else: 
              noise_var = self.Gaussian_noise_var
              self.hist_kern = np.vstack((self.hist_kern, self.kern.param_array))
              self.hist_GPmodel = np.vstack((self.hist_GPmodel, self.model_param_array))

           self.hist_nll[i] = self.negative_loglikelihood
           # compute the grassmannian gradient
           self._gradient_U_Grassmann(mode=1)

           self.hist_G_norm[i] = self.compute_norm_G()
           self.hist_cond_dK[i] = self.cond_dK
           self.hist_cond_K[i] = self.cond_K
           #eta_temp = eta_i
           # update the conjugate gradient
           if i > 0:
              U0 = self.U_pre
              G0 = self.G_pre
              H0 = self.H_pre
              eta0 = self.hist_eta[i-1]
              if i%(self.ndim_latent*(self.nsample- self.ndim_latent)) != 0: 
                 self.H, self.gamma_conjugate, self.delta_G, self.tH, self.tG = Conjugate_grad_compute(U0, self.G, G0, H0, eta0)
                 
              self.hist_gamma[i] = self.gamma_conjugate
              self.hist_delta_G[i] = self.delta_G

           self.hist_Hsig_pre[i,:] =  sp.linalg.svdvals(self.H).T
           if i == 0:
              self.G_pre = self.G
              self.H_pre = self.H

           self.hist_Hsig[i,:] =  sp.linalg.svdvals(self.H).T
#           while not check_step(self.H, eta_i): #eta_temp):
#              #print("stepsize decrease {0:6.5f}".format(eta_temp*0.9))
#              print("stepsize decrease {0:6.5f}".format(eta_i*0.9))
#              #eta_temp = eta_temp*0.9
#              eta_i = eta_i*0.9
           #self.hist_eta[i] = eta_i #eta_temp

           print("[step {0:3d}] neg-log-likelihood {1:9f} stepsize {2:6.4f}".format(i, self.negative_loglikelihood, eta_i))
           if True: #check_step(self.H, eta_i): 
              self.G_pre = self.G
              self.H_pre = self.H
              if i > 0:
                 self.U_pre = self.U
          
              # update the subspace through geodesic on Grassmann manifold     
              U_temp = Grassmann_update(self.U, self.H, eta_i)
              self.U = U_temp
              # update the other kernel function
              self.kern_param_array, _ , self.model_param_array, _ ,  _, _ = self.kernel_update(X=self.X, U=self.U, noise_var = noise_var, kernel=self.kern, noise_interval = self.noise_interval, messages = messages)

              self.kern.param_array[:] = self.kern_param_array
              self.kern.parameters_changed()
      
      
              self.Gaussian_noise_var = self.model_param_array[-1]
              self.negative_loglikelihood_update()


       return (self.hist_nll, self.hist_eta, self.hist_kern, self.hist_Hsig)




    def optimize_gradient_descent_Euclidean(self, max_iters=10, tol=1e-3, init_mode = "fixed", verbose=False):
       '''

         Optimization procedure using gradient descent on Eulidean space.
         See optimize() method.

         Auxilary params:
             :param hist_GPmodel: records of models. Used for kernel update.
             :param hist_G_norm:  records for Frobenius norm of gradient matrix G
             :param hist_cond_dK: records of condition number of T = dL/dK 
             :param hist_cond_K:  records of condition number of K


       '''
       self.max_iters = max_iters
       self.tol = tol
       self.hist_kern = []
       self.hist_GPmodel = []      
       self.hist_nll = np.zeros([max_iters])
       self.hist_eta = np.zeros([max_iters]) 
       self.hist_Hsig = np.zeros([max_iters, self.ndim_latent])
       self.hist_cond_dK = np.zeros([max_iters])
       self.hist_cond_K = np.zeros([max_iters])
       messages = verbose
     
       eta = self.eta*np.ones([max_iters])
       self.hist_eta = eta
       for i, eta_i in enumerate(eta):
           if i == 0:
              noise_var = 1.
              self.hist_kern = self.kern_param_array
              self.hist_GPmodel = self.model_param_array
           else: 
              noise_var = self.Gaussian_noise_var
              self.hist_kern = np.vstack((self.hist_kern, self.kern.param_array))
              self.hist_GPmodel = np.vstack((self.hist_GPmodel, self.model_param_array))
           
           self.hist_nll[i] = self.negative_loglikelihood
           print("[step {0:3d}] neg-log-likelihood {1:9f} stepsize {2:6.4f}".format(i, self.negative_loglikelihood, eta_i))
           # compute the grassmannian gradient
           #self._gradient_U_Grassmann(mode=1)
           self._gradient_U_Euclidean(mode=1)
           self.hist_cond_dK[i] = self.cond_dK
           self.hist_cond_K[i] = self.cond_K
           # update the matrix U via gradient descent 
           self.hist_Hsig[i,:] =  sp.linalg.svdvals(self.H).T
           U_temp = Euclidean_update(self.U, self.H, eta_i)
           self.U = U_temp
           # update the other kernel function
           self.kern_param_array, _ , self.model_param_array, _ ,  _, _ = self.kernel_update(X=self.X, U=self.U, noise_var = noise_var, kernel=self.kern, noise_interval = self.noise_interval, messages = messages)

           self.kern.param_array[:] = self.kern_param_array
           self.kern.parameters_changed()
      
           self.Gaussian_noise_var = self.model_param_array[-1]
           self.negative_loglikelihood_update()
       
       return (self.hist_nll, self.hist_eta, self.hist_kern, self.hist_Hsig)


    def kernel_update(self, U, X, kernel, noise_interval,  noise_var = 1. ,  messages = True):
       '''

           Given the subspace U, and output X, update the kernel parameters
           Input:
             

           Return:
              (kern_param_array, kern_param_names,  model_param_array, model_param_names,  XX, KiX2)

              :param kern_param_array: the list of values of all kernel parameters
              :param kern_param_names: the list of names of corresponding kernel parameters
              :param model_param_array: the list of values of kernel parameters and Gaussian noise variance
              :param model_param_names: the corresponding names for model parameters
              :param XX:  the gram matrix  X*X.T
              :param kiX2: the matrix K^{-1}*X*X.T
       '''
       nsample , ndim = X.shape 
       nsample2 , ndim_latent = U.shape
        
       assert nsample == nsample2, "The number of rows of X and U should match"
       assert isinstance(kernel, GPy.kern.Kern), "Use the GPy.kern.Kern type module"  
       assert kernel.input_dim == self.ndim_latent , "The input dimension of U and kernel should match" 

       assert len(noise_interval) == 2 and noise_interval[0] < noise_interval[1], "noise_interval = [lower, upper]"
       model = GPy.models.GPRegression(X= U, Y = X, kernel = kernel, noise_var = noise_var)
       try: 
           model.kern.white
       except AttributeError: 
           pass
       else:
           model.kern.white.constrain_bounded(noise_interval[0], noise_interval[1], warning=False ) 
       model.optimize(max_iters = self.max_iters_kernel, messages=messages)
       kern_param_array = model.kern.param_array
       kern_param_names = model.kern.parameter_names()
       model_param_names = model.parameter_names()
       model_param_array = model.param_array

 
       Kernel_mat = model.kern.K(U)             
       XX = gutil.linalg.tdot(X)

       KiX2 , _ = gutil.linalg.dpotrs(np.asfortranarray(Kernel_mat), np.asfortranarray(XX), lower=1)
       return (kern_param_array, kern_param_names,  model_param_array, model_param_names,  XX, KiX2)




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
       Kernel = self.compute_kernel_mat()
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
       self.cond_dK = np.linalg.cond(self.dL_dK)  
       return (self.dL_dK, self.cond_dK) 
     
     

     
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
     
       self.lengthscale = self.kern.rbf.lengthscale.values
       self.variance  = self.kern.rbf.variance.values     
       self.white_variance = self.kern.white.variance.values

       U_scaled = self.U / self.lengthscale
       Usquare = np.sum(np.square(U_scaled), 1)
       K_dist2 = -2.*gutil.linalg.tdot(U_scaled) + (Usquare[:, None] + Usquare[None, :])
        
       K_dvar = np.exp(-0.5 * K_dist2) # exp(-0.5*||zm - zn||^2 )
   
       lengthscale2 = np.square(self.lengthscale)  
       #K_dist[i,j,k] = 2*[U_{i,k} - U_{j,k}]
       K_dist = 2*(self.U[:, None, :] - self.U[None, :, :])
       #dK_dZ[i,j,k] = exp(-|| U_{i,:} - U_{j,:} ||^2) * K_dist[i,j,k] 
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
           elif self.kern.name == 'sum':
              try:
                 self.kern.rbf
              except AttributeError: 
                 pass
              else:
                 self._rbf_kernel_gradient_U()
       else:
           if(mode == 1):
              if self.kern.name == 'rbf':
                 self._rbf_kernel_gradient_U(mode)
              elif self.kern.name == 'sum':
                 try:
                    self.kern.rbf
                 except AttributeError: 
                    pass
                 else:
                    self._rbf_kernel_gradient_U()

       self.cond_dK = np.linalg.cond(self.dL_dK) 
       Kernel = self.kern.K(self.U)
       self.cond_K = np.linalg.cond(Kernel)
       #dK_dZ[i,j,k] = transpose(exp(-|| U_{i,:} - U_{j,:} ||^2) * K_dist[i,j,k])
       #dL_dK[i,j,0] = p*inv(K)[i,j] - inv(K)[i,:]*X*X.T*inv(K)[:,j]
       #(dK_dZ* dL_dK)[i,j,k] = K_dist[i,j,k]*exp(-|| U_{i,:} - U_{j,:} ||^2) * dL_dK[i,j]
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

       dL_dU = self.dL_dU
       if self.add_reg:
          dL_dU =  dL_dU + self.lambda_var*np.dot((np.eye(self.nsample) - gutil.linalg.tdot(self.U_ref)), self.U)
       #projection on orthogonal direction
       #ones = np.ones([self.nsample,])/np.sqrt(self.nsample)
       U_orth = np.eye(self.nsample) - gutil.linalg.tdot(np.column_stack((self.U, self._ones))) 
       #U_orth = np.eye(self.nsample) - gutil.linalg.tdot(self.U)
       self.G = np.dot(U_orth, dL_dU)
       
       self.H = -self.G
       return (self.G , self.H)
        
    
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
       if self.add_reg:
          self.G =  self.G + self.lambda_var*np.dot((np.eye(self.nsample) - gutil.linalg.tdot(self.U_ref)), self.U)
       
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



    def compute_norm_G(self):
       G_norm = np.linalg.norm(self.G, 'fro')
       return G_norm

    def compute_kernel_mat(self):
       return self.kern.K(self.U)    
    
    
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
    
def check_step(H, eta):
    '''
         Make sure the not all eigenvalues of H times eta is above pi/2 
    
    '''
    sigval = sp.linalg.svdvals(H) 
    #the last is the smallest
    k = int((eta*sigval[0])/(2*np.pi))
    res = eta*sigval[0] - 2*k*np.pi
    return  (res <  np.pi/2)
            #(eta*sigval[-1] <  np.pi/2)

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
    nsample, k = U.shape
    U_sig, sig, Vh_sig =  sp.linalg.svd(H, full_matrices=False)
    ndim_latent, _ = Vh_sig.shape
    # compute the two orthogonal channels
    Sig_cos = sp.linalg.diagsvd(np.cos(eta*sig), ndim_latent, ndim_latent)
    Sig_sin = sp.linalg.diagsvd(np.sin(eta*sig), ndim_latent, ndim_latent)       
    # compute the geodesic on Grassmann manifold
    U_new = gutil.linalg.mdot(U, (Vh_sig.T, Sig_cos, Vh_sig)) + gutil.linalg.mdot(U_sig, Sig_sin, Vh_sig)
    assert (np.allclose(np.dot(U_new.T, U_new), np.eye(ndim_latent))), "Output not orthogonal"
    assert (np.allclose(np.dot(U_new.T, np.ones([nsample,])), np.zeros([k,]))), "Output not orthogonal to all ones"
    return U_new


def Conjugate_grad_compute(U0, G, G0, H0, eta):
    ''' 
       Compute the conjugate gradient direction

       H_{k} = -G_{k} + \gamma*tH_{k-1}, where \gamma = trace(G_{k}.T*(G_{k} - tG_{k-1}))/trace(G_{k-1}.T, G_{k-1})

    '''
    tH0, tG0 = parallel_transport(U0, G0, H0,  eta)
    diff_G0 = G- tG0
    diff_G0_1 = G - G0
    norm_G0 = np.linalg.norm(G0)
    
    delta_G = np.inner(G.flatten('F'), diff_G0.flatten('F'))
    if abs(delta_G) < 1e-3 or norm_G0 < 1e-5:
        gamma = 0
    #elif norm_G0 < 1e-3:
    #    gamma = 0
    else:
        log_gamma = np.log(abs(delta_G)) - np.log(np.square(norm_G0)+1e-5)
        gamma = np.sign(delta_G)*np.exp(log_gamma) 
       
    #if abs(gamma) > 1e4:
    #    gamma = 0 
 
    H = -G + gamma*tH0
#    eta_temp = eta
#    while not check_step(H, eta_temp):
#       print("stepsize decrease {0:6.5f}".format(eta_temp*0.9))
#       eta_temp = eta_temp*0.9
#       tH0, tG0 = parallel_transport(U0, G0, H0,  eta_temp)
#       diff_G0 = G- tG0
#       diff_G0_1 = G - G0
#       norm_G0 = np.linalg.norm(G0)
#       delta_G = np.inner(G.flatten('F'), diff_G0.flatten('F'))
#       gamma = delta_G / np.square(norm_G0) 
#       #gamma = np.trace(np.dot(G.T, diff_G0_1)) / np.square(norm_G0)
#       #gamma = np.trace(np.dot(G.T, diff_G0)) / np.square(norm_G0)
#       H = -G + gamma*tH0
       
    return (H, gamma, delta_G, tH0, tG0)


def parallel_transport(U, G, H,  eta):
    '''
          
       Compute the parallel transport on Grassmann manifold for conjugate gradient descent
          
       .. math::
          tH = -U*V*\sin(\eta*\Sigma)*\Sigma*V^{T} + W*\cos(\eta*\Sigma)*\Sigma*V^{T} 
          tG = G - U*V*\sin(\eta*\Sigma)*W^{T}*G - W*(I - \cos(\eta*\Sigma))*W^T*G
    '''
    # compute the svd of tangent vector H 
    U_sig, sig, Vh_sig =  sp.linalg.svd(H, full_matrices=False)
    ndim_latent, _ = Vh_sig.shape
    # compute the two orthogonal channels
    Sig_cos_Sig = sp.linalg.diagsvd(np.cos(eta*sig)*sig, ndim_latent, ndim_latent)
    Sig_sin_Sig = sp.linalg.diagsvd(np.sin(eta*sig)*sig, ndim_latent, ndim_latent)       
    I_Sig_cos = sp.linalg.diagsvd(1-np.cos(eta*sig), ndim_latent, ndim_latent)
    Sig_sin = sp.linalg.diagsvd(np.sin(eta*sig), ndim_latent, ndim_latent)       
    # compute the parallel transport
    tH = -gutil.linalg.mdot(U, (Vh_sig.T, Sig_sin_Sig, Vh_sig)) + gutil.linalg.mdot(U_sig, Sig_cos_Sig, Vh_sig)
    
    tG = G - gutil.linalg.mdot(U, (Vh_sig.T, Sig_sin, U_sig.T), G) - gutil.linalg.mdot((U_sig, I_Sig_cos, U_sig.T), G)

    return (tH, tG)

    

    
 

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