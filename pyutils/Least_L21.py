## FUNCTION Least_TGL
# L21 Joint Feature Learning with Least Squares Loss.
#
## OBJECTIVE
# argmin_W { sum_i^t (0.5 * norm (Y[i] - X[i]' * W(:, i))^2)
#            + opts.rho_L2 * \|W\|_2^2 + rho1 * \|W\|_{2,1} }
#
## INPUT
# X: {n * d} * t - input matrix
# Y: {n * 1} * t - output matrix
# rho1: L2,1-norm group Lasso parameter.
# optional:
#   opts.rho_L2: L2-norm parameter (default = 0).
#
## OUTPUT
# W: model: d * t
# funcVal: function value vector.
#
## LICENSE
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye
#
#   You are suggested to first read the Manual.
#   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
#
#   Last modified on June 3, 2012.
#
## RELATED PAPERS
#
#   [1] Evgeniou, A. and Pontil, M. Multi-task feature learning, NIPS 2007.
#   [2] Liu, J. and Ye, J. Efficient L1/Lq Norm Regularization, Technical
#       Report, 2010.
#
## RELATED FUNCTIONS
#  Least_L21, init_opts

## Code starts here

from .init_opts import *
from numpy.linalg import norm
from numpy.matlib import repmat


def Least_L21(X, Y, rho1, opts=[]):

    # X = multi_transpose(X)
    X = [x.T for x in X]

    # initialize options.
    opts=init_opts(opts)

    if isfield(opts, 'rho_L2'):
        rho_L2 = opts.rho_L2
    else:
        rho_L2 = 0

    task_num  = len(X)
    dimension = X[0].shape[0]

    funcVal = []

    # initialize a starting point
    if isfield(opts,'W0'):
        W0=opts.W0
        if (np.count_nonzero(W0.shape-np.array([dimension, task_num]))):
            raise ValueError('Check the input .W0')
    elif opts.init==2:
        W0 = np.zeros(dimension, task_num)
    elif opts.init == 0:
        XY = []
        W0_prep = []
        for t_idx in range(task_num):
            XY.append(X[t_idx] @ Y[t_idx])
            # W0_prep = cat(2, W0_prep, XY[t_idx])

        W0 = np.vstack(XY).T


    bFlag=0 # this flag tests whether the gradient step only changes a little

    Wz= W0
    Wz_old = W0

    t = 1
    t_old = 0

    iter = 0
    gamma = 1
    gamma_inc = 2


    ## private functions

    # smooth part gradient.
    def gradVal_eval (W):
        # if opts.pFlag
        #     grad_W = zeros(zeros(W))
        #     parfor i = 1:task_num
        #         grad_W (i, :) = X[i]*(X[i]' * W(:,i)-Y[i])
        # else
        grad_W = [X[i] @ (X[i].T @ W[:,i]-Y[i]) for i in range(task_num)]
        grad_W = np.vstack(grad_W).T
        grad_W += rho_L2 * 2 * W
        return grad_W

    # smooth part function value.
    def funVal_eval (W):
        # funcVal = 0
        # if opts.pFlag
        #     parfor i = 1: task_num
        #         funcVal = funcVal + 0.5 * norm (Y[i] - X[i]' * W(:, i))^2
        # else:
        funcVal = np.sum([0.5 * norm (Y[i] - X[i].T @ W[:, i])**2 for i in range(task_num)])
        funcVal += rho_L2 * norm(W,'fro')**2
        return funcVal

    def FGLasso_projection (D, Lambda):
        # l2.1 norm projection.
        p = np.fmax(np.zeros(D.shape[0]), 1 - Lambda/np.sqrt(np.sum(D**2,axis=-1)))
        X = repmat(p.reshape(-1,1), 1, D.shape[-1]) * D
        return X

    def nonsmooth_eval(W, rho_1):
        # non_smooth_value = 0
#         if opts.pFlag
#             parfor i = 1 : size(W, 1)
#                 w = W(i, :)
#                 non_smooth_value = non_smooth_value ...
#                     + rho_1 * norm(w, 2)
#         else
        non_smooth_value = np.sum([rho_1 * norm(W[i, :], 2) for i in range(W.shape[0])])
        return non_smooth_value
            

    while iter < opts.maxIter:
        alpha = (t_old - 1) /t
        
        Ws = (1 + alpha) * Wz - alpha * Wz_old
        
        # compute function value and gradients of the search point
        gWs  = gradVal_eval(Ws)
        Fs   = funVal_eval (Ws)
    
        while True:
            Wzp = FGLasso_projection(Ws - gWs/gamma, rho1 / gamma)
            Fzp = funVal_eval(Wzp)
            
            delta_Wzp = Wzp - Ws
            r_sum = norm(delta_Wzp, 'fro')**2
            #  Fzp_gamma = Fs + trace(delta_Wzp' * gWs) + gamma/2 * norm(delta_Wzp, 'fro')^2
            Fzp_gamma = Fs + np.sum(np.sum(delta_Wzp * gWs, axis=0), axis=0) + gamma/2 * norm(delta_Wzp, 'fro')**2
           
            if (r_sum <=1e-20):
                bFlag=1 # this shows that, the gradient step makes little improvement
                break
            
            if (Fzp <= Fzp_gamma):
                break
            else:
                gamma *= gamma_inc
            
        Wz_old = Wz
        Wz = Wzp
        
        funcVal.append(Fzp + nonsmooth_eval(Wz, rho1))
        
        if (bFlag):
            print('\n The program terminates as the gradient step changes the solution very small.')
            break
        
        
        # test stop condition.
        if opts.tFlag == 0:
            if iter>=2:
                if (abs( funcVal[-1] - funcVal[-2] ) <= opts.tol):
                    break
        elif opts.tFlag == 1:
            if iter>=2:
                if (abs( funcVal[-1] - funcVal[-2] ) <= opts.tol* funcVal[-2]):
                    break
        elif opts.tFlag == 2:
            if ( funcVal[-1] <= opts.tol):
                break
        elif opts.tFlag == 3:
            if iter>=opts.maxIter:
                break
        else:
            raise ValueError
            
        iter += 1
        t_old = t
        t = 0.5 * (1 + (1+ 4 * t**2)**0.5)

    W = Wzp

    return W, np.array(funcVal)
