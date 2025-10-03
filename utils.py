# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:41:58 2025

@author: aclouthier

Functions related to statistical shape modelling

"""

import numpy as np

def procrustes(X,Y,scale=True):
    '''
    Procrustes Analysis
    * adapted from Matlab procrustes.m

    Determines a linear transformation (translation,
    reflection, orthogonal rotation, and scaling) of the points in the
    matrix Y to best conform them to the points in the matrix X.  The
    "goodness-of-fit" criterion is the sum of squared errors.  PROCRUSTES
    returns the minimized value of this dissimilarity measure in D.  D is
    standardized by a measure of the scale of X, given by

       sum(sum((X - repmat(mean(X,1), size(X,1), 1)).^2, 1))

    i.e., the sum of squared elements of a centered version of X.  However,
    if X comprises repetitions of the same point, the sum of squared errors
    is not standardized.

    X and Y are assumed to have the same number of points (rows), and
    PROCRUSTES matches the i'th point in Y to the i'th point in X.  Points
    in Y can have smaller dimension (number of columns) than those in X.
    In this case, PROCRUSTES adds columns of zeros to Y as necessary.

    Z = b * Y * T + c.

   References:
     [1] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.
     [2] Gower, J.C. and Dijskterhuis, G.B., Procrustes Problems, Oxford
         Statistical Science Series, Vol 30. Oxford University Press, 2004.
     [3] Bulfinch, T., The Age of Fable; or, Stories of Gods and Heroes,
         Sanborn, Carter, and Bazin, Boston, 1855.

    Parameters
    ----------
    X : numpy.ndarray
        n x 3 matrix of point coordinates for target/reference mesh.
    Y : numpy.ndarray
        n x 3 matrix of point coordinates for the mesh to be transformed.
    scale : bool, optional
        If True, compute a procrustes solution that includes a scale component.
        The default is True.

    Returns
    -------
    Z : numpy.ndarray
        The transformed mesh, now aligned with X.
    T : numpy.ndarray
        The orthogonal rotation and reflection component of the transformation
        that maps Y to Z.
    b : float
        The scale component of the transformation that maps Y to Z.
    c : numpu.ndarray
        The translation component of the transformation that maps Y to Z.
    d : float
        the standardized distance.

    '''


    X0 = X - np.tile(X.mean(axis=0),(X.shape[0],1))
    Y0 = Y - np.tile(Y.mean(axis=0),(Y.shape[0],1))

    ssqX = np.square(X0).sum(axis=0)
    ssqY = np.square(Y0).sum(axis=0)
    constX = (ssqX <= np.square(np.abs(np.spacing(1)*X.shape[0]*X.mean(axis=0)))).any()
    constY = (ssqY <= np.square(np.abs(np.spacing(1)*X.shape[0]*Y.mean(axis=0)))).any()
    ssqX = ssqX.sum()
    ssqY = ssqY.sum()

    if (not constX) and (not constY):
        # The "centred" Frobenius norm
        normX = np.sqrt(ssqX) # == sqrt(trace(X0*X0'))
        normY = np.sqrt(ssqY)

        # Scale to equal (unit) norm
        X0 = X0 / normX
        Y0 = Y0 / normY

        # Make sure they're in the same dimension space
        if Y.shape[1] < X.shape[1]:
            Y0 = np.concatenate((Y0,np.zeros(Y.shape[0],X.shape[1]-Y.shape[1])))

        # The optimum rotation matrix of Y
        A = np.matmul(X0.transpose(),Y0)
        [L,D,M] = np.linalg.svd(A)
        T = np.matmul(M.transpose(),L.transpose())

        # can include code to force reflection or no here

        # The minimized unstandardized distance D(X0,b*Y0*T) is
        # ||X0||^2 + b^2*||Y0||^2 - 2*b*trace(T*X0'*Y0)
        traceTA = D.sum()

        if scale == True:
            b = traceTA * normX /normY # the optimum scaling of Y
            d = 1 - traceTA**2 # the standardized distance between X and b*Y*T+c
            Z = normX * traceTA * np.matmul(Y0,T) + np.tile(X.mean(axis=0),(X.shape[0],1))
        else:
            b = 1
            d = 1 + ssqY/ssqX - 2*traceTA*normY/normX # The standardized distance between X and Y*T+c.
            Z = normY * np.matmul(Y0,T) + np.tile(X.mean(axis=0),(X.shape[0],1))

        c = X.mean(axis=0) - b * np.matmul(Y.mean(axis=0),T)

    # The degenerate cases: X all the same, and Y all the same.
    elif constX:
        d = 0
        Z = np.tile(X.mean(axis=0),(X.shape[0],1))
        T = np.eye(Y.shape[1],X.shape[1])
        b = 0
        c = Z
    else: # constX and constY
        d = 1
        Z = np.tile(X.mean(axis=0),(X.shape[0],1))
        T = np.eye(Y.shape[1],X.shape[1])
        b = 0
        c = Z

    return Z, T, b, c, d


def gr(x,y):
    '''
    For Givens plane rotation
    
    Adapted from /Matlab_tools/KneeACS/Tools/gr.m
    Created by I M Smith 08 Mar 2002

    Parameters
    ----------
    x : float
        DESCRIPTION.
    y : float
        DESCRIPTION.

    Returns
    -------
    U : numpy.array
        2x2 rotation matrix [c s; -s c], with U * [x y]' = [z 0]'
    c : float
        cosine of the rotation angle
    s : float
        sine of the rotation angle

    '''
    if y == 0:
        c = 1
        s = 0
    elif np.abs(y) > np.abs(x):
        t = x/y
        s = 1/np.sqrt(1+t*t)
        c = t*s
    else:
        t = y/x
        c = 1/np.sqrt(1+t*t)
        s=t*c
    U = np.array([[c,s],[-s,c]])
    
    return U, c, s

def fgrrot3(theta,R0=np.eye(3)):
    '''
    Form rotation matrix R = R3*R2*R1*R0 and its derivatives using right-
    handed rotation matrices.
             R1 = [ 1  0   0 ]  R2 = [ c2 0  s2 ] and R3 = [ c3 -s3 0 ]
                  [ 0 c1 -s1 ],      [ 0  1   0 ]          [ s3  c3 0 ].
                  [ 0 s1  c2 ]       [-s2 0  c2 ]          [  0   0 1 ]
                  
    Adapted from Matlab_Tools/KneeACS/Tools/fgrrot3.m
    by I M Smith 27 May 2002

    Parameters
    ----------
    theta : numpy.array
        Array of plane rotation angles (t1,t2,t3).
    R0 : numpy.array
        3x3 rotation matrix, optional with default = I.

    Returns
    -------
    R : numpy.array
        3x3 rotation matrix.
    DR1 : numpy.array
        Derivative of R wrt t1.
    DR2 : numpy.array
        Derivative of R wrt t2.
    DR3 : numpy.array
        Derivative of R wrt t3.

    '''
    ct = np.cos(theta)
    st = np.sin(theta)
    R1 = np.array([[1,0,0],[0,ct[0],-st[0]],[0,st[0],ct[0]]])
    R2 = np.array([[ct[1],0,st[1]],[0,1,0],[-st[1],0,ct[1]]])
    R3 = np.array([[ct[2],-st[2],0],[st[2],ct[2],0],[0,0,1]])
    R = np.matmul(R3,np.matmul(R2,R1))
    # evaluate derivative matrices
    # drrot3  function
    dR1 = np.array([[0,0,0],[0,-R1[2,1],-R1[1,1]],[0,R1[1,1],-R1[2,1]]])
    dR2 = np.array([[-R2[0,2],0,R2[0,0]],[0,0,0],[-R2[0,0],0,-R2[0,2]]])
    dR3 = np.array([[-R3[1,0],-R3[0,0],0],[R3[0,0],-R3[2,0],0],[0,0,0]])
    DR1 = np.matmul(R3,np.matmul(R2,np.matmul(dR1,R0)))
    DR2 = np.matmul(R3,np.matmul(dR2,np.matmul(R1,R0)))
    DR3 = np.matmul(dR3,np.matmul(R2,np.matmul(R1,R0)))
    
    return R, DR1, DR2, DR3

def fgcylinder(a,X,w):
    '''
    Function and gradient calculation for least-squares cylinder fit.
    
    Adapted from Matlab_Tools/KneeACS/Tools/fgcylinder.m
    by I M Smith 27 May 2002

    Parameters
    ----------
    a : numpy.array
        Parameters [x0 y0 alpha beta s].
    X : numpy.array
        Array [x y z] where x = vector of x-coordinates, 
        y = vector of y-coordinates and z = vector of z-coordinates. .
    w : numpy.array
        Weights.

    Returns
    -------
    f : numpy.array
        Signed distances of points to cylinder:
         f(i) = sqrt(xh(i)^2 + yh(i)^2) - s, where 
         [xh yh zh]' = Ry(beta) * Rx(alpha) * ([x y z]' - [x0 y0 0]').
         Dimension: m x 1.
    J : numpy.array
        Jacobian matrix df(i)/da(j). Dimension: m x 5.

    '''
    m = X.shape[0]
    # if no weights are specified, use unit weights
    # if w == None:
    #     w = np.ones((m,1))
    
    x0 = a[0]
    y0 = a[1]
    alpha = a[2]
    beta = a[3]
    s = a[4]
    
    R, DR1, DR2, _ = fgrrot3(np.array([alpha,beta,0]))
    
    Xt = np.matmul(X - np.array([x0,y0,0]),R.T)
    rt = np.linalg.norm(Xt[:,:2],axis=1)
    Nt = np.zeros((m,3))
    Nt[:,0] = np.divide(Xt[:,0],rt)
    Nt[:,1] = np.divide(Xt[:,1],rt)
    f = np.divide(Xt[:,0]**2,rt) + np.divide(Xt[:,1]**2,rt)
    f = f - s
    f = np.multiply(w,f)
    
    # form the Jacobian matrix
    J = np.zeros((m,5))
    A1 = np.matmul(R,np.array([-1,0,0]).T)
    J[:,0] = A1[0] * Nt[:,0] + A1[1] * Nt[:,1]
    A2 = np.matmul(R,np.array([0,-1,0]).T)
    J[:,1] = A2[0] * Nt[:,0] + A2[1] * Nt[:,1]
    A3 = np.matmul(X-np.array([x0,y0,0]),DR1.T)
    J[:,2] = np.multiply(A3[:,0],Nt[:,0]) + np.multiply(A3[:,1],Nt[:,1])
    A4 = np.matmul(X-np.array([x0,y0,0]),DR2.T)
    J[:,3] = np.multiply(A4[:,0],Nt[:,0]) + np.multiply(A4[:,1],Nt[:,1])
    J[:,4] = -1 * np.ones(m)
    
    return f,J

def rot3z(a):
    '''
    Form rotation matrix U to rotate the vector a to a point along
    the positive z-axis. 
    
    Adapted from /Matlab_Tools/KneeACS/Tools/rot3z.m
    Created by I M Smith 2 May 2002

    Parameters
    ----------
    a : numpy.array
        3x1 array.

    Returns
    -------
    U : numpy.array
        3x3 array. Rotation matrix with U * a = [0 0 z]', z > 0. 

    '''

    # form first Givens rotation
    W, c1, s1 = gr(a[1], a[2])
    z = c1*a[1] + s1*a[2]
    V = np.array([[1,0,0],[0,s1,-c1],[0,c1,s1]])

    # form second Givens rotation
    W, c2, s2 = gr(a[0],z);
    
    # check positivity
    if c2 * a[0] + s2 * z < 0:
        c2 = -c2
        s2 = -s2
     
    W = np.array([[s2,0,-c2],[0,1,0],[c2,0,s2]])
    U = np.matmul(W,V)
      
    return U

def nlss11(ai,tol,p1,p2):
    '''
    Nonlinear least squares solver. Minimize f'*f.
    
    Adapted from /Matlab_Tools/KneeACS/Tools/nlss11.m
    by AB Forbes, CMSC, NPL

    Parameters
    ----------
    ai : numpy.array
        Optimisation parameters, intial estimates.
    tol : numpy.array
        Convergence tolerances [tolr tols]', where 
          tolr = relative tolerance, and 
          tols = scale for function values. 
          Dimension: 2 x 1. 
    p1 : numpy.array
        DESCRIPTION.
    p2 : numpy.array
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    a : numpy.array
        Solution estimates of the optimisation parameters.
          Dimension: n x 1.
    f : numpy.array
        Functions evaluated at a.
          Dimension: m x 1.
          Constraint: m >= n..
    R : numpy.array
        Triangular factor of the Jacobian matrix evaluated at a.
          Dimension: n x n.
    GNlog : list
        Log of the Gauss-Newton iterations. 
          Rows 1 to niter contain 
          [iter, norm(f_iter), |step_iter|, |gradient_iter|]. 
          Row (niter + 1) contains 
          [conv, norm(d), 0, 0]. 
          Dimension: (niter + 1) x 4. 

    '''
    a0 = ai
    n = len(a0)
    
    if n == 0:
        raise ValueError('Empty vector of parameter estimates')

    mxiter = int(100 + np.ceil(np.sqrt(n)))
    conv = 0
    niter = 0
    eta = 0.01
    GNlog = []
    
    # G-N iterations
    while (niter < mxiter) and (conv ==0):
        f0,J = fgcylinder(a0,p1,p2)
        if niter == 0:
            # scale by norm of columns of J
            mJ,nJ = J.shape
            scale = np.linalg.norm(J,axis=0)
        
        m = len(f0)
        # check on m, n
        if (niter==0) and (m<n):
            raise ValueError('Number of observation less than number of parameters')
        
        # Calculate update step and gradient
        F0 = np.linalg.norm(f0)
        _,Rqr = np.linalg.qr(np.concatenate((J,np.expand_dims(f0,axis=1)),axis=1))
        Ra = np.triu(Rqr)
        R = Ra[:nJ,:nJ]
        q = Ra[:nJ,nJ]
        p = np.matmul(np.linalg.inv(-R),q.T)
        g = 2 * np.matmul(R.T,q.T)
        G0 = np.matmul(g,p.T)
        a1 = a0 + p
        niter = niter+1
        
        # Check on convergence
        f1,J1 = fgcylinder(a1,p1,p2)
        F1 = np.linalg.norm(f1)
        # Gauss-Newton convergence conditions
        # from Matlab_Tools/KneeACS/Tools/gncc2.m
        #gncc2(F0, F1, p, g, scale, tol(1)=tolr, tol(2)=scalef);
        conv = 0
        sp = np.max(np.abs(p * scale))
        sg = np.max(np.abs(g / scale))
        c = np.full(5,np.nan)       
        c[0] = sp / (tol[1] * (tol[0]**0.7))
        c[1] = np.abs(F0-F1) / (tol[0]*tol[1])
        c[2] = sg / ((tol[0]**0.7)*tol[1])
        c[3] = F1 / (tol[1] * np.finfo(float).eps**0.7)
        c[4] = sg / ((np.finfo(float).eps**0.7)*tol[1])
        if (c[0] < 1) and (c[1] < 1) and (c[2] < 1):
            conv = 1
        elif (c[3] < 1) or (c[4] < 1):
            conv = 1

        if conv != 1:
            # otherwise check on reduction of sum of squares
            # evaluate f at a1
            rho = (F1-F0) * (F1+F0)/G0
            if rho < eta:
                tmin = np.max(np.array([0.001,1/(2*(1-rho))]))
                a0 = a0 + tmin * p
            else:
                a0 = a0 + p
        GNlog.append([niter,F0,sp,sg])
    
    a = a0 + p
    f = f1
    GNlog.append([conv,F1,0,0])
    
    return a, f, R, GNlog

def lscylinder(X,x0,a0,r0,tolp=0.1,tolg=0.1,w=None):
    '''
    Least-squares cylinder using Gauss-Newton
    
    Adapted from /Matlab_Tools/KneeACS/Tools/lscylinder.m
    by I M Smith 27 May 2002

    Parameters
    ----------
    X : numpy.array
        Array [x y z] where x = vector of x-coordinates, 
        y = vector of y-coordinates and z = vector of z-coordinates.
        Dimension: m x 3. 
    x0 : numpy.array
        Estimate of the point on the axis. 
        Dimension: 3 x 1. 
    a0 : numpy.array
        Estimate of the axis direction. 
        Dimension: 3 x 1.
    r0 : float
        Estimate of the cylinder radius. 
        Dimension: 1 x 1.
    tolp : float, optional
        Tolerance for test on step length. The default is 0.1.
    tolg : float, optional
        Tolerance for test on gradient. The default is 0.1.
    w : numpy.array, optional
        Weights. The default is None. If None, it will be an array of ones.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    x0n : numpy.array
        Estimate of the point on the axis. Dimension: 3x1
    an : numpy.array
        Estimate of the axis direction. Dimension: 3x1
    rn : float
        Estimate of the cylinder radius.
    stats : dict
        Dictionary of dditonal statistics and results.
        stats = {'sigmah':sigmah,'conv':conv,'Vx0n':Vx0n,'Van':Van,'urn':urn,'GNlog':GNlog,
                 'a':a,'R0':R0,'R':R}
        sigmah   Estimate of the standard deviation of the weighted residual errors. 
                Dimension: 1 x 1. 
 
        conv     If conv = 1 the algorithm has converged, if conv = 0 the algorithm
                has not converged and x0n, rn, d, and sigmah are current estimates. 
                Dimension: 1 x 1. 
 
        Vx0n     Covariance matrix of point on the axis. Dimension: 3 x 3. 

        Van      Covariance matrix of axis direction. Dimension: 3 x 3. 

        urn      Uncertainty in cylinder radius. Dimension: 1 x 1. 
 
        GNlog    Log of the Gauss-Newton iterations. 
                Rows 1 to niter contain [iter, norm(f_iter), |step_iter|, |gradient_iter|]. 
                Row (niter + 1) contains [conv, norm(d), 0, 0]. 
                Dimension: (niter + 1) x 4. 
 
        a        Optimisation parameters at the solution. Dimension: 5 x 1. 
 
        R0       Fixed rotation matrix. Dimension: 3 x 3. 
 
        R        Upper-triangular factor of the Jacobian matrix at the solution. 
                Dimension: 5 x 5.     

    '''
    m = X.shape[0]
    if m < 5:
        raise ValueError('At least 5 data points required')
    
    if w is None:
        w = np.ones(m)

    # find the centroid of the data
    xb = X.mean(axis=0)
    
    # transform the data to close to standard position via a rotation 
    # followed by a translation 
    R0 = rot3z(a0) # U * a0 = [0 0 1]' 
    x1 = np.matmul(R0,x0)
    xb1 = np.matmul(R0,xb) 
    # find xp, the point on axis nearest the centroid of the rotated data 
    t = x1 + (xb1[2] - x1[2]) * np.array([0,0,1]) 
    X2 = np.matmul(X,R0.T) - t
    x2 = x1 - t
    xb2 = xb1 - t
     
    ai = np.array([0,0,0,0,r0]) 
    tol = np.array([tolp,tolg])     
    
    # Gauss-Newton algorithm to find estimates of roto-translation parameters
    # that transform the data so that the best-fit circle is one in the standard
    # position
    a, d, R, GNlog = nlss11(ai,tol,X2,w)
    
    # inverse transformation to find axis and point on axis corresponding to 
    # original data
    rn = a[4]
    R3, DR1, DR2, DR3 = fgrrot3(np.array([a[2],a[3],0]))
    an = np.matmul(R0.T,np.matmul(R3.T,np.array([0,0,1]).T))
    p = np.matmul(R3,(xb2-np.array([a[0],a[1],0])).T)
    pz = np.array([0,0,p[2]])
    x0n = np.matmul(R0.T,(t + np.array([a[0],a[1],0]) + np.matmul(R3.T,pz.T)).T)
    
    nGN = len(GNlog)
    conv = GNlog[nGN-1][0]
    if conv == 0:
        print(' *** Gauss-Newton algorithm has not converged ***')
    
    # Calculate statistics
    dof = m - 5
    sigmah = np.linalg.norm(d)/np.sqrt(dof)
    ez = np.array([0,0,1])
    G = np.zeros((7,5))
    # derivatives of x0n
    dp1 = np.matmul(R3,np.array([-1,0,0]).T)
    dp2 = np.matmul(R3,np.array([0,-1,0]).T)
    dp3 = np.matmul(DR1,(xb2 - np.array([a[0],a[1],0])).T)
    dp4 = np.matmul(DR2,(xb2 - np.array([a[0],a[1],0])).T)
    G[0:3,0] = np.matmul(R0.T,np.array([1,0,0]) + np.matmul(R3.T,np.array([0,0,np.matmul(dp1.T,ez)]).T))
    G[0:3,1] = np.matmul(R0.T,np.array([0,1,0]) + np.matmul(R3.T,np.array([0,0,np.matmul(dp2.T,ez)]).T))
    G[0:3,2] = np.matmul(R0.T,np.matmul(DR1.T,np.array([0,0,np.matmul(p.T,ez)]).T) + \
                    np.matmul(R3.T,np.array([0,0,np.matmul(dp3.T,ez)]).T))
    G[0:3,3] = np.matmul(R0.T,np.matmul(DR2.T,np.array([0,0,np.matmul(p.T,ez)]).T) + \
                    np.matmul(R3.T,np.array([0,0,np.matmul(dp4.T,ez)]).T))
    # derivatives of an
    G[3:6,2] = np.matmul(R0.T,np.matmul(DR1.T,np.array([0,0,1]).T))
    G[3:6,3] = np.matmul(R0.T,np.matmul(DR2.T,np.array([0,0,1]).T))
    # derivatives of rn
    G[6,4] = 1
    Gt = np.matmul(np.linalg.inv(R.T),sigmah*G.T)
    Va = np.matmul(Gt.T,Gt)
    Vx0n = Va[0:3,0:3] # covariance matrix for x0n
    Van = Va[3:6,3:6] # covariance matrix for an
    urn = np.sqrt(Va[6,6]) # uncertainty in rn
    
    stats = {'d':d,'sigmah':sigmah,'conv':conv,'Vx0n':Vx0n,'Van':Van,'urn':urn,'GNlog':GNlog,
             'a':a,'R0':R0,'R':R}
    
    return x0n,an,rn,stats