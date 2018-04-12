#%%
#%matplotlib inline
import matplotlib.pyplot as plt  # Import library for direct plotting functions
import numpy as np               # Import Numerical Python
from IPython.core.display import display, HTML #Import HTML for formatting output

#%% Gaussian beam specification
w0 = 20*1.e-6  # beam waist
x_um = np.linspace(-15*w0*1.e6,15*w0*1.e6,100) # radial distribution in um
x = x_um *1.e-6 # radial distribution in m
y_um = np.linspace(-15*w0*1.e6,15*w0*1.e6,100) # radial distribution in um
y = y_um *1.e-6 # radial distribution in m

E0 = 1
lam= 780*1.e-9
k0  = 2*np.pi/lam
n0  = 1
zr = w0**2*n0*np.pi/lam # rayleigh range
def gaussbeam(x,y,z):
    wz = w0*np.sqrt(1+((z-z0)/zr)**2)
    Rz = (z-z0)*(1+(zr/(z-z0))**2)
    E = E0 * w0/wz * np.exp(-(x**2+y**2)/(1*wz**2)  + 1j*k0*(x**2+y**2)/(2*Rz)  -1j*np.arctan((z-z0)/zr))# + 1j*k*(z-z0))#  + 1j*(k*z-w*t))
    return E
z0 = 0

#%% create beam propagation using gaussian approximation
xx1,yy1  = np.meshgrid(x,y)

z1 = np.linspace(-.0015,.0005,200)  # position of the beam along the beam propagation axis
efld = []
for i in range(len(z1)):
#    if i%(len(z1)/10) == 0:
#        print('.',end='')
    zz1 = z1[i]
    efld.append(gaussbeam(xx1,yy1,zz1))
efld = np.array(efld)
#%%
efield = np.abs(efld[150])**2 # beam profile at the gaussian beam waist
fig = plt.figure(figsize=(16,12))
v = np.linspace(np.min(efield), np.max(efield), 100, endpoint=True)
cp = plt.contourf(xx1*1.e6,yy1*1.e6,efield, v,cmap='hot')
fig.colorbar(cp)
plt.xlabel('x position [um]',fontsize=16)
plt.ylabel('y position [um]',fontsize=16)
plt.title('Intensity Gaussian',fontsize=20)
plt.show()

#%% atomic cloud 
n_at = 120000          # number of trapped atoms 
l_cl = 1*1.e-3        # length of atomic cloud
sigma_cl = 9*1.e-6    # radial distribution of the cloud

def at_den(sigma,N):  # returns atomic density for a given total atom number and cloud waist
    c = N/(2*np.pi*l_cl*sigma**2)
    rho = c*np.exp(-(xx1**2+yy1**2)/(2*sigma**2))
    return rho
rho = at_den(sigma_cl,n_at)  

fig = plt.figure(figsize=(16,12))
v = np.linspace(np.min(rho), np.max(rho), 100, endpoint=True)
cp = plt.contourf(xx1*1.e6,yy1*1.e6,rho, v,cmap='hot')
fig.colorbar(cp)
plt.xlabel('x position [um]',fontsize=16)
plt.ylabel('y position [um]',fontsize=16)
plt.title('atom density',fontsize=20)
plt.show()

#%% calculate succeptibility, chi using the rho
lam = 780*1.e-9  			# wavelength in nm (780 nm is resonant for Rb atoms)
sigma0 = 3*lam**2/(2*np.pi) # scattering cross section
Gamm = 2*np.pi*6.067*1.e6   # Natural linewidht of Rb atoms
k  = 2*np.pi/lam     		# wavenumber

det = np.linspace(-50,50,50)  	# laser detuing in MHz
del_p = det*2*np.pi*1.e6		# detuing in Hz

def lineshape(detuning,rho):    # returns succeptibility for two level system
    trm1 = -sigma0*rho/k
    trm2 = 2*detuning/Gamm -1j
    trm3 = 1 + 4*(detuning/Gamm)**2
    chi = trm1*trm2/trm3
    return chi

chi = []
for i in range(len(det)):
    chi.append(lineshape(del_p[i],rho))
chi = np.array(chi)             # succeptibility for all detunings   
nr  = 1 + 1/2* np.real(chi)     # real part of refractive index
ni  = 1/2* np.imag(chi)			# imaginary part of refractive index

#%%

det_ind = 15

fig = plt.figure(figsize=(16,12))
v = np.linspace(np.min(ni), np.max(ni), 100, endpoint=True)
cp = plt.contourf(xx1*1.e6,yy1*1.e6,ni[det_ind], v,cmap='hot')
fig.colorbar(cp)
plt.xlabel('x axis [um]',fontsize=16)
plt.ylabel('y axis [um]',fontsize=16)
plt.title('imaginary part of refractive index',fontsize=20)
plt.show()

fig = plt.figure(figsize=(16,12))
v = np.linspace(np.min(nr), np.max(nr), 100, endpoint=True)
cp = plt.contourf(xx1*1.e6,yy1*1.e6,nr[det_ind], v,cmap='bwr')
fig.colorbar(cp)
plt.xlabel('x axis [um]',fontsize=16)
plt.ylabel('y axis [um]',fontsize=16)
plt.title('real part of refractive inedex',fontsize=20)
plt.show()
det[det_ind]

#%% fourier transformed axes (frequency basis)
dx = x[1]-x[0]
dy = y[1]-y[0]

kx = np.fft.fftfreq(x.shape[0])*2*np.pi/dx
ky = np.fft.fftfreq(y.shape[0])*2*np.pi/dy

kkx,kky = np.meshgrid(kx,ky)

#%% FFT beam propagation integral
dz = z1[1] - z1[0]  # step size along the beam propagation direction
in_fld = efld[0]    # initial electric field at the input of the cloud 
lam = 780*1.e-9		# wavelength
k0 = 2*np.pi/lam	# wavenumber
beta = k0			

def fftbpm(in_fld,chi,nr):
    fftfld1 = np.fft.fft2(in_fld)
    ifftfld = fftfld1*np.exp(-1j*(kkx**2+kky**2)*dz/(2*beta*nr))    
    out_fld = np.fft.ifft2(ifftfld)*np.exp(1j*k0/(2*nr**2)*chi*dz)
    return out_fld

pos_cld = z1[0]  # initial position of the cloud

def fftcal(in_fld,chi,nr):
    phi_z = []
    for i in range(len(z1)):
        if i%(len(z1)/10) ==0:
            print('.',end='')
        if z1[i] < pos_cld:   								# before the cloud
            out_fld = fftbpm(in_fld,0,1)
            phi_z.append(in_fld)
            in_fld = out_fld
        if (z1[i] < pos_cld + l_cl) * (z1[i] >= pos_cld):	# cloud region
            out_fld = fftbpm(in_fld,chi,nr)
            phi_z.append(in_fld)
            in_fld = out_fld
        if z1[i] >= pos_cld + l_cl:							# after the cloud
            out_fld = fftbpm(in_fld,0,1)
            phi_z.append(in_fld)
            in_fld = out_fld
    phi_z = np.array(phi_z)
    return phi_z

#%% generate list of x,y,z electric field profile for different detuning
xt = round(len(x)/2) # middle of the beam (along x or y axis)
psi1 = []
for i in range(len(det)):
    in_fld = efld[0]
    phph = fftcal(in_fld,chi[i],nr[i])
    psi1.append(phph.T[xt])
    print(i)
#% 
psi1 = np.array(psi1) 		# complex electric field of the beam as a function of radial and axial direction
psi0 = fftcal(in_fld,0,1)   # reference complex electric field (without atomic presence)

#%%

gaus_fld_ref = efld.T[xt]    # reference gaussian propagation (somewhat analytical solution)
fft_fld_ref = psi0.T[xt]     # reference fft model, numerical solution

fr_ind = 14
fft_fld_sig = psi1[fr_ind]   # reference cloud signal, numerical solution

rad1,ax1 = np.meshgrid(x,z1) # meshgrid of radial and axial vectors

#% intensity plots
gaus_int_ref = np.abs(gaus_fld_ref)**2 
fft_int_ref = np.abs(fft_fld_ref)**2 
fft_int_sig = np.abs(fft_fld_sig)**2 

fig = plt.figure(figsize=(16,8))
v = np.linspace(np.min(gaus_int_ref), np.max(gaus_int_ref), 100, endpoint=True)
cp = plt.contourf(ax1*1.e3,rad1*1.e6,gaus_int_ref.T, v,cmap='hot')
fig.colorbar(cp)
plt.xlabel('z axis [mm]',fontsize=16)
plt.ylabel('y axis [um]',fontsize=16)
plt.title('Analytical ref profile'.format(det[fr_ind]),fontsize=20)
plt.ylim([-100,100])
plt.show()
img_name_pdf = 'Intensity_profile_free_space_analytical' + '.png'
fig.savefig(img_name_pdf)

fig = plt.figure(figsize=(16,8))
#v = np.linspace(np.min(gaus_int_ref), np.max(gaus_int_ref), 100, endpoint=True)
v = np.linspace(np.min(fft_int_ref),np.max(fft_int_ref), 100, endpoint=True)
cp = plt.contourf(ax1*1.e3,rad1*1.e6,fft_int_ref.T, v,cmap='hot')
fig.colorbar(cp)
plt.xlabel('z axis [mm]',fontsize=16)
plt.ylabel('y axis [um]',fontsize=16)
plt.title('fft intensity ref profile'.format(det[fr_ind]),fontsize=20)
plt.ylim([-100,100])
plt.show()
img_name_pdf = 'Intensity_profile_free_space_numerical' + '.png'
fig.savefig(img_name_pdf)

#%%
fig = plt.figure(figsize=(16,8))
v = np.linspace(np.min(gaus_int_ref), np.max(gaus_int_ref), 100, endpoint=True)
cp = plt.contourf(ax1*1.e3,rad1*1.e6,fft_int_sig.T, v,cmap='hot')
fig.colorbar(cp)
plt.xlabel('z axis [mm]',fontsize=16)
plt.ylabel('y axis [um]',fontsize=16)
plt.title('fft intensity signal at {:.2f} MHz'.format(det[fr_ind]),fontsize=20)
plt.ylim([-100,100])
plt.show()#%% step index
img_name_pdf = 'Intensity_profile_cloud_21MHz_neg' + '.png'
fig.savefig(img_name_pdf)

#%% gaussian overlap integral to estimate the total transmission

def effcnc(sig1,sig0,x):
    ef1 = sig1
    ef0 = sig0
    X = x

    ov1 = (np.abs( np.sum(np.conjugate(ef1)*ef0*np.pi*np.abs(X)) ) )**2
    ov2 = (np.sum(  (np.abs(ef1))**2*np.pi*np.abs(X) )  )
    ov3 = (np.sum(  (np.abs(ef0))**2*np.pi*np.abs(X) )  )
    efcnc = ov1/(ov2*ov3)
    return efcnc

#% mode filtering and final detection
fib_core_rad = 26*1.e-6
msk_ind = np.arange(len(x))
cmpr1 = len(x)-(-fib_core_rad < x) * msk_ind[::-1]-1
cmpr2 = (fib_core_rad > x) * msk_ind

msk1 =np.min(cmpr1)
msk2 =np.max(cmpr2)

fr_ind = 10
fftfld = psi1[fr_ind]
reffld = psi0.T[xt]

def pwrout(fftfld):
    pwr = []
    efc = []
    for i in range(len(z1)):
        pw = np.abs(fftfld.T[i])**2 * np.abs(x)
        ef1 = fftfld.T[i]
        ef0 = reffld.T[i]
        if z1[i] < 0:
            pwr.append(np.sum(pw))
            efc.append(effcnc(ef1,ef0,x))
        else:
            pwr.append(np.sum(pw[msk1:msk2]))
            efc.append(effcnc(ef1[msk1:msk2],ef0[msk1:msk2],x[msk1:msk2]))
    pwr = np.array(pwr)
    pwr = pwr/pwr[0]
    efc = np.array(efc)
    return pwr, efc
#%
pout = []
efcc = []
for i in range(len(det)):
    print('.', end='')
    fr_ind = i
    fftfld = psi1[fr_ind]
    pout.append(pwrout(fftfld)[0])
    efcc.append(pwrout(fftfld)[1])

pout = np.array(pout)
efcc = np.array(efcc)

#%% final transmission plot vs detuning
loc = 160
op = pout.T[loc]
ce = efcc.T[loc]
out_tot = op*ce

fig = plt.figure(figsize=(12,8))
plt.plot(det,op,'g.--')
plt.plot(det,out_tot,'r.--')
plt.xlabel('detuning [MHz]')
plt.ylabel('Transmission')
plt.title('Tranmission')
plt.legend(['Only due to absoprion',' Dispersive part included'])
plt.ylim([0,1])
plt.grid()

img_name_png = 'Transmission' + '.png'
fig.savefig(img_name_png)








