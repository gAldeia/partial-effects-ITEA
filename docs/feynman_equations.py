import autograd.numpy as npgrad
import numpy          as np


pi = np.pi


original_expressions = {
    "I.10.7"    : 'm_0/npgrad.sqrt(1-v**2/c**2)',
    "I.11.19"   : 'x1*y1+x2*y2+x3*y3',
    "I.12.1"    : 'mu*Nn',
    "I.12.11"   : 'q*(Ef+B*v*npgrad.sin(theta))',
    "I.12.2"    : 'q1*q2*r/(4*pi*epsilon*r**3)',
    "I.12.4"    : 'q1*r/(4*pi*epsilon*r**3)',
    "I.12.5"    : 'q2*Ef',
    "I.13.12"   : 'G*m1*m2*(1/r2-1/r1)',
    "I.13.4"    : '1/2*m*(v**2+u**2+w**2)',
    "I.14.3"    : 'm*g*z',
    "I.14.4"    : '1/2*k_spring*x**2',
    "I.15.10"   : 'm_0*v/npgrad.sqrt(1-v**2/c**2)',
    "I.15.3t"   : '(t-u*x/c**2)/npgrad.sqrt(1-u**2/c**2)',
    "I.15.3x"   : '(x-u*t)/npgrad.sqrt(1-u**2/c**2)',
    "I.16.6"    : '(u+v)/(1+u*v/c**2)',
    "I.18.12"   : 'r*F*npgrad.sin(theta)',
    "I.18.14"   : 'm*r*v*npgrad.sin(theta)',
    "I.18.4"    : '(m1*r1+m2*r2)/(m1+m2)',
    "I.24.6"    : '1/2*m*(omega**2+omega_0**2)*1/2*x**2',
    "I.25.13"   : 'q/C',
    "I.26.2"    : 'npgrad.arcsin(n*npgrad.sin(theta2))',
    "I.27.6"    : '1/(1/d1+n/d2)',
    "I.29.16"   : 'npgrad.sqrt(x1**2+x2**2-2*x1*x2*npgrad.cos(theta1-theta2))',
    "I.29.4"    : 'omega/c',
    "I.30.3"    : 'Int_0*npgrad.sin(n*theta/2)**2/npgrad.sin(theta/2)**2',
    "I.30.5"    : 'npgrad.arcsin(lambd/(n*d))',
    "I.32.17"   : '(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)',
    "I.32.5"    : 'q**2*a**2/(6*pi*epsilon*c**3)',
    "I.34.1"    : 'omega_0/(1-v/c)',
    "I.34.14"   : '(1+v/c)/npgrad.sqrt(1-v**2/c**2)*omega_0',
    "I.34.27"   : '(h/(2*pi))*omega',
    "I.34.8"    : 'q*v*B/p',
    "I.37.4"    : 'I1+I2+2*npgrad.sqrt(I1*I2)*npgrad.cos(delta)',
    "I.38.12"   : '4*pi*epsilon*(h/(2*pi))**2/(m*q**2)',
    "I.39.1"    : '3/2*pr*V',
    "I.39.11"   : '1/(gamma-1)*pr*V',
    "I.39.22"   : 'n*kb*T/V',
    "I.40.1"    : 'n_0*npgrad.exp(-m*g*x/(kb*T))',
    "I.41.16"   : 'h/(2*pi)*omega**3/(pi**2*c**2*(npgrad.exp((h/(2*pi))*omega/(kb*T))-1))',
    "I.43.16"   : 'mu_drift*q*Volt/d',
    "I.43.31"   : 'mob*kb*T',
    "I.43.43"   : '1/(gamma-1)*kb*v/A',
    "I.44.4"    : 'n*kb*T*npgrad.log(V2/V1)',
    "I.47.23"   : 'npgrad.sqrt(gamma*pr/rho)',
    "I.48.20"   : 'm*c**2/npgrad.sqrt(1-v**2/c**2)',
    "I.50.26"   : 'x1*(npgrad.cos(omega*t)+alpha*npgrad.cos(omega*t)**2)',
    "I.6.2"     : 'npgrad.exp(-(theta/sigma)**2/2)/(npgrad.sqrt(2*pi)*sigma)',
    "I.6.2a"    : 'npgrad.exp(-theta**2/2)/npgrad.sqrt(2*pi)',
    "I.6.2b"    : 'npgrad.exp(-((theta-theta1)/sigma)**2/2)/(npgrad.sqrt(2*pi)*sigma)',
    "I.8.14"    : 'npgrad.sqrt((x2-x1)**2+(y2-y1)**2)',
    "I.9.18"    : 'G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)',
    "II.10.9"   : 'sigma_den/epsilon*1/(1+chi)',
    "II.11.17"  : 'n_0*(1+p_d*Ef*npgrad.cos(theta)/(kb*T))',
    "II.11.20"  : 'n_rho*p_d**2*Ef/(3*kb*T)',
    "II.11.27"  : 'n*alpha/(1-(n*alpha/3))*epsilon*Ef',
    "II.11.28"  : '1+n*alpha/(1-(n*alpha/3))',
    "II.11.3"   : 'q*Ef/(m*(omega_0**2-omega**2))',
    "II.13.17"  : '1/(4*pi*epsilon*c**2)*2*I/r',
    "II.13.23"  : 'rho_c_0/npgrad.sqrt(1-v**2/c**2)',
    "II.13.34"  : 'rho_c_0*v/npgrad.sqrt(1-v**2/c**2)',
    "II.15.4"   : '-mom*B*npgrad.cos(theta)',
    "II.15.5"   : '-p_d*Ef*npgrad.cos(theta)',
    "II.2.42"   : 'kappa*(T2-T1)*A/d',
    "II.21.32"  : 'q/(4*pi*epsilon*r*(1-v/c))',
    "II.24.17"  : 'npgrad.sqrt(omega**2/c**2-pi**2/d**2)',
    "II.27.16"  : 'epsilon*c*Ef**2',
    "II.27.18"  : 'epsilon*Ef**2',
    "II.3.24"   : 'Pwr/(4*pi*r**2)',
    "II.34.11"  : 'g_*q*B/(2*m)',
    "II.34.2"   : 'q*v*r/2',
    "II.34.29a" : 'q*h/(4*pi*m)',
    "II.34.29b" : 'g_*mom*B*Jz/(h/(2*pi))',
    "II.34.2a"  : 'q*v/(2*pi*r)',
    "II.35.18"  : 'n_0/(npgrad.exp(mom*B/(kb*T))+npgrad.exp(-mom*B/(kb*T)))',
    "II.35.21"  : 'n_rho*mom*npgrad.tanh(mom*B/(kb*T))',
    "II.36.38"  : 'mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M',
    "II.37.1"   : 'mom*(1+chi)*B',
    "II.38.14"  : 'Y/(2*(1+sigma))',
    "II.38.3"   : 'Y*A*x/d',
    "II.4.23"   : 'q/(4*pi*epsilon*r)',
    "II.6.11"   : '1/(4*pi*epsilon)*p_d*npgrad.cos(theta)/r**2',
    "II.6.15a"  : 'p_d/(4*pi*epsilon)*3*z/r**5*npgrad.sqrt(x**2+y**2)',
    "II.6.15b"  : 'p_d/(4*pi*epsilon)*3*npgrad.cos(theta)*npgrad.sin(theta)/r**3',
    "II.8.31"   : 'epsilon*Ef**2/2',
    "II.8.7"    : '3/5*q**2/(4*pi*epsilon*d)',
    "III.10.19" : 'mom*npgrad.sqrt(Bx**2+By**2+Bz**2)',
    "III.12.43" : 'n*(h/(2*pi))',
    "III.13.18" : '2*E_n*d**2*k/(h/(2*pi))',
    "III.14.14" : 'I_0*(npgrad.exp(q*Volt/(kb*T))-1)',
    "III.15.12" : '2*U*(1-npgrad.cos(k*d))',
    "III.15.14" : '(h/(2*pi))**2/(2*E_n*d**2)',
    "III.15.27" : '2*pi*alpha/(n*d)',
    "III.17.37" : 'beta*(1+alpha*npgrad.cos(theta))',
    "III.19.51" : '-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)',
    "III.21.20" : '-rho_c_0*q*A_vec/m',
    "III.4.32"  : '1/(npgrad.exp((h/(2*pi))*omega/(kb*T))-1)',
    "III.4.33"  : '(h/(2*pi))*omega/(npgrad.exp((h/(2*pi))*omega/(kb*T))-1)',
    "III.7.38"  : '2*mom*B/(h/(2*pi))',
    "III.8.54"  : 'npgrad.sin(E_n*t/(h/(2*pi)))**2',
    "III.9.52"  : '(p_d*Ef*t/(h/(2*pi)))*npgrad.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2',
}

latex_expressions = {
    'I.6.2a'    : r'e^{(-\theta^2/2)}/\sqrt{(2*\pi)}',
    'I.6.2'     : r'e^{(-{(\theta/\sigma)}^2/2)}/{(\sqrt{(2*\pi)}*\sigma)}',
    'I.6.2b'    : r'e^{(-{({(\theta-\theta_1)}/\sigma)}^2/2)}/{(\sqrt{(2*\pi)}*\sigma)}',
    'I.8.14'    : r'\sqrt{({(x2-x1)}^2+{(y2-y1)}^2)}',
    'I.9.18'    : r'G*m1*m2/{({(x2-x1)}^2+{(y2-y1)}^2+{(z2-z1)}^2)}',
    'I.10.7'    : r'm_0/\sqrt{(1-v^2/c^2)}',
    'I.11.19'   : r'x1*y1+x2*y2+x3*y3',
    'I.12.1'    : r'mu*Nn',
    'I.12.2'    : r'q1*q2*r/{(4*\pi*\epsilon*r^3)}',
    'I.12.4'    : r'q1*r/{(4*\pi*\epsilon*r^3)}',
    'I.12.5'    : r'q2*Ef',
    'I.12.11'   : r'q*{(Ef+B*v*sin{(\theta)})}',
    'I.13.4'    : r'1/2*m*{(v^2+u^2+w^2)}',
    'I.13.12'   : r'G*m1*m2*{(1/r2-1/r1)}',
    'I.14.3'    : r'm*g*z',
    'I.14.4'    : r'1/2*k_spring*x^2',
    'I.15.3x'   : r'{(x-u*t)}/\sqrt{(1-u^2/c^2)}',
    'I.15.3t'   : r'{(t-u*x/c^2)}/\sqrt{(1-u^2/c^2)}',
    'I.15.10'   : r'm_0*v/\sqrt{(1-v^2/c^2)}',
    'I.16.6'    : r'{(u+v)}/{(1+u*v/c^2)}',
    'I.18.4'    : r'{(m1*r1+m2*r2)}/{(m1+m2)}',
    'I.18.12'   : r'r*F*sin{(\theta)}',
    'I.18.14'   : r'm*r*v*sin{(\theta)}',
    'I.24.6'    : r'1/2*m*{(\omega^2+\omega_0^2)}*1/2*x^2',
    'I.25.13'   : r'q/C',
    'I.26.2'    : r'arcsin{(n*sin{(\theta_2)})}',
    'I.27.6'    : r'1/{(1/d1+n/d2)}',
    'I.29.4'    : r'\omega/c',
    'I.29.16'   : r'\sqrt{(x1^2+x2^2-2*x1*x2*cos{(\theta_1-\theta_2)})}',
    'I.30.3'    : r'Int_0*sin{(n*\theta/2)}^2/sin{(\theta/2)}^2',
    'I.30.5'    : r'arcsin{(\lambda/{(n*d)})}',
    'I.32.5'    : r'q^2*a^2/{(6*\pi*\epsilon*c^3)}',
    'I.32.17'   : r'{(1/2*\epsilon*c*Ef^2)}*{(8*\pi*r^2/3)}*{(\omega^4/{(\omega^2-\omega_0^2)}^2)}',
    'I.34.8'    : r'q*v*B/p',
    'I.34.1'    : r'\omega_0/{(1-v/c)}',
    'I.34.14'   : r'{(1+v/c)}/\sqrt{(1-v^2/c^2)}*\omega_0',
    'I.34.27'   : r'{(h/{(2*\pi)})}*\omega',
    'I.37.4'    : r'I1+I2+2*\sqrt{(I1*I2)}*cos{(\delta)}',
    'I.38.12'   : r'4*\pi*\epsilon*{(h/{(2*\pi)})}^2/{(m*q^2)}',
    'I.39.1'    : r'3/2*pr*V',
    'I.39.11'   : r'1/{(\gamma-1)}*pr*V',
    'I.39.22'   : r'n*kb*T/V',
    'I.40.1'    : r'n_0*e^{(-m*g*x/{(kb*T)})}',
    'I.41.16'   : r'h/{(2*\pi)}*\omega^3/{(\pi^2*c^2*{(e^{({(h/{(2*\pi)})}*\omega/{(kb*T)})}-1)})}',
    'I.43.16'   : r'mu_drift*q*Volt/d',
    'I.43.31'   : r'mob*kb*T',
    'I.43.43'   : r'1/{(\gamma-1)}*kb*v/A',
    'I.44.4'    : r'n*kb*T*log{(V2/V1)}',
    'I.47.23'   : r'\sqrt{(\gamma*pr/\rho)}',
    'I.48.20'   : r'm*c^2/\sqrt{(1-v^2/c^2)}',
    'I.50.26'   : r'x1*{(cos{(\omega*t)}+\alpha*cos{(\omega*t)}^2)}',
    'II.2.42'   : r'kappa*{(T2-T1)}*A/d',
    'II.3.24'   : r'Pwr/{(4*\pi*r^2)}',
    'II.4.23'   : r'q/{(4*\pi*\epsilon*r)}',
    'II.6.11'   : r'1/{(4*\pi*\epsilon)}*p_d*cos{(\theta)}/r^2',
    'II.6.15a'  : r'p_d/{(4*\pi*\epsilon)}*3*z/r^5*\sqrt{(x^2+y^2)}',
    'II.6.15b'  : r'p_d/{(4*\pi*\epsilon)}*3*cos{(\theta)}*sin{(\theta)}/r^3',
    'II.8.7'    : r'3/5*q^2/{(4*\pi*\epsilon*d)}',
    'II.8.31'   : r'\epsilon*Ef^2/2',
    'II.10.9'   : r'\sigma_den/\epsilon*1/{(1+chi)}',
    'II.11.3'   : r'q*Ef/{(m*{(\omega_0^2-\omega^2)})}',
    'II.11.17'  : r'n_0*{(1+p_d*Ef*cos{(\theta)}/{(kb*T)})}',
    'II.11.20'  : r'n_\rho*p_d^2*Ef/{(3*kb*T)}',
    'II.11.27'  : r'n*\alpha/{(1-{(n*\alpha/3)})}*\epsilon*Ef',
    'II.11.28'  : r'1+n*\alpha/{(1-{(n*\alpha/3)})}',
    'II.13.17'  : r'1/{(4*\pi*\epsilon*c^2)}*2*I/r',
    'II.13.23'  : r'\rho_{c_0}/\sqrt{(1-v^2/c^2)}',
    'II.13.34'  : r'\rho_{c_0}*v/\sqrt{(1-v^2/c^2)}',
    'II.15.4'   : r'-mom*B*cos{(\theta)}',
    'II.15.5'   : r'-p_d*Ef*cos{(\theta)}',
    'II.21.32'  : r'q/{(4*\pi*\epsilon*r*{(1-v/c)})}',
    'II.24.17'  : r'\sqrt{(\omega^2/c^2-\pi^2/d^2)}',
    'II.27.16'  : r'\epsilon*c*Ef^2',
    'II.27.18'  : r'\epsilon*Ef^2',
    'II.34.2a'  : r'q*v/{(2*\pi*r)}',
    'II.34.2'   : r'q*v*r/2',
    'II.34.11'  : r'g_*q*B/{(2*m)}',
    'II.34.29a' : r'q*h/{(4*\pi*m)}',
    'II.34.29b' : r'g_*mom*B*Jz/{(h/{(2*\pi)})}',
    'II.35.18'   : r'n_0/{(e^{(mom*B/{(kb*T)})}+e^{(-mom*B/{(kb*T)})})}',
    'II.35.21'   : r'n_\rho*mom*tanh{(mom*B/{(kb*T)})}',
    'II.36.38'   : r'mom*H/{(kb*T)}+{(mom*\alpha)}/{(\epsilon*c^2*kb*T)}*M',
    'II.37.1'    : r'mom*{(1+chi)}*B',
    'II.38.3'    : r'Y*A*x/d',
    'II.38.14'   : r'Y/{(2*{(1+\sigma)})}',
    'III.4.32'   : r'1/{(e^{({(h/{(2*\pi)})}*\omega/{(kb*T)})}-1)}',
    'III.4.33'   : r'{(h/{(2*\pi)})}*\omega/{(e^{({(h/{(2*\pi)})}*\omega/{(kb*T)})}-1)}',
    'III.7.38'   : r'2*mom*B/{(h/{(2*\pi)})}',
    'III.8.54'   : r'sin{(E_n*t/{(h/{(2*\pi)})})}^2',
    'III.10.19'  : r'mom*\sqrt{(Bx^2+By^2+Bz^2)}',
    'III.12.43'  : r'n*{(h/{(2*\pi)})}',
    'III.13.18'  : r'2*E_n*d^2*k/{(h/{(2*\pi)})}',
    'III.14.14'  : r'I_0*{(e^{(q*Volt/{(kb*T)})}-1)}',
    'III.15.12'  : r'2*U*{(1-cos{(k*d)})}',
    'III.15.14'  : r'{(h/{(2*\pi)})}^2/{(2*E_n*d^2)}',
    'III.15.27'  : r'2*\pi*\alpha/{(n*d)}',
    'III.17.37'  : r'beta*{(1+\alpha*cos{(\theta)})}',
    'III.19.51'  : r'-m*q^4/{(2*{(4*\pi*\epsilon)}^2*{(h/{(2*\pi)})}^2)}*{(1/n^2)}',
    'III.21.20'  : r'-\rho_{c_0}*q*A_vec/m'
}

representable = {
    'I.6.2a'    : False,
    'I.6.2'     : False,
    'I.6.2b'    : False,
    'I.8.14'    : False,
    'I.9.18'    : False,
    'I.10.7'    : False,
    'I.11.19'   : True,
    'I.12.1'    : True,
    'I.12.2'    : True,
    'I.12.4'    : True,
    'I.12.5'    : True,
    'I.12.11'   : False,
    'I.13.4'    : True,
    'I.13.12'   : True,
    'I.14.3'    : True,
    'I.14.4'    : True,
    'I.15.3x'   : False,
    'I.15.3t'   : False,
    'I.15.10'   : False,
    'I.16.6'    : False,
    'I.18.4'    : False,
    'I.18.12'   : False,
    'I.18.14'   : False,
    'I.24.6'    : True,
    'I.25.13'   : True,
    'I.26.2'    : False,
    'I.27.6'    : False,
    'I.29.4'    : True,
    'I.29.16'   : False,
    'I.30.3'    : False,
    'I.30.5'    : True,
    'I.32.5'    : True,
    'I.32.17'   : False,
    'I.34.8'    : True,
    'I.34.1'    : False,
    'I.34.14'   : False,
    'I.34.27'   : True,
    'I.37.4'    : False,
    'I.38.12'   : True,
    'I.39.1'    : True,
    'I.39.11'   : False,
    'I.39.22'   : True,
    'I.40.1'    : True,
    'I.41.16'   : False,
    'I.43.16'   : True,
    'I.43.31'   : True,
    'I.43.43'   : False,
    'I.44.4'    : False,
    'I.47.23'   : True,
    'I.48.20'   : False,
    'I.50.26'   : False,
    'II.2.42'   : True,
    'II.3.24'   : True,
    'II.4.23'   : True,
    'II.6.11'   : False,
    'II.6.15a'  : False,
    'II.6.15b'  : False,
    'II.8.7'    : True,
    'II.8.31'   : True,
    'II.10.9'   : False,
    'II.11.3'   : False,
    'II.11.17'  : False,
    'II.11.20'  : True,
    'II.11.27'  : False,
    'II.11.28'  : False,
    'II.13.17'  : True,
    'II.13.23'  : False,
    'II.13.34'  : False,
    'II.15.4'   : False,
    'II.15.5'   : False,
    'II.21.32'  : False,
    'II.24.17'  : False,
    'II.27.16'  : True,
    'II.27.18'  : True,
    'II.34.2a'  : True,
    'II.34.2'   : True,
    'II.34.11'  : True,
    'II.34.29a' : True,
    'II.34.29b' : True,
    'II.35.18'  : False,
    'II.35.21'  : False,
    'II.36.38'  : True,
    'II.37.1'   : True,
    'II.38.3'   : True,
    'II.38.14'  : False,
    'III.4.32'  : False,
    'III.4.33'  : False,
    'III.7.38'  : True,
    'III.8.54'  : False,
    'III.10.19' : False,
    'III.12.43' : True,
    'III.13.18' : True,
    'III.14.14' : False,
    'III.15.12' : False,
    'III.15.14' : True,
    'III.15.27' : True,
    'III.17.37' : False,
    'III.19.51' : True,
    'III.21.20' : True,
}

original_lambdas = {
    "I.10.7"    : lambda args : (lambda m_0,v,c:m_0/npgrad.sqrt(1-v**2/c**2))(*args),
    "I.11.19"   : lambda args : (lambda x1,x2,x3,y1,y2,y3:x1*y1+x2*y2+x3*y3)(*args),
    "I.12.1"    : lambda args : (lambda mu,Nn:mu*Nn)(*args),
    "I.12.11"   : lambda args : (lambda q,Ef,B,v,theta:q*(Ef+B*v*npgrad.sin(theta)))(*args),
    "I.12.2"    : lambda args : (lambda q1,q2,epsilon,r:q1*q2*r/(4*pi*epsilon*r**3))(*args),
    "I.12.4"    : lambda args : (lambda q1,epsilon,r:q1*r/(4*pi*epsilon*r**3))(*args),
    "I.12.5"    : lambda args : (lambda q2,Ef:q2*Ef)(*args),
    "I.13.12"   : lambda args : (lambda m1,m2,r1,r2,G:G*m1*m2*(1/r2-1/r1))(*args),
    "I.13.4"    : lambda args : (lambda m,v,u,w:1/2*m*(v**2+u**2+w**2))(*args),
    "I.14.3"    : lambda args : (lambda m,g,z:m*g*z)(*args),
    "I.14.4"    : lambda args : (lambda k_spring,x:1/2*k_spring*x**2)(*args),
    "I.15.10"   : lambda args : (lambda m_0,v,c:m_0*v/npgrad.sqrt(1-v**2/c**2))(*args),
    "I.15.3t"   : lambda args : (lambda x,c,u,t:(t-u*x/c**2)/npgrad.sqrt(1-u**2/c**2))(*args),
    "I.15.3x"   : lambda args : (lambda x,u,c,t:(x-u*t)/npgrad.sqrt(1-u**2/c**2))(*args),
    "I.16.6"    : lambda args : (lambda c,v,u:(u+v)/(1+u*v/c**2))(*args),
    "I.18.12"   : lambda args : (lambda r,F,theta:r*F*npgrad.sin(theta))(*args),
    "I.18.14"   : lambda args : (lambda m,r,v,theta:m*r*v*npgrad.sin(theta))(*args),
    "I.18.4"    : lambda args : (lambda m1,m2,r1,r2:(m1*r1+m2*r2)/(m1+m2))(*args),
    "I.24.6"    : lambda args : (lambda m,omega,omega_0,x:1/2*m*(omega**2+omega_0**2)*1/2*x**2)(*args),
    "I.25.13"   : lambda args : (lambda q,C:q/C)(*args),
    "I.26.2"    : lambda args : (lambda n,theta2:npgrad.arcsin(n*npgrad.sin(theta2)))(*args),
    "I.27.6"    : lambda args : (lambda d1,d2,n:1/(1/d1+n/d2))(*args),
    "I.29.16"   : lambda args : (lambda x1,x2,theta1,theta2:npgrad.sqrt(x1**2+x2**2-2*x1*x2*npgrad.cos(theta1-theta2)))(*args),
    "I.29.4"    : lambda args : (lambda omega,c:omega/c)(*args),
    "I.30.3"    : lambda args : (lambda Int_0,theta,n:Int_0*npgrad.sin(n*theta/2)**2/npgrad.sin(theta/2)**2)(*args),
    "I.30.5"    : lambda args : (lambda lambd,d,n:npgrad.arcsin(lambd/(n*d)))(*args),
    "I.32.17"   : lambda args : (lambda epsilon,c,Ef,r,omega,omega_0:(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2))(*args),
    "I.32.5"    : lambda args : (lambda q,a,epsilon,c:q**2*a**2/(6*pi*epsilon*c**3))(*args),
    "I.34.1"    : lambda args : (lambda c,v,omega_0:omega_0/(1-v/c))(*args),
    "I.34.14"   : lambda args : (lambda c,v,omega_0:(1+v/c)/npgrad.sqrt(1-v**2/c**2)*omega_0)(*args),
    "I.34.27"   : lambda args : (lambda omega,h:(h/(2*pi))*omega)(*args),
    "I.34.8"    : lambda args : (lambda q,v,B,p:q*v*B/p)(*args),
    "I.37.4"    : lambda args : (lambda I1,I2,delta:I1+I2+2*npgrad.sqrt(I1*I2)*npgrad.cos(delta))(*args),
    "I.38.12"   : lambda args : (lambda m,q,h,epsilon:4*pi*epsilon*(h/(2*pi))**2/(m*q**2))(*args),
    "I.39.1"    : lambda args : (lambda pr,V:3/2*pr*V)(*args),
    "I.39.11"   : lambda args : (lambda gamma,pr,V:1/(gamma-1)*pr*V)(*args),
    "I.39.22"   : lambda args : (lambda n,T,V,kb:n*kb*T/V)(*args),
    "I.40.1"    : lambda args : (lambda n_0,m,x,T,g,kb:n_0*npgrad.exp(-m*g*x/(kb*T)))(*args),
    "I.41.16"   : lambda args : (lambda omega,T,h,kb,c:h/(2*pi)*omega**3/(pi**2*c**2*(npgrad.exp((h/(2*pi))*omega/(kb*T))-1)))(*args),
    "I.43.16"   : lambda args : (lambda mu_drift,q,Volt,d:mu_drift*q*Volt/d)(*args),
    "I.43.31"   : lambda args : (lambda mob,T,kb:mob*kb*T)(*args),
    "I.43.43"   : lambda args : (lambda gamma,kb,A,v:1/(gamma-1)*kb*v/A)(*args),
    "I.44.4"    : lambda args : (lambda n,kb,T,V1,V2:n*kb*T*npgrad.log(V2/V1))(*args),
    "I.47.23"   : lambda args : (lambda gamma,pr,rho:npgrad.sqrt(gamma*pr/rho))(*args),
    "I.48.20"   : lambda args : (lambda m,v,c:m*c**2/npgrad.sqrt(1-v**2/c**2))(*args),
    "I.50.26"   : lambda args : (lambda x1,omega,t,alpha:x1*(npgrad.cos(omega*t)+alpha*npgrad.cos(omega*t)**2))(*args),
    "I.6.2"     : lambda args : (lambda sigma,theta:npgrad.exp(-(theta/sigma)**2/2)/(npgrad.sqrt(2*pi)*sigma))(*args),
    "I.6.2a"    : lambda args : (lambda theta:npgrad.exp(-theta**2/2)/npgrad.sqrt(2*pi))(*args),
    "I.6.2b"    : lambda args : (lambda sigma,theta,theta1:npgrad.exp(-((theta-theta1)/sigma)**2/2)/(npgrad.sqrt(2*pi)*sigma))(*args),
    "I.8.14"    : lambda args : (lambda x1,x2,y1,y2:npgrad.sqrt((x2-x1)**2+(y2-y1)**2))(*args),
    "I.9.18"    : lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2:G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2))(*args),
    "II.10.9"   : lambda args : (lambda sigma_den,epsilon,chi:sigma_den/epsilon*1/(1+chi))(*args),
    "II.11.17"  : lambda args : (lambda n_0,kb,T,theta,p_d,Ef:n_0*(1+p_d*Ef*npgrad.cos(theta)/(kb*T)))(*args),
    "II.11.20"  : lambda args : (lambda n_rho,p_d,Ef,kb,T:n_rho*p_d**2*Ef/(3*kb*T))(*args),
    "II.11.27"  : lambda args : (lambda n,alpha,epsilon,Ef:n*alpha/(1-(n*alpha/3))*epsilon*Ef)(*args),
    "II.11.28"  : lambda args : (lambda n,alpha:1+n*alpha/(1-(n*alpha/3)))(*args),
    "II.11.3"   : lambda args : (lambda q,Ef,m,omega_0,omega:q*Ef/(m*(omega_0**2-omega**2)))(*args),
    "II.13.17"  : lambda args : (lambda epsilon,c,I,r:1/(4*pi*epsilon*c**2)*2*I/r)(*args),
    "II.13.23"  : lambda args : (lambda rho_c_0,v,c:rho_c_0/npgrad.sqrt(1-v**2/c**2))(*args),
    "II.13.34"  : lambda args : (lambda rho_c_0,v,c:rho_c_0*v/npgrad.sqrt(1-v**2/c**2))(*args),
    "II.15.4"   : lambda args : (lambda mom,B,theta:-mom*B*npgrad.cos(theta))(*args),
    "II.15.5"   : lambda args : (lambda p_d,Ef,theta:-p_d*Ef*npgrad.cos(theta))(*args),
    "II.2.42"   : lambda args : (lambda kappa,T1,T2,A,d:kappa*(T2-T1)*A/d)(*args),
    "II.21.32"  : lambda args : (lambda q,epsilon,r,v,c:q/(4*pi*epsilon*r*(1-v/c)))(*args),
    "II.24.17"  : lambda args : (lambda omega,c,d:npgrad.sqrt(omega**2/c**2-pi**2/d**2))(*args),
    "II.27.16"  : lambda args : (lambda epsilon,c,Ef:epsilon*c*Ef**2)(*args),
    "II.27.18"  : lambda args : (lambda epsilon,Ef:epsilon*Ef**2)(*args),
    "II.3.24"   : lambda args : (lambda Pwr,r:Pwr/(4*pi*r**2))(*args),
    "II.34.11"  : lambda args : (lambda g_,q,B,m:g_*q*B/(2*m))(*args),
    "II.34.2"   : lambda args : (lambda q,v,r:q*v*r/2)(*args),
    "II.34.29a" : lambda args : (lambda q,h,m:q*h/(4*pi*m))(*args),
    "II.34.29b" : lambda args : (lambda g_,h,Jz,mom,B:g_*mom*B*Jz/(h/(2*pi)))(*args),
    "II.34.2a"  : lambda args : (lambda q,v,r:q*v/(2*pi*r))(*args),
    "II.35.18"  : lambda args : (lambda n_0,kb,T,mom,B:n_0/(npgrad.exp(mom*B/(kb*T))+npgrad.exp(-mom*B/(kb*T))))(*args),
    "II.35.21"  : lambda args : (lambda n_rho,mom,B,kb,T:n_rho*mom*npgrad.tanh(mom*B/(kb*T)))(*args),
    "II.36.38"  : lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M:mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M)(*args),
    "II.37.1"   : lambda args : (lambda mom,B,chi:mom*(1+chi)*B)(*args),
    "II.38.14"  : lambda args : (lambda Y,sigma:Y/(2*(1+sigma)))(*args),
    "II.38.3"   : lambda args : (lambda Y,A,d,x:Y*A*x/d)(*args),
    "II.4.23"   : lambda args : (lambda q,epsilon,r:q/(4*pi*epsilon*r))(*args),
    "II.6.11"   : lambda args : (lambda epsilon,p_d,theta,r:1/(4*pi*epsilon)*p_d*npgrad.cos(theta)/r**2)(*args),
    "II.6.15a"  : lambda args : (lambda epsilon,p_d,r,x,y,z:p_d/(4*pi*epsilon)*3*z/r**5*npgrad.sqrt(x**2+y**2))(*args),
    "II.6.15b"  : lambda args : (lambda epsilon,p_d,theta,r:p_d/(4*pi*epsilon)*3*npgrad.cos(theta)*npgrad.sin(theta)/r**3)(*args),
    "II.8.31"   : lambda args : (lambda epsilon,Ef:epsilon*Ef**2/2)(*args),
    "II.8.7"    : lambda args : (lambda q,epsilon,d:3/5*q**2/(4*pi*epsilon*d))(*args),
    "III.10.19" : lambda args : (lambda mom,Bx,By,Bz:mom*npgrad.sqrt(Bx**2+By**2+Bz**2))(*args),
    "III.12.43" : lambda args : (lambda n,h:n*(h/(2*pi)))(*args),
    "III.13.18" : lambda args : (lambda E_n,d,k,h:2*E_n*d**2*k/(h/(2*pi)))(*args),
    "III.14.14" : lambda args : (lambda I_0,q,Volt,kb,T:I_0*(npgrad.exp(q*Volt/(kb*T))-1))(*args),
    "III.15.12" : lambda args : (lambda U,k,d:2*U*(1-npgrad.cos(k*d)))(*args),
    "III.15.14" : lambda args : (lambda h,E_n,d:(h/(2*pi))**2/(2*E_n*d**2))(*args),
    "III.15.27" : lambda args : (lambda alpha,n,d:2*pi*alpha/(n*d))(*args),
    "III.17.37" : lambda args : (lambda beta,alpha,theta:beta*(1+alpha*npgrad.cos(theta)))(*args),
    "III.19.51" : lambda args : (lambda m,q,h,n,epsilon:-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2))(*args),
    "III.21.20" : lambda args : (lambda rho_c_0,q,A_vec,m:-rho_c_0*q*A_vec/m)(*args),
    "III.4.32"  : lambda args : (lambda h,omega,kb,T:1/(npgrad.exp((h/(2*pi))*omega/(kb*T))-1))(*args),
    "III.4.33"  : lambda args : (lambda h,omega,kb,T:(h/(2*pi))*omega/(npgrad.exp((h/(2*pi))*omega/(kb*T))-1))(*args),
    "III.7.38"  : lambda args : (lambda mom,B,h:2*mom*B/(h/(2*pi)))(*args),
    "III.8.54"  : lambda args : (lambda E_n,t,h:npgrad.sin(E_n*t/(h/(2*pi)))**2)(*args),
    "III.9.52"  : lambda args : (lambda p_d,Ef,t,h,omega,omega_0:(p_d*Ef*t/(h/(2*pi)))*npgrad.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2)(*args),
}

datasets = list(original_lambdas.keys())