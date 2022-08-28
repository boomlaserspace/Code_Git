from cProfile import label
from calendar import c
import numpy as np 
import matplotlib.pyplot as plt 
Tc = 0.1
Gr_list = []
Fr_list = []
Tref_list = []
q_list = []
while Tc < 12.5:

    Th = 26 - Tc 
    T_mid = (Th + Tc)/2 # ==13 

    #ReferenzTemperatur in K 
    Tref = (Th - Tc)

    #Referenzlänge in m 
    lref = 0.03

    #Erdbeschleunigung in m/s^2
    g = 9.81 

    #Dichte Wasser  in kg/m^3
    rho0 = 999.38 

    #Dynamische Viskositöt in  kg/m*s
    mu = 1.2005e-3 

    #Kinematische Viskosität in m^2/s
    nu = mu / rho0 

    #Wärme-Kapazität in m^2/s^2*K
    cp = 4191 

    #Wärmeleitfähigkeit in (kg*m)/(s^2*K) 
    lam = 0.597 

   #Wärmeleit Koeffizient
    a= (nu * lam ) / (mu * cp)

    #Wärmeausdehnungskoeffizient in 1/K
    beta = 0.1267e-3 
 
    Ra = (g*beta*Tref*lref**3)/(nu*a)
    Fr = np.sqrt(beta*Tref)
    Pr = (mu * cp) / lam
    Gr = Ra / Pr
    Ec = (g* beta * Tref * lref)/mu**2


    ##Nusselt number as a function of Rayleigh and Prandtl
    Nu = (0.60 + (0.387*Ra)**(1/6)/(1 + (0.492/Pr)**(9/16))**(8/27))**2


    #Materialgesetz
    Gr_list.append(np.sqrt(Pr/Ra))

    #Bodyforces 
    Fr_list.append(1/Fr**2)

    #Wärmediffussion
    q_list.append(1./np.sqrt(Pr*Ra))


    #Nu_list.append()


    Tref_list.append(Tref)
    #print(np.sqrt(Pr/Ra),1/Fr**2,1/(1./np.sqrt(Pr*Ra)), Th ,Tc ,Tref)

    print("Rayleigh:",Ra,"Prandtl:",Pr, "TH-TC",Tref, "Nusselt", Nu)

    Tc+= 0.1





plt.figure(0)
plt.xlabel(r'$T_{ref}=T_H - T_C$')
plt.grid()
plt.plot(Tref_list, Gr_list, label =r'$2 \sqrt{\dfrac{1}{\mathrm{Gr}}}$')
plt.plot(Tref_list, q_list,label = r'$\dfrac{1}{\mathrm{Pr}\sqrt{\mathrm{Gr}}}$' )
plt.plot()


plt.legend()
plt.show()

















