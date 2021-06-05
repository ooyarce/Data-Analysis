import matplotlib.pyplot as plt
import statistics as stats
import math as mt
import numpy as np


def graficar_sd(data):
    t1   = data[0]
    t2   = data[1]
    a_1  = data[2]
    a_2  = data[3]
    w    = data[4]
    T    = data[5]
    chi1 = data[6]
    
    Sd = spectral_response(t1,a_1,w,chi1)[0]
    Sd2 = spectral_response(t2,a_2,w,chi1)[0]

    plt.plot(T,Sd)
    plt.plot(T,Sd2)

    plt.grid()
    plt.xlim([0,3])
    plt.title("Spectral Relative Displacement with chi = 3%")
    plt.xlabel("Period T [s]")
    plt.ylabel("Sd [m]")
    plt.legend(["Los Gatos", "Tarzana"])
    plt.show()
    
def graficar_sd2(data):
    t1   = data[0]
    t2   = data[1]
    a_1  = data[2]
    a_2  = data[3]
    w    = data[4]
    T    = data[5]
    chi2 = data[7]
    
    Sd3 = spectral_response(t1,a_1,w,chi2)[0]
    Sd4 = spectral_response(t2,a_2,w,chi2)[0]
    
    plt.plot(T,Sd3)
    plt.plot(T,Sd4)
    plt.grid()
    plt.xlim([0,3])
    plt.title("Spectral Relative Displacement with chi = 5%")
    plt.xlabel("Period T [s]")
    plt.ylabel("Sd [m]")
    plt.legend(["Los Gatos", "Tarzana"])
    plt.show()
    
def graficar_sv(data):
    t1   = data[0]
    t2   = data[1]
    a_1  = data[2]
    a_2  = data[3]
    w    = data[4]
    T    = data[5]
    chi1 = data[6]
    
    Sd = spectral_response(t1,a_1,w,chi1)[1]
    Sd2 = spectral_response(t2,a_2,w,chi1)[1]
    Sd3 = spectral_response(t1,a_1,w,chi1)[3]
    Sd4 = spectral_response(t2,a_2,w,chi1)[3]
    
    plt.plot(T,Sd)
    plt.plot(T,Sd2)
    plt.plot(T,Sd3,"--")
    plt.plot(T,Sd4,"--")
    plt.grid()
    plt.xlim([0,3])
    plt.title("Spectral Relative Velocity with chi = 3%")
    plt.xlabel("Period T [s]")
    plt.ylabel("Sv [m/s]")
    plt.legend(["Los Gatos", "Tarzana","Spv Los Gatos", "Spv Tarzana"])
    plt.show()
    
def graficar_sv2(data):
    t1   = data[0]
    t2   = data[1]
    a_1  = data[2]
    a_2  = data[3]
    w    = data[4]
    T    = data[5]
    chi2 = data[7]
    
    Sd = spectral_response(t1,a_1,w,chi2)[1]
    Sd2 = spectral_response(t2,a_2,w,chi2)[1]
    Sd3 = spectral_response(t1,a_1,w,chi2)[3]
    Sd4 = spectral_response(t2,a_2,w,chi2)[3]
    
    plt.plot(T,Sd)
    plt.plot(T,Sd2)
    plt.plot(T,Sd3,"--")
    plt.plot(T,Sd4,"--")
    plt.grid()
    plt.xlim([0,3])
    plt.title("Spectral Relative Velocity with chi = 5%")
    plt.xlabel("Period T [s]")
    plt.ylabel("Sv [m/s]")
    plt.legend(["Los Gatos", "Tarzana","Spv Los Gatos", "Spv Tarzana"])
    plt.show()
    
def graficar_sa(data):
    t1   = data[0]
    t2   = data[1]
    a_1  = data[2]
    a_2  = data[3]
    w    = data[4]
    T    = data[5]
    chi1 = data[6]

    
    Sd = spectral_response(t1,a_1,w,chi1)[2]
    Sd2 = spectral_response(t2,a_2,w,chi1)[2]
    Sd3 = spectral_response(t1,a_1,w,chi1)[4]
    Sd4 = spectral_response(t2,a_2,w,chi1)[4]
    
    plt.plot(T,Sd)
    plt.plot(T,Sd2)
    plt.plot(T,Sd3,"--")
    plt.plot(T,Sd4,"--")
    plt.grid()
    plt.xlim([0,3])
    plt.title("Spectral Relative Acceleration with chi = 3%")
    plt.xlabel("Period T [s]")
    plt.ylabel("Sa [m/s^2]")
    plt.legend(["Los Gatos", "Tarzana","Spa Los Gatos", "Spa Tarzana"])
    plt.show()
    
def graficar_sa2(data):
    t1   = data[0]
    t2   = data[1]
    a_1  = data[2]
    a_2  = data[3]
    w    = data[4]
    T    = data[5]
    chi2 = data[7]

    
    Sd = spectral_response(t1,a_1,w,chi2)[2]
    Sd2 = spectral_response(t2,a_2,w,chi2)[2]
    Sd3 = spectral_response(t1,a_1,w,chi2)[4]
    Sd4 = spectral_response(t2,a_2,w,chi2)[4]
    
    plt.plot(T,Sd)
    plt.plot(T,Sd2)
    plt.plot(T,Sd3,"--")
    plt.plot(T,Sd4,"--")
    plt.grid()
    plt.xlim([0,3])
    plt.title("Spectral Relative Acceleration with chi = 5%")
    plt.xlabel("Period T [s]")
    plt.ylabel("Sa [m/s^2]")
    plt.legend(["Los Gatos", "Tarzana","Spa Los Gatos", "Spa Tarzana"])
    plt.show()

def graficar(t,a,r,title):
    a2 = []
    
    for i in range(len(a)):
        a2.append(a[i]/100)
        
    plt.plot(t,a2)
    plt.grid()
    plt.xlim(r)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [g]")
    plt.show()
    
def graficar2(t1,t2,a1,a2):
    ace = []
    ace2 = []
    for i in range(len(a1)):
        ace.append(a1[i]/100)
    for i in range(len(a2)):
        ace2.append(a2[i]/100)
        
    plt.plot(t2,ace2)
    plt.plot(t1,ace)
    plt.grid()
    plt.xlim([0,30])
    plt.title("Earthquake Acceleration")
    plt.xlabel("Time [s]")
    plt.legend(["Northridge 1994 - Estación Tarzana","Loma Prieta 1989 - Los Gatos"])
    plt.ylabel("Acceleration [g]")
    plt.show()
    
def graficaru(data,r,title):
    u_piese = enlistar(data[0],data[2],data[3])[0][0] #largo 2000
    u_rect = enlistar(data[0],data[2],data[3])[0][1] #largo 1999
    u_newman = enlistar(data[0],data[2],data[3])[0][2] #largo 2000
    plt.plot(data[0],u_piese,"-")
    plt.plot(data[1],u_rect,"--")
    plt.plot(data[0],u_newman,":")
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    #plt.yticks(r[0],r[1])
    plt.xlim(r)
    plt.legend(["Piese Wise Linear", "Duhamel Rect", "Newmark"])
    plt.title(title)
    plt.show()
    
def graficarv(data,r,title):
    u_piese = enlistar(data[0],data[2],data[3])[1][0] #largo 2000
    u_rect = enlistar(data[0],data[2],data[3])[1][1] #largo 1999
    u_newman = enlistar(data[0],data[2],data[3])[1][2] #largo 2000
    plt.plot(data[0],u_piese,"-")
    plt.plot(data[1],u_rect,"--")
    plt.plot(data[0],u_newman,":")
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    #plt.yticks(r[2],r[3])
    plt.legend(["Piese Wise Linear", "Duhamel Rect", "Newmark"])
    plt.xlim(r)
    plt.title(title)
    plt.show()
    
def graficara(data,r,title):
    u_piese = enlistar(data[0],data[2],data[3])[2][0] #largo 2000
    u_rect = enlistar(data[0],data[2],data[3])[2][1] #largo 1999
    u_newman = enlistar(data[0],data[2],data[3])[2][2] #largo 2000
    plt.plot(data[0],u_piese,"-")
    plt.plot(data[1],u_rect,"--")
    plt.plot(data[0],u_newman,":")
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s^2]")
    #plt.yticks(r[4],r[5])
    plt.legend(["Piese Wise Linear", "Duhamel Rect", "Newmark"])
    plt.xlim(r)
    plt.title(title)
    plt.show()

def enlistar(t,a,d):
    u = []
    v = []
    ac = []
    for i in range(3):
        u1 = piece_wise_linear(t,a,d)[i]#2000 2000
        u2 = rectangular(t,a,d)[i]#2000 2000
        u3 = newman(t,a,d)[i]#2000 2000
        if i == 0:
            u.append(u1)
            u.append(u2)
            u.append(u3)
        elif i == 1:
            v.append(u1)
            v.append(u2)
            v.append(u3)
        else:
            ac.append(u1)
            ac.append(u2)
            ac.append(u3)
    return u,v,ac

def vectores_t_a(file): #retorna vectores t y a de cada terremoto
    with open(file) as f:
        data = f.read()
        
    data = data.split('\n')
    numbers = []
    time = []
    delta = 1/50
    time_counter = 0
    
    for i in range(len(data)):
        data2 = data[i].split('\tab')
        data3 = data2[0]
        if len(data3) >10:
            col1 = (float(data3[0:10]))
            col2 = (float(data3[11:20]))
            col3 = (float(data3[21:30]))
            col4 = (float(data3[31:40]))
            col5 = (float(data3[41:50]))
            col6 = (float(data3[51:60]))
            col7 = (float(data3[61:70]))
            col8 = (float(data3[71:80]))
            row = [col1,col2,col3,col4,col5,col6,col7,col8]
            numbers.append(row)
            for j in range(8):
                time_counter+=delta
                time.append(time_counter)
        else:
            col = float(data3[0:10])
            numbers.append([col])
            time_counter+=delta
            time.append(time_counter)            
    t = time
    a = []
    for i in numbers:
        for j in i:
            a.append(j)
    
    return (t,a)

def rectangular(vector_t,vector_a,properties): #retorna la integral de p(t) entre 0 y vectort[-1] por el método de Trapecio
    #print("Rectangular")    
    #variables 
    h = 1/50
    integral_values1 = [] #u(x)
    integral_values2 = [] #u(x)_p
    integral_values3 = [] #u(x)_pp
   
    
    #parámetros
    m = properties[0] #kg
    k = properties[1]*9.81 #(kgf/m)/kg --> kgf = 9.81m/s^2 *Kg ---> 4200kg * 9.81m/s^2  = x [1/s] =  hz
    c = properties[2]*9.81 #kgf -> c*9.81 m/s^2 --> kg/s
    
    #propiedades
    w = np.sqrt(k/m) #1/s
    c_cr = 2*m*w  # 2 * kg * 1/s = kg/s
    chi = c/c_cr #adimensional
    w_d = w*np.sqrt(1-chi**2) #1/s 
    
    #----------------------------------------------------------------------------------------------------------------------
    #u(t)
    y_vector1 = []
    y_vector2 = []
    A = 0
    B = 0
    A_list = []
    B_list = []
    #guardo los valores de la función f(tau) en una lista, para todos los tau entre [0,t], con t = 40 segs para los gatos
    #print(f"t = {len(vector_t)}, a = {len(vector_a)}\n")
    for i in range(len(vector_t)):
        tau = vector_t[i]
        ur = vector_a[i]/100 #m/s2
        yi1 = -np.exp(chi*w*tau)*ur* np.cos(w_d*tau)/w_d #integral de duhamel
        yi2 = -np.exp(chi*w*tau)*ur* np.sin(w_d*tau)/w_d #integral de duhamel
        y_vector1.append(yi1)
        y_vector2.append(yi2)
    #pesco cada valor de f(tau) y resuelvo la integral de duhamel para calcular A y B, aplicando el método       
    for i in range(1,len(y_vector1)):
        areai1 = y_vector1[i-1]*h
        areai2 = y_vector2[i-1]*h
        A+=areai1
        B+=areai2
        A_list.append(A)
        B_list.append(B)
    #guardo los valores de u(t) para todos los valores de t entre [0,td]
    for i in range(len(A_list)) :
        t = vector_t[i]
        u_t = A_list[i] * np.exp(-chi*w*t) * np.sin(w_d*t) - B_list[i] * np.exp(-chi*w*t) * np.cos(w_d*t)
        integral_values1.append(u_t)
    #----------------------------------------------------------------------------------------------------------------------
    #u_p(t) #vacío los valores porque la integral de duhamel es diferente
    y_vector1 = []
    y_vector2 = []
    A = 0
    B = 0
    A_list = []
    B_list = []

    #calculo los f(tau) para cada tau y los almaceno en y_vector1 y y_vector2
    for i in range(len(vector_t)):
        tau = vector_t[i]
        ur = vector_a[i]/100 #m/s2
        yi1 = np.exp(chi*w*tau)*ur* np.cos(w_d*tau) #integral de duhamel
        yi2 = np.exp(chi*w*tau)*ur* np.sin(w_d*tau) #integral de duhamel
        y_vector1.append(yi1)
        y_vector2.append(yi2)

    #pesco cada valor de f(tau) y resuelvo la integral de duhamel para calcular A y B, aplicando el método       
    for i in range(1,len(y_vector1)):
        areai1 = y_vector1[i-1]*h
        areai2 = y_vector2[i-1]*h
        A+=areai1
        B+=areai2
        A_list.append(A)
        B_list.append(B)   
        
    #guardo los valores de u_p(t) para todos los valores de t entre [0,40]
    for i in range(len(A_list)) :
        t = vector_t[i]
        I1 = A_list[i] * np.exp(-chi*w*t) * np.sin(w_d*t) - B_list[i] * np.exp(-chi*w*t) * np.cos(w_d*t)
        I2 = A_list[i] * np.exp(-chi*w*t) * np.cos(w_d*t) + B_list[i] * np.exp(-chi*w*t) * np.sin(w_d*t)
        up_t = chi/(np.sqrt(1-chi**2))*I1 - I2
        integral_values2.append(up_t)  
   
    #----------------------------------------------------------------------------------------------------------------------
    #u_pp(t) #vacío los valores porque la integral de duhamel es diferente

    for i in range(len(A_list)) :
        t = vector_t[i]
        I1 = A_list[i] * np.exp(-chi*w*t) * np.sin(w_d*t) - B_list[i] * np.exp(-chi*w*t) * np.cos(w_d*t)
        I2 = A_list[i] * np.exp(-chi*w*t) * np.cos(w_d*t) + B_list[i] * np.exp(-chi*w*t) * np.sin(w_d*t)
        upp_t = (w*(1-2*chi**2))/(np.sqrt(1-chi**2))*I1 + 2*chi*w*I2
        integral_values3.append(upp_t)  
    #print(f"u = {len(integral_values1)} , v = {len(integral_values2)}, a = {len(integral_values3)}")
   
    return integral_values1,integral_values2,integral_values3
       
def piece_wise_linear(vector_t,vector_a,properties): #retorna la integral de p(t) entre 0 y vectort[-1] por el método de Trapecio
    #print("Piese wise linear")   
    #variables 
    h = 1/50
    u_t = [0.]
    up_t = [0.]
    upp_t = [0.]
    
    #parámetros
    m = properties[0] #kg
    k = properties[1]*9.81 #(kgf/m)/kg --> kgf = 9.81m/s^2 *Kg ---> 4200kg * 9.81m/s^2  = x [1/s] =  hz
    c = properties[2]*9.81 #kgf -> c*9.81 m/s^2 --> kg/s
    
    #propiedades
    w = np.sqrt(k/m) #1/s
    c_cr = 2*m*w  # 2 * kg * 1/s = kg/s
    chi = c/c_cr #adimensional
    w_d = w*np.sqrt(1-chi**2) #1/s 
    
    sin = np.sin(w_d*h)
    cos = np.cos(w_d*h)
    e = np.exp(-chi*w*h)
    raiz = np.sqrt(1-chi**2)
    división = 2*chi/(w*h)
    
    A = e * (chi*sin/raiz+cos) #check
    B = e * (sin/w_d) #check
    C = (1/k) * (división  + e * (((1 - (2*chi**2))/(w_d*h) - chi/raiz)*sin - (1+división)*cos)) #check
    D = (1/k) * (1-división + e * ((2*chi**2-1)*sin/(w_d*h)+división*cos)) #check
    
    A1 = -e * ((w*sin)/raiz) #check
    B1 =  e * ( cos - chi*sin/raiz  ) #check
    C1 = (1/k) * (- 1/h + e*((w/raiz + chi/(h*raiz) ) * sin + cos/h)) #check 
    D1 = (1/k) * (1/h - (e/h*( chi*sin/raiz + cos   ))) #check
    
    vector_a.insert(0,0)
    
    for i in range(len(vector_a)-1):
        pi = -(vector_a[i])*m/100#pi
        pi1 = -(vector_a[i+1])*m/100 #pi+1
        
        ui = u_t[i] #u_i(t)
        vi = up_t[i] #v_i(t)
        ui1 = A*ui + B*vi + C*pi + D*pi1 #u_i+1
        upi1 = A1*ui + B1*vi + C1*pi + D1*pi1 #up_i+1 
        upp1 = (- c*upi1 - k*ui1)/m 
        
        u_t.append(ui1)
        up_t.append(upi1)
        upp_t.append(upp1)
    vector_a.pop(0)
    u_t.pop(0)
    up_t.pop(0)
    upp_t.pop(0)
    """
    print(f"u = {len(u_t)} , v = {len(up_t)}, a = {len(upp_t)}")
    print(f"t = {len(vector_t)}, a = {len(vector_a)}\n")
    """
    return u_t,up_t,upp_t
    
def newman(vector_t,vector_a,properties):
    #print("Newman")
    #variables 
    h = 1/50
    u_t = [0.]
    up_t = [0.]
    upp_t = [0.]
    ace = [0.]
    #parámetros
    m = properties[0] #kg
    k = properties[1]*9.81 #(kgf/m)/kg --> kgf = 9.81m/s^2 *Kg ---> 4200kg * 9.81m/s^2  = x [1/s] =  hz
    c = properties[2]*9.81 #kgf -> c*9.81 m/s^2 --> kg/s
    
    gamma = 0.5
    beta = 0.25
    d = (m + c*(gamma*h) + k*(beta*h**2))
    vector_a.insert(0,0)
    for i in range(len(vector_a)-1):
        pi1 = -(vector_a[i+1])*m/100 #pi+1
        ui = u_t[i] #u_i(t)
        vi = up_t[i] #v_i(t)
        ai = upp_t[i] #a_i(t)
        
        n = pi1 - c*(vi + (1-gamma)*h*ai) - k*(ui + h*vi + (((0.5-beta)*h**2)*ai))
        upp1 = n/d 
        upi1 = vi + ((1-gamma)*h)*ai + (gamma*h)*upp1
        ui1 = ui + h*vi + (((0.5-beta)*h**2)*ai) + (beta*h**2)*upp1
        
        a = upp1 +vector_a[i+1]/100
        u_t.append(ui1)
        up_t.append(upi1)
        upp_t.append(upp1)
        ace.append(a)
    
    u_t.pop(0)
    up_t.pop(0)
    upp_t.pop(0)
    ace.pop(0)
    vector_a.pop(0)
    """
    print(f"u = {len(u_t)} , v = {len(up_t)}, a = {len(upp_t)}")
    print(f"t = {len(vector_t)}, a = {len(vector_a)}\n")
    """
    return u_t,up_t,ace

def spectral_response(vector_t,vector_a,w,chi): #retorna la integral de p(t) entre 0 y vectort[-1] por el método de Trapecio
    #print("Rectangular")    
    #variables
    #print("Newman")
    #variables 
    h = 1/50
    Sd = []
    Sv = []
    Sa = []
    Spv = []
    Spa = []
    #parámetros
    m = 1 #kg
    gamma = 0.5
    beta = 0.25
    for i in range(len(w)):
        u_t = [0.]
        v_t = [0.]
        a_t = [0.]
        ace = [0.]

        wi = w[i]
        cr = 2*m*w[i]
        c = cr*chi
        k = wi**2
        d = (m + c*(gamma*h) + k*(beta*h**2))
        vector_a.insert(0,0)
        for i in range(len(vector_a)-1):
            pi1 = -(vector_a[i+1])*m/100 #pi+1
            ui = u_t[i] #u_i(t)
            vi = v_t[i] #v_i(t)
            ai = a_t[i] #a_i(t)
            
            n = pi1 - c*(vi + (1-gamma)*h*ai) - k*(ui + h*vi + (((0.5-beta)*h**2)*ai))
            upp1 = n/d 
            upi1 = vi + ((1-gamma)*h)*ai + (gamma*h)*upp1
            ui1 = ui + h*vi + (((0.5-beta)*h**2)*ai) + (beta*h**2)*upp1
            
            a = upp1 +vector_a[i+1]/100
            u_t.append(ui1)
            v_t.append(upi1)
            a_t.append(upp1)
            ace.append(a)
        
        u_t.pop(0)
        v_t.pop(0)
        a_t.pop(0)
        ace.pop(0)
        vector_a.pop(0)
        Sdi = max(max(u_t),abs(min(u_t)))
        Svi = max(max(v_t),abs(min(v_t)))
        Sai = max(max(ace),abs(min(ace)))
        Spvi = wi*Sdi
        Spai = wi**2 * Sdi
        Sd.append(Sdi)
        Sv.append(Svi)
        Sa.append(Sai)
        Spv.append(Spvi)
        Spa.append(Spai)
        
    return Sd,Sv,Sa,Spv,Spa

def PGA(vector_a,nombre):
    pga = max(max(vector_a),abs(min(vector_a)))/100
    print (f"PGA de {nombre} = {pga}\n")
    return pga

def newman2(vector_t,vector_a,properties,sistema):
    #print("Newman")
    #variables 
    h = 1/50
    u_t = [0.]
    v_t = [0.]
    a_t = [0.]
    ace = [0.]
    #parámetros
    m = properties[0] #kg
    k = properties[1]*9.81 #(kgf/m)/kg --> kgf = 9.81m/s^2 *Kg ---> 4200kg * 9.81m/s^2  = x [1/s] =  hz
    c = properties[2]*9.81 #kgf -> c*9.81 m/s^2 --> kg/s
    
    #propiedades
    w = np.sqrt(k/m) #1/s
    c_cr = 2*m*w  # 2 * kg * 1/s = kg/s
    chi = c/c_cr #adimensional
    #si chi cambia a 0.05, entonce c ahora es c_cr*chi
    gamma = 0.5
    beta = 0.25
    d = (m + c*(gamma*h) + k*(beta*h**2))
    vector_a.insert(0,0)
    for i in range(len(vector_a)-1):
        pi1 = -(vector_a[i+1])*m/100 #pi+1
        ui = u_t[i] #u_i(t)
        vi = v_t[i] #v_i(t)
        ai = a_t[i] #a_i(t)
        
        n = pi1 - c*(vi + (1-gamma)*h*ai) - k*(ui + h*vi + (((0.5-beta)*h**2)*ai))
        upp1 = n/d 
        upi1 = vi + ((1-gamma)*h)*ai + (gamma*h)*upp1
        ui1 = ui + h*vi + (((0.5-beta)*h**2)*ai) + (beta*h**2)*upp1
        
        a = upp1 +vector_a[i+1]/100
        u_t.append(ui1)
        v_t.append(upi1)
        a_t.append(upp1)
        ace.append(a)
    
    u_t.pop(0)
    v_t.pop(0)
    a_t.pop(0)
    ace.pop(0)
    vector_a.pop(0)
    Sdi = max(max(u_t),abs(min(u_t)))
    Svi = max(max(v_t),abs(min(v_t)))
    Sai = max(max(ace),abs(min(ace)))
    Spvi = w*Sdi
    Spai = w**2 * Sdi
    
    
    print(f"Para chi = {chi*100}%, el sistema {sistema}\n Sd = {Sdi}\n Sv = {Svi}\n Sa = {Sai}\n Spv = {Spvi}\n Spa = {Spai}\n")

def newman3(vector_t,vector_a,properties,sistema):
    #print("Newman")
    #variables 
    h = 1/50
    u_t = [0.]
    v_t = [0.]
    a_t = [0.]
    ace = [0.]
    
    #parámetros
    m = properties[0] #kg
    k = properties[1]*9.81 #(kgf/m)/kg --> kgf = 9.81m/s^2 *Kg ---> 4200kg * 9.81m/s^2  = x [1/s] =  hz
    
    #propiedades
    w = np.sqrt(k/m) #1/s
    c_cr = 2*m*w  # 2 * kg * 1/s = kg/s
    chi = 0.05 #adimensional
    c = c_cr*chi
    #si chi cambia a 0.05, entonce c ahora es c_cr*chi
    gamma = 0.5
    beta = 0.25
    d = (m + c*(gamma*h) + k*(beta*h**2))
    vector_a.insert(0,0)
    for i in range(len(vector_a)-1):
        pi1 = -(vector_a[i+1])*m/100 #pi+1
        ui = u_t[i] #u_i(t)
        vi = v_t[i] #v_i(t)
        ai = a_t[i] #a_i(t)
        
        n = pi1 - c*(vi + (1-gamma)*h*ai) - k*(ui + h*vi + (((0.5-beta)*h**2)*ai))
        upp1 = n/d 
        upi1 = vi + ((1-gamma)*h)*ai + (gamma*h)*upp1
        ui1 = ui + h*vi + (((0.5-beta)*h**2)*ai) + (beta*h**2)*upp1
        
        a = upp1 +vector_a[i+1]/100
        u_t.append(ui1)
        v_t.append(upi1)
        a_t.append(upp1)
        ace.append(a)
    
    u_t.pop(0)
    v_t.pop(0)
    a_t.pop(0)
    ace.pop(0)
    vector_a.pop(0)
    Sdi = max(max(u_t),abs(min(u_t)))
    Svi = max(max(v_t),abs(min(v_t)))
    Sai = max(max(ace),abs(min(ace)))
    Spvi = w*Sdi
    Spai = w**2 * Sdi
    
    print(f"Para chi = 5%, el sistema {sistema}\n Sd = {Sdi}\n Sv = {Svi}\n Sa = {Sai}\n Spv = {Spvi}\n Spa = {Spai}\n")
    

   
