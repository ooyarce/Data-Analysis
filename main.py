from functions import *

#-----------------------------------------------------------------------------
#--------------------------------LOS GATOS-------------------------------------
#-----------------------------------------------------------------------------
#vectores tiempo
time = vectores_t_a("Los Gatos.V2")[0]
time2 = vectores_t_a("Los Gatos.V2")[0]
time2.pop(-1)
a1 = vectores_t_a("Los Gatos.V2")[1]
graficar(time,a1,[0,40],"Loma Prieta 1989 - Los Gatos")

#------------------------------------------------------------------------------
#--------------------------------TARZANA---------------------------------------
#------------------------------------------------------------------------------
#vectores tiempo
time3 = vectores_t_a("Tarzana.V2")[0]
time4 = vectores_t_a("Tarzana.V2")[0]
time4.pop(-1)
a2 = vectores_t_a("Tarzana.V2")[1] 
graficar(time3,a2,[0,60],"Northridge 1994 - Estación Tarzana")
graficar2(time,time3,a1,a2)

#-----------------------------------------------------------------------------
#----------------------------GRAFICANDO MÉTODOS--------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#--------------------------SISTEMA 1 - LOS GATOS-------------------------------
#-----------------------------------------------------------------------------
d = [1500.0,4200.0,50.0] #parámetros sistema 1
data_sistema = [time,time2,a1,d]

#vector de datos para graficar
#data_plot_G1 = [u_val,u_val_ticks,up_val,up_val_ticks,upp_val,upp_val_ticks]
graficaru(data_sistema,[0,40],"System 1 - Los Gatos")
graficarv(data_sistema,[0,40],"System 1 - Los Gatos")
graficara(data_sistema,[0,40],"System 1 - Los Gatos")

#-----------------------------------------------------------------------------
#--------------------------SISTEMA 2 - LOS GATOS-------------------------------
#-----------------------------------------------------------------------------
d2 = [130.0,4200.0,15.0]  #parámetros sistema 2
data_sistema = [time,time2,a1,d2]

#vector de datos para graficar
#data_plot_G1 = [u_val,u_val_ticks,up_val,up_val_ticks,upp_val,upp_val_ticks]
graficaru(data_sistema,[0,40],"System 2 - Los Gatos")
graficarv(data_sistema,[0,40],"System 2 - Los Gatos")
graficara(data_sistema,[0,40],"System 2 - Los Gatos")

#-----------------------------------------------------------------------------
#--------------------------SISTEMA 1 - TARZANA-------------------------------
#-----------------------------------------------------------------------------
d = [1500.0,4200.0,50.0] #parámetros sistema 1
data_sistema = [time3,time4,a2,d]

#vector de datos para graficar
#data_plot_G1 = [u_val,u_val_ticks,up_val,up_val_ticks,upp_val,upp_val_ticks]
graficaru(data_sistema,[0,60],"System 1 - Tarzana")
graficarv(data_sistema,[0,60],"System 1 - Tarzana")
graficara(data_sistema,[0,60],"System 1 - Tarzana")

#-----------------------------------------------------------------------------
#--------------------------SISTEMA 2 - TARZANA-------------------------------
#-----------------------------------------------------------------------------
d2 = [130.0,4200.0,15.0]  #parámetros sistema 2
data_sistema = [time3,time4,a2,d2]

#vector de datos para graficar
#data_plot_G1 = [u_val,u_val_ticks,up_val,up_val_ticks,upp_val,upp_val_ticks]
graficaru(data_sistema,[0,60],"System 2 - Tarzana")
graficarv(data_sistema,[0,60],"System 2 - Tarzana")
graficara(data_sistema,[0,60],"System 2 - Tarzana")
#-----------------------------------------------------------------------------
#---------------------------------ESPECTROS----------------------------------
#-----------------------------------------------------------------------------
T = np.linspace(0.01,3,100)#vector de períodos entre (0,3]
w = [] #vector de frecuencias
for t in T:
     wi = 2*np.pi/t
     w.append(wi)

#graficando Sd,Sv,Sv,Spv y Spa
data1 = (time,time2,a1,a2,w,T,0.03,0.05) #parámetros de funciones
graficar_sd(data1)
graficar_sd2(data1)

graficar_sv(data1)
graficar_sv2(data1)

graficar_sa(data1)
graficar_sa2(data1)


PGA(a1,"Los Gatos")
PGA(a2,"Tarzana")

newman2(time,a1,d,1)
newman2(time,a1,d2,2)

newman2(time3,a2,d,1)
newman2(time3,a2,d2,2)
newman3(time,a1,d,1)
newman3(time,a1,d2,2)

newman3(time3,a2,d,1)
newman3(time3,a2,d2,2)
