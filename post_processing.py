import json
import matplotlib.pyplot as plt 

with open("Auswertung", "r") as fp:   # Unpickling
   load_dict = (json.load(fp))



for keys,values in load_dict.items():
    print(keys,":",values[-1]) 


plt.xlabel(r'$dt$')
plt.plot(load_dict.get("time_steps"), load_dict.get("skewness"),label =r'$ \epsilon_{12} = T_1 + T_2 $')
plt.plot(load_dict.get("time_steps"), load_dict.get("average_Temperature_p1"),label =r'$ T_{avg.} $')
plt.plot(load_dict.get("time_steps"), load_dict.get("average_velocity_p1"),label =r'$ v_{avg.} $')
plt.plot(load_dict.get("time_steps"), load_dict.get("average_pressure_diff_14"),label =r'$ \Delta p_{14}$')
plt.plot(load_dict.get("time_steps"), load_dict.get("average_pressure_diff_35"),label =r'$ \Delta p_{35}$')
plt.plot(load_dict.get("time_steps"), load_dict.get("average_pressure_diff_51"),label =r'$ \Delta p_{51}$')
plt.legend()
plt.show()



