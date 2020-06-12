# PROFILES
# temperature = 1 #celsius
# utilization = 2 #%
# power = 3 #W
# voltage = 4 #V
# clock = 5
# fanspeed = 6 #RPM
# Null = 7

# VALUES
# Temperature
T_ideal = 25
T_maximum = 50

# Utilization
U_ideal = 10
U_maximum = 50

# Power
P_ideal = 10
P_maximum = 30

# Voltage
V_ideal = 10
V_maximum = 20

# Clock
C_ideal = 1000
C_maximum = 1500

# Fanspeed
F_ideal = 8000
F_maximum =  10000

# Null

def values (profile, value):

    def punctuation(value, ideal, maximum):
        if value <= ideal:
            punctuation_value = 1
        else:
            punctuation_value = (maximum - value) / (maximum - ideal)
        return punctuation_value

    if profile==1:
        # Temperature
        ideal = T_ideal
        maximum = T_maximum
        status_value=punctuation(value, ideal, maximum)
    elif profile == 2:
        # Utilization
        ideal = U_ideal
        maximum = U_maximum
        status_value = punctuation(value, ideal, maximum)
    elif profile == 3:
        # Power
        ideal = P_ideal
        maximum = P_maximum
        status_value = punctuation(value, ideal, maximum)
    elif profile == 4:
        # Voltage
        ideal = V_ideal
        maximum = V_maximum
        status_value = punctuation(value, ideal, maximum)
    elif profile == 5:
        # Clock
        ideal = C_ideal
        maximum = C_maximum
        status_value = punctuation(value, ideal, maximum)
    elif profile == 6:
        # Fanspeed
        ideal = F_ideal
        maximum = F_maximum
        status_value = punctuation(value, ideal, maximum)
    else:
        # Null
        # pass
        status_value=""

    return status_value


a=values(1,37.5)
print(a)