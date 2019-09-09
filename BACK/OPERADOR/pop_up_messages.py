# NEEDED TO INSTALL tkinter WITH THE COMMAND FOR PY3: apt-get install python3-tk
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tkinter as tk
import NewsCore.dao.EmployeeDAOImpl as dao
import datetime
import time
import random
import settings

flag=1

daoStatus = dao.EmployeeDAOImpl()
daoStatus.createsession(settings.ip_DCOS_cassandra)
daoStatus.setlogger()
daoStatus.loadkeyspace(settings.keyspace_cassandra)
PATH = os.getcwd()+"/OPERADOR"

def pop_up(image, text):
    root = tk.Tk()
    kill_time = 60000 #time in ms: 5 s = 5000 ms
    root.after(kill_time, lambda: root.destroy())
    root.wm_title("SUGGESTION")
    logo = tk.PhotoImage(file=os.path.join(PATH,image))

    w1 = tk.Label(root, image=logo).pack(side="left")

    explanation = text

    w2 = tk.Label(root,
                  justify=tk.LEFT,
                  padx = 10,
                  text=explanation,
                  font=("Verdana", 11)).pack(side="right")
    # root.geometry("500x500")  # You want the size of the app to be 500x500
    # root.resizable(0, 0)

    root.mainloop()

def suggestion(emotion,hr):
    hr_normal=range(60,81)

    if emotion==0 and hr==0:
        image, text="", ""

    elif (emotion=="happy" or emotion=="neutral") or (hr in hr_normal): #Estado IDEAL

        list_image=["pc_staff_pop_up.gif", "welldone_pop_up.gif"]
        list_text=["""Por favor, cuando puedas, comprueba
         el estado de las máquinas""",
                   """Parece que está todo en orden, por favor,
 comprueba el estado de las máquinas
  para asegurarte de que todo siga así"""]

        secure_random = random.SystemRandom()
        image = secure_random.choice(list_image)

        if image == list_image[0]:
            text = list_text[0]
        else:
            text = list_text[1]

    elif (emotion=="anger" or emotion=="surprise" or emotion=="sadness") or (hr not in hr_normal): #Estado ANORMAL

        list_image = ["clock_pop_up.gif", "coffee_pop_up.gif"]
        list_text = ["""Por favor, tómate un descanso de
                     5 minutos cuando puedas""",
                     """Por favor, ve a tomarte un café o
                     alguna bebida y descansa 5 minutos"""]

        secure_random = random.SystemRandom()
        image = secure_random.choice(list_image)

        if image==list_image[0]:
            text=list_text[0]
        else:
            text=list_text[1]

    return image, text

while(True):
    if flag==1:

        current_time = datetime.datetime.now() #settings.current_time#
        currentTime = current_time - datetime.timedelta(seconds=current_time.second, microseconds=current_time.microsecond)
        # print(currentTime)
        someMinutesAgo = currentTime - datetime.timedelta(minutes=settings.minutes_ago_30)
        # print(someMinutesAgo)

        emotion = daoStatus.select_dominant_emotion(settings.employee_name,someMinutesAgo,currentTime)
        hr = daoStatus.select_hr_inRange(settings.employee_name, someMinutesAgo, currentTime)
        flag=0

        message=suggestion(emotion,hr)
        if(message[0]=="" or message[1]==""):
            continue
        else:
            pop_up(message[0], message[1])

    else:
        time.sleep(900) #30 min = 1800 s

        current_time = datetime.datetime.now() #settings.current_time#
        currentTime = current_time - datetime.timedelta(seconds=current_time.second, microseconds=current_time.microsecond)
        someMinutesAgo = currentTime - datetime.timedelta(minutes=settings.minutes_ago_30)

        emotion = daoStatus.select_dominant_emotion(settings.employee_name, someMinutesAgo, currentTime)
        hr = daoStatus.select_hr_inRange(settings.employee_name, someMinutesAgo, currentTime)

        message = suggestion(emotion, hr)
        if (message[0] == "" or message[1] == ""):
            continue
        else:
            pop_up(message[0], message[1])
