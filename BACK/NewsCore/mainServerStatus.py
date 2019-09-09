import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import NewsCore.model.ServerStatus as status
import NewsCore.dao.ServersStatusDAOImpl as dao
import datetime
import NewsCore.model.ServerStatus as serverStatus

if __name__ == "__main__":

    daoServerStatus = dao.ServersStatusDAOImpl()
    daoServerStatus.createsession('localhost')
    daoServerStatus.setlogger()
    daoServerStatus.loadkeyspace('mydb')
    daoServerStatus.create_table()

    #creation of AAdata
    # currentTime = datetime.datetime.now()
    # log1 = status.ServerStatus(rack_id="1", metric_id=1, server_id="groot",date=currentTime,
    #                            server_component="Core4", profile=serverStatus.statusProfile.clock, value=1226, punctuation=0.3)
    # log2 = status.ServerStatus(rack_id="1", metric_id=1, server_id="groot", date=currentTime+datetime.timedelta(seconds=60), server_component="Core4",
    #                            profile=serverStatus.statusProfile.temperature, value=30, punctuation=0.4)
    #
    # log3 = status.ServerStatus(rack_id="1", metric_id=1, server_id="groot", date=currentTime, server_component="Core3",
    #                            profile=serverStatus.statusProfile.clock, value=1223, punctuation=0.2)
    # log4 = status.ServerStatus(rack_id="2", metric_id=1, server_id="hulk2",
    #                            date=currentTime + datetime.timedelta(seconds=60), server_component="Core4",
    #                            profile=serverStatus.statusProfile.temperature, value=20, punctuation=0.1)
    # log5 = status.ServerStatus(rack_id="1", metric_id=1, server_id="groot", date=currentTime, server_component="Core5",
    #                             profile=serverStatus.statusProfile.temperature, value=12, punctuation=0.2)
    # log6 = status.ServerStatus(rack_id="1", metric_id=1, server_id="groot2", date=currentTime, server_component="Core5",
    #                            profile=serverStatus.statusProfile.temperature, value=12, punctuation=0.2)
    # log7 = status.ServerStatus(rack_id="1", metric_id=1, server_id="groot2", date=currentTime, server_component="Core5",
    #                            profile=serverStatus.statusProfile.clock, value=12, punctuation=0.4)

    # daoServerStatus.insert_metric(log1)
    # daoServerStatus.insert_metric(log2)
    # daoServerStatus.insert_metric(log3)
    # daoServerStatus.insert_metric(log4)
    # daoServerStatus.insert_metric(log5)
    # daoServerStatus.insert_metric(log6)
    # daoServerStatus.insert_metric(log7)


    someMinutesAgo = currentTime - datetime.timedelta(minutes = 50)
    print(currentTime)
    print(someMinutesAgo)
    # print("temperature: ", daoServerStatus.select_temperature_inRange('1', someMinutesAgo, currentTime))
    # print("noise level: ", daoServerStatus.select_noise_level_inRange

    # racks = daoServerStatus.select_rack_ids()
    # servers = daoServerStatus.select_server_ids_by_rack_id("1")
    # #daoServerStatus.select_avg_by_date_id("1", server_id, profile, initial_timestamp,end_timestamp)
    # #datos = daoServerStatus.select_avg_by_date_id("1", "groot", "clock", someMinutesAgo, currentTime)
    # #datos = daoServerStatus.select_stateRack_by_rack_servers_date_profile("1", servers, someMinutesAgo, currentTime)
    # #datos2 = daoServerStatus.select_stateServers_by_rack_servers_date_profile("1", servers, someMinutesAgo, currentTime)
    # datos3 = daoServerStatus.select_stateServer_by_rack_id_server_date("1", 'groot', someMinutesAgo, currentTime)
    # print(datos3)



