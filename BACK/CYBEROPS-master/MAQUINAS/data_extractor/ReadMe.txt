Configuration files in: https://github.com/cristinalunaj/CYBEROPS_config
Run: python parser-installer       in the machine from whihc we want to obtain the data

Intervalo de envío de datos: 10s
Topic en kafka: cyberops_machines

Campos por métrica en JSON mandado a kafka
	host: nombre del equipo
	type: tipo de métrica
	plugin: nombre de la métrica
	value: valor numérico de la métrica

Número de métricas (Ejemplo para gamora: 10.40.39.30, podría ser diferente en otros)
	Potencia[W]: 2 (una por socket)
	Temperatura[ºC]: 12 + 1 (12 cores y una memoria)
	Voltaje[V]: 2 (una por socket)
	Ventiladores[RPM]: 8 (5 inlet +  3 outlet)
	Frecuencia[MHz]: 24 (dos por core, una por core virutal)
	Utilización[%]: 48 (cuatro por core, dos por core virtual: user y nice)
	Total: 97 métricas

Paquetes (usuario root):
	yum -y install epel-release 
	yum -y group install "Development Tools"
	yum -y install collectd collectd-write_kafka python-pip ipmitool lm_sensors systat wget
	pip install collectd (kafka-python para pruebas)
	sensors-detect --auto
	Rapl:
		mkdir /root/rapl-read
		cd /root/rapl-read
		wget https://raw.githubusercontent.com/deater/uarch-configure/master/rapl-read/Makefile
		wget https://raw.githubusercontent.com/deater/uarch-configure/master/rapl-read/rapl-plot.c
		wget https://raw.githubusercontent.com/deater/uarch-configure/master/rapl-read/rapl-read.c
		https://raw.githubusercontent.com/deater/uarch-configure/master/rapl-read/README
		make
	Parser(cambiar repositorio en el futuro):
		cd /root
		wget https://raw.githubusercontent.com/spmorillo/Cargas/master/parser.py
	Collectd config:
		wget https://raw.githubusercontent.com/spmorillo/Cargas/master/collectd.conf
		yes | cp -rf /root/collectd.conf /etc/collectd.conf
		wget https://raw.githubusercontent.com/spmorillo/Cargas/master/types.db
		yes | cp -rf /root/types.db /usr/share/collectd/types.db
	Borrar archivos en root:
		rm -rf types.db collectd.conf

	EN CASO DE QUE LOS SERVIDORES SE CAIGAN SÓLO HAY QUE VOLVER A CORRER EL COMANDO DE DEBAJO PARA QUE VUELVAN A ENVIAR DATOS:
	    systemctl start collectd


Comandos
	Potencia: /root/rapl-read/rapl-read
	Temperatura: sensors
	Temperatura memoria, voltaje, ventiladores: ipmitool sdr
	Frecuencia: mpstat -P ALL
	Utilización: cat /proc/cpuinfo