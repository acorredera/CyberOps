// DHT Temperature & Humidity Sensor
// Unified Sensor Library Example
// Written by Tony DiCola for Adafruit Industries
// Released under an MIT license.

// Depends on the following Arduino libraries:
// - Adafruit Unified Sensor Library: https://github.com/adafruit/Adafruit_Sensor
// - DHT Sensor Library: https://github.com/adafruit/DHT-sensor-library
#include <Arduino.h>
#include <ESP8266WiFi.h>    /*  WiFi Shield ESP8266 ESP-12f Library */
#include <Adafruit_Sensor.h>
#include <WiFiUdp.h>
#include <DHT.h>
#include <time.h>

#define DHTPIN            D4         // Pin which is connected to the DHT sensor.


// Uncomment the type of sensor in use:
//#define DHTTYPE           DHT11     // DHT 11 
#define DHTTYPE           DHT22     // DHT 22 (AM2302)
//#define DHTTYPE           DHT21     // DHT 21 (AM2301)

/**--------------------Serial WIfi Module --------------**/
const char* ssid = "GreenLSI";   /*  Network's name --const char* ssid */
const char* password = "glsi640046";    /*  Network's password --const char* password  */
WiFiUDP Udp;
// See guide for details on sensor wiring and usage:
//   https://learn.adafruit.com/dht/overview


const int timezone = +2;  /*  Timezone (Spain)   */

unsigned char direccionIp[4] = {10,40,38,30}; //IP Ordenador que actúa como servidor al que estoy enviando os datos .26(mi ordena) o .30(venus)
int value = 4;
unsigned int localUdpPort = 2004; 
char buffer[255] = "hola";


/*  Structure to store DHT sensor data  */
struct Metrics {
  float  humidity;
  float  celsius;
  float  heatIndexCels;
};


DHT dht(DHTPIN, DHTTYPE); /*  DHT sensor client  */
//WiFiClient wifiClient;    /*  WiFi Shield client */
//PubSubClient client(server, 1883, callback, wifiClient);    /*  MQTT Broker client  */

/*  Structure to store DHT sensor data  */
void setup() {
  Serial.begin(9600); 
  // Initialize device.
  dht.begin();
  // Wifi connection
  wifiConnect();
  configureTime();
  //UDP connection
  Udp.begin(localUdpPort);
}

void loop() {
  // Delay between measurements.
  
  float temperatura = 0;
  float humedad = 0;

  time_t now = time(nullptr);
  String datetime = buildDateTime(now);
  struct Metrics *metric = readSensor();
  printResults(datetime, metric);
  
  String payload = createPayload(datetime, now, metric);
  payload.toCharArray(buffer, sizeof(buffer));
  Udp.beginPacket(direccionIp, localUdpPort); 
  Udp.write(buffer);
  Udp.endPacket();
  delay(60000);
}




/*
 * Method: wifiConnect
 * ----------------------------
 *   Establish WiFi connection.
 *   Doesn't  finish until conection is stablished.
 *
 */
void wifiConnect(){
  
  //Serial.println();
  //Serial.println();
  String MAC = WiFi.macAddress();
  String dirMAC = "ESP_"+MAC.substring(0,11)+MAC.substring(12,14)+MAC.substring(15,17);
  Serial.println(MAC);
  Serial.println(dirMAC);
  Serial.print("Connecting to ");
  Serial.print(ssid);
  delay(500);
  //WiFi.mode(WIFI_AP_STA);
  WiFi.begin(ssid, password);
  int tries = 0;
  
  while (WiFi.status() != WL_CONNECTED) {
    //Serial.println("status"+WiFi.status());
    delay(500);
    Serial.print(".");
    tries += 1;

    if (tries == 30){
      tries = 0;
      Serial.println();
      Serial.println("Failed.");
      Serial.print("Trying again");
    }
  }
  
  Serial.println("");
  Serial.println("WiFi connected");  
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());  
}


/*
 * Method: configureTime
 * ----------------------------
 *   Gets date/time info for an specific timezone.
 *
 */
void configureTime(){
  configTime(timezone * 3600, 0, "pool.ntp.org", "time.nist.gov");
  Serial.print("\nWaiting for time");
  
  while (!time(nullptr)) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("");
  Serial.println("Time configured.");
  Serial.println("");
  
}

/*
 * Function: buildDateTime
 * ----------------------------
 *  Builds date/time information.
 *  
 *  now: Current date/time timestamp.
 *   
 *  returns: date/time string information in 'yyyy-MM-ddTHH:mm:ssZ' format.
 *   
 */
String buildDateTime(time_t now){
  struct tm * timeinfo;
  timeinfo = localtime(&now);
  String c_year = (String) (timeinfo->tm_year + 1900);
  String c_month = (String) (timeinfo->tm_mon + 1);
  String c_day = (String) (timeinfo->tm_mday);
  String c_hour = (String) (timeinfo->tm_hour);
  String c_min = (String) (timeinfo->tm_min);
  String c_sec = (String) (timeinfo->tm_sec);
  String datetime = c_year + "-" + c_month + "-" + c_day + "T" + c_hour + ":" + c_min + ":" + c_sec;
  
  if (timezone < 0){
    datetime += "-0" + (String)(timezone * -100);
  }else{
    datetime += "0" + (String)(timezone * 100);
  }
  return datetime;
}
/*
 * Function: readSensor
 * ----------------------------
 *  Reads data from DHT sensor.
 *  
 *  returns: DHT data of temperatures and heat indexes.
 *   
 */
struct Metrics * readSensor(){

  struct Metrics *metric = (Metrics*)malloc(sizeof(struct Metrics));
  metric->humidity = dht.readHumidity();
  metric->celsius = dht.readTemperature();  
  if (isnan(metric->humidity) || isnan(metric->celsius)) {
    Serial.println("Failed to read from DHT sensor!");
    delay(1000); 
    metric->humidity = 0;
    metric->celsius = 0;
    metric->heatIndexCels =0;
  } else {
    metric->heatIndexCels = dht.computeHeatIndex(metric->celsius, metric->humidity, false);
  }
  return metric;
  
}

/*
 * Method: printResults
 * ----------------------------
 *   Print collected data.
 *
 */
void printResults(String datetime, Metrics* metric){
  Serial.println("Lecture: "+ (String)datetime);
  Serial.println("Humidity: " + (String)metric->humidity);
  Serial.println("Temperature: " + (String)metric->celsius + " *C ");
  Serial.println("Heat index: " + (String)metric->heatIndexCels + " *C ");  
}


/*
 * Function: createPayload
 * ----------------------------
 *  Creates payload to send.
 *  
 *  datetime: String with date/time information.
 *  now: Current date/time timestamp.
 *  metric: DHT sensor data.
 *  
 *  returns: payload in JSON format.
 *   
 */
String createPayload(String datetime, time_t now, Metrics* metric){
  String payload = "{\"client\":";
  payload += "\"sensorTemp\"";
  payload += ",\"room_id\":";
  payload += "\"1\"";
  payload += ",\"date\":";
  payload += "\"" + datetime + "\"";
  payload += ",\"timestamp\":";
  payload += "\"" + (String)now + "\"";
  payload += ",\"temperature\":";
  payload += "\"" + (String)metric->celsius + "\"";
  payload += ",\"humidity\":";
  payload += "\"" + (String)metric->humidity + "\"";
  payload += ",\"heat_index\":";
  payload += "\"" + (String)metric->heatIndexCels + "\"";
  payload += "}";
  return payload;
}
