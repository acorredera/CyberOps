import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class MainServiceService {
  personaSlected;
  nameUser;
  userEmail;
  porcentajePersonal: number;
  colorMediaPersonal;
  porcentajeDataCenter: number;

  temperatura;
  humedad;
  ruido;
  constructor(public http: HttpClient, private router: Router) {

  }
  // getEmotions(inicio, fin){
  //   return this.http.get("http://10.40.38.26:3000/getEmotionEmployee/EmotionEmployee?alias=Marta&initial_time="+inicio+"&final_time="+fin);
  // }
  getEmotions(inicio, fin, personaSlected){
    return this.http.get("http://10.40.38.26:3000/getEmotionEmployee/EmotionEmployee?alias="+personaSlected+"&initial_time="+inicio+"&final_time="+fin);
  }
  getHR(inicio, fin, personaSlected){
    return this.http.get("http://10.40.38.26:3000/getHREmployee/HREmployee?alias="+personaSlected+"&initial_time="+inicio+"&final_time="+fin);
  }
  getDataPersona(nombre) {
    console.log("Cogiendo datos de" + nombre);
    return this.http.get("http://10.40.38.26:3000/getStatusEmployee/stateEmployee?alias="+nombre);
  }

  getArousal(inicio, fin, personaSlected){
    return this.http.get("http://10.40.38.26:3000/getArousalEmployee/ArousalEmployee?alias="+personaSlected+"&initial_time="+inicio+"&final_time="+fin);
  }

  getDataSala() {
    console.log("Cogiendoo datos temperatura y ruido de la sala...");
    return this.http.get('http://10.40.38.26:3000/getOperatorsRoomSensorsData');
  }
  getPersonal() {
    console.log("Cogiendo lista de personal");
    return this.http.get('http://10.40.38.26:3000/getSummaryStatusEmployees');
  }
  getStatusRacks(){
    console.log("Cogiendo estado racks");
    return this.http.get('http://10.40.38.26:3000/getStatusRacks');
  }
  getStatusServersOfRacks(){
    console.log("Cogiendo estado servidores del rack");
    return this.http.get('http://10.40.38.26:3000/getStatusServersOfRack/rack?rackId=1&roomId=1');
  }
  // getDataPersona(nombre) {
  //   console.log("Cogiendo datos de" + nombre);
  //   return this.http.get("http://10.40.38.26:3000/getStatusEmployee/stateEmployee?alias=" + nombre);
  // }
 
  logOut() {
    // navigator['app'].exitApp();
    // window.navigator['app'].exitApp();
    //this.nameUser = undefined;
    this.userEmail = undefined;
    this.porcentajePersonal = undefined;
    this.nameUser=undefined;
    this.porcentajeDataCenter = undefined;

    this.temperatura = undefined;
    this.humedad = undefined;
    this.ruido = undefined;

    this.router.navigateByUrl("/login");
  }

 
 
}
