import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class MainServiceService {

  ip;
  

  personaSlected;
  nameUser;
  userEmail;
  porcentajePersonal: number;
  colorMediaPersonal;
  porcentajeDataCenter: number;

  temperatura;
  humedad;
  ruido;

  rackIdSelected: number;
  roomIdSelected: number;

  constructor(public http: HttpClient, private router: Router) {
    this.ip="10.40.38.26:3000";
  }
  // getEmotions(inicio, fin){
  //   return this.http.get("http://10.40.38.26:3000/getEmotionEmployee/EmotionEmployee?alias=Marta&initial_time="+inicio+"&final_time="+fin);
  // }
  getEmotions(inicio, fin, personaSlected){
    return this.http.get("http://"+this.ip+"/getEmotionEmployee/EmotionEmployee?alias="+personaSlected+"&initial_time="+inicio+"&final_time="+fin);
  }
  getHR(inicio, fin, personaSlected){
    return this.http.get("http://"+this.ip+"/getHREmployee/HREmployee?alias="+personaSlected+"&initial_time="+inicio+"&final_time="+fin);
  }
  getDataPersona(nombre) {
    console.log("Cogiendo datos de" + nombre);
    return this.http.get("http://"+this.ip+"/getStatusEmployee/stateEmployee?alias="+nombre);
  }

  getStatusDC(){
    return this.http.get("http://"+this.ip+"/getStatusDC");
  }

  getArousal(inicio, fin, personaSlected){
    return this.http.get("http://"+this.ip+"/getArousalEmployee/ArousalEmployee?alias="+personaSlected+"&initial_time="+inicio+"&final_time="+fin);
  }

  getDataSala() {
    console.log("Cogiendoo datos temperatura y ruido de la sala...");
    return this.http.get("http://"+this.ip+"/getOperatorsRoomSensorsData");
  }
  getPersonal() {
    console.log("Cogiendo lista de personal");
    return this.http.get("http://"+this.ip+"/getSummaryStatusEmployees");
  }
  getStatusRacks(){
    console.log("Cogiendo estado racks");
    return this.http.get("http://"+this.ip+"/getStatusRacks");
  }
  getStatusServersOfRacks(){
    this.rackIdSelected=JSON.parse(localStorage.getItem('rackIdSelected'));
    this.roomIdSelected=JSON.parse(localStorage.getItem('roomIdSelected'));
    console.log("Cogiendo estado servidores del rack");
    return this.http.get("http://"+this.ip+"/getStatusServersOfRack/rack?rackId="+this.rackIdSelected+"&roomId="+this.roomIdSelected);
  }
  // getDataPersona(nombre) {
  //   console.log("Cogiendo datos de" + nombre);
  //   return this.http.get("http://10.40.38.26:3000/getStatusEmployee/stateEmployee?alias=" + nombre);
  // }
 
  logOut() {
    
    //this.nameUser = undefined;
    // window.sessionStorage.clear();
    window.localStorage.clear();

    this.router.navigateByUrl("/login");
  } 
}
