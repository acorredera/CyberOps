import { Component } from '@angular/core';

import { Platform } from '@ionic/angular';
import { SplashScreen } from '@ionic-native/splash-screen/ngx';
import { StatusBar } from '@ionic-native/status-bar/ngx';
import { MainServiceService } from './main-service.service';

import { Router } from '@angular/router';
import { THIS_EXPR } from '@angular/compiler/src/output/output_ast';

@Component({
  selector: 'app-root',
  templateUrl: 'app.component.html'
})
export class AppComponent {
  nameUser;
  userEmail;
  porcentajePersonal: number;
  colorMediaPersonal;
  porcentajeDataCenter: number;

  dataSala=[];
  temperatura;
  humedad;
  ruido;

  //Botones de navegación
  medioAmbienteActive: boolean;
  energiaActive: boolean;
  finanzasActive: boolean;
  personalActive: boolean;
  dataCenterActive: boolean;
  filtrarPalabrasActive: boolean;

  public appPages = [
    {
      title: 'Home',
      url: '/home',
      icon: 'home'
    },
    {
      title: 'Medio Ambiente',
      url: '/medio-ambiente',
    },
    {
      title: 'Energía',
      url: '/energia',
    },
    {
      title: 'Finanzas',
      url: '/finanzas',
    },
    {
      title: 'Personal',
      url: '/personal',
    },
    {
      title: 'Data Center',
      url: '/data-center',
    },
    {
      title: 'Filtrar palabras',
      url: '/filtro-palabras',
    },
  
  ];

  constructor(
    private platform: Platform,
    private splashScreen: SplashScreen,
    private statusBar: StatusBar,
    public mainService: MainServiceService,
    private router: Router
  ) {
   
    this.porcentajeDataCenter=JSON.parse(localStorage.getItem('mediaDC'));
    this.porcentajePersonal=JSON.parse(localStorage.getItem("mediaPersonal"));
    this.temperatura=JSON.parse(localStorage.getItem('temperature'));
    this.humedad=JSON.parse(localStorage.getItem('humidity'));
    this.ruido=JSON.parse(localStorage.getItem('noise'));

    //Datos almacenados en el service
    this.nameUser=this.mainService.nameUser;
    this.userEmail=this.mainService.userEmail;
    
    //Manejadir de botones
    this.medioAmbienteActive=true;
    this.energiaActive=false;
    this.finanzasActive=false;
    this.personalActive=false;
    this.dataCenterActive=false;
    this.filtrarPalabrasActive=false;
    this.initializeApp();
    
  }

  initializeApp() {
    this.platform.ready().then(() => {
      this.statusBar.styleDefault();
      this.splashScreen.hide();
    });
  }
 goToMedioAmbiente(){
  this.energiaActive=false;
  this.finanzasActive=false;
  this.personalActive=false;
  this.dataCenterActive=false;
  this.filtrarPalabrasActive=false;
  this.medioAmbienteActive=true;
  // this.router.navigateByUrl("/medio-ambiente");

   
 }
 goToEnergia(){
  this.finanzasActive=false;
  this.personalActive=false;
  this.dataCenterActive=false;
  this.filtrarPalabrasActive=false;
  this.medioAmbienteActive=false;
  this.energiaActive=true;

  
 }
 goToFinanzas(){
  this.finanzasActive=true;
  this.personalActive=false;
  this.dataCenterActive=false;
  this.filtrarPalabrasActive=false;
  this.medioAmbienteActive=false;
  this.energiaActive=false;
 }
 goToPersonal(){
  this.personalActive=true;
  this.dataCenterActive=false;
  this.filtrarPalabrasActive=false;
  this.medioAmbienteActive=false;
  this.energiaActive=false;
  this.finanzasActive=false;
 }
 goToDataCenter(){
  this.dataCenterActive=true;
  this.filtrarPalabrasActive=false;
  this.medioAmbienteActive=false;
  this.energiaActive=false;
  this.finanzasActive=false;
  this.personalActive=false;
 }
 goToFiltrarPalabras(){
  this.filtrarPalabrasActive=true;
  this.medioAmbienteActive=false;
  this.energiaActive=false;
  this.finanzasActive=false;
  this.personalActive=false;
  this.dataCenterActive=false;
 }

logOut(){
  this.mainService.logOut();
  
 
}
  
}
