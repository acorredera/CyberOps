import { Component, OnInit, Input, EventEmitter, Output } from '@angular/core';
import { MenuController } from '@ionic/angular';
import { Router } from '@angular/router';
import { MainServiceService } from '../main-service.service';
import { AppComponent } from '../app.component';

@Component({
  selector: 'app-login',
  templateUrl: './login.page.html',
  styleUrls: ['./login.page.scss'],
})
export class LoginPage {

  @Output() public paraPadre = new EventEmitter<string>();
  password;
  userEmail: string;
  errorAuthentication: boolean;

  dataPersonal;
  mediaPersonal;
  colorMediaPersonal;
  // dataSala;
  
  constructor(public menuCtrl: MenuController, private router: Router, public mainService: MainServiceService,
    private appComponent: AppComponent) {
   

  }
  loadDataPersonel(){
    this.mainService.getPersonal().subscribe(
      data=>{
        this.dataPersonal=data;
        console.log("ESTE ES EL DATA PERSONEL");
        console.log(this.dataPersonal);
      
        this.appComponent.porcentajePersonal=Math.round((94+3+60+94+3+90+90+37+5+94+3+90+37+7+this.dataPersonal[0]["profile"]*100+
        this.dataPersonal[1]["profile"]*100+this.dataPersonal[2]["profile"]*100)/18);
       
        localStorage.setItem('mediaPersonal', JSON.stringify(this.appComponent.porcentajePersonal));
        // localStorage.setItem('userName', this.appComponent.nameUser);

        if(this.appComponent.porcentajePersonal<=40){
          this.appComponent.colorMediaPersonal="red";
          
          localStorage.setItem('colorMediaPersonal', JSON.stringify(this.appComponent.colorMediaPersonal));
        }
        if(this.appComponent.porcentajePersonal>40 && this.mediaPersonal<70){
          this.appComponent.colorMediaPersonal="orange";

          localStorage.setItem('colorMediaPersonal', JSON.stringify(this.appComponent.colorMediaPersonal));
        }
        if(this.appComponent.porcentajePersonal>=70){
          this.appComponent.colorMediaPersonal="green";
         
          localStorage.setItem('colorMediaPersonal', JSON.stringify(this.appComponent.colorMediaPersonal));
        }

      }
    )
  }
  ionViewWillEnter() {
    this.errorAuthentication = false;
    this.menuCtrl.enable(false);
  }

  goToMedioAmbiente() {//Aquí cargaré los datos de porcentajePersonal, porcentajeDtaCenter, Temperatura, humedad y ruido.
    if (this.userEmail == "eva@prueba.com" && this.password == "12345") {
      this.appComponent.nameUser = "Eva García";
      this.appComponent.userEmail = this.userEmail;
      this.loadDataPersonel();
      this.router.navigateByUrl('/medio-ambiente');
      return;

    }
    if (this.userEmail == "sonia@prueba.com" && this.password == "678910") {
      this.appComponent.nameUser = "Sonia Escolano";
      this.appComponent.userEmail = this.userEmail;
      this.loadDataPersonel();
      this.router.navigateByUrl('/medio-ambiente');
      return;
    }
    else {
      this.errorAuthentication = true;
    }
  }
  // goToMedioAmbiente() {//Aquí cargaré los datos de porcentajePersonal, porcentajeDtaCenter, Temperatura, humedad y ruido.
  //   if (this.userEmail == "eva@prueba.com" && this.password == "12345") {
  //     this.mainService.getDataSala().subscribe(
  //       data=>{
  //         this.dataSala=data;
  //         this.appComponent.temperatura=Math.round(this.dataSala[0]["temperature"]);
  //         this.appComponent.ruido=Math.round(this.dataSala[0]["noise"]);
  //         this.appComponent.humedad=Math.round(this.dataSala[0]["humidity"]*100)/100;
  //         this.appComponent.nameUser = "Eva Garcia";
  //         this.appComponent.userEmail = this.userEmail;
  //         this.router.navigateByUrl('/medio-ambiente');
  //       }
  //     )
  //   }
  //   if (this.userEmail == "sonia@prueba.com" && this.password == "678910") {
  //     this.mainService.getDataSala().subscribe(
  //       data=>{
  //         this.dataSala=data;
  //         this.appComponent.temperatura=Math.round(this.dataSala[0]["temperature"]);
  //         this.appComponent.ruido=Math.round(this.dataSala[0]["noise"]);
  //         this.appComponent.humedad=Math.round(this.dataSala[0]["humidity"]*100)/100;
  //         this.appComponent.nameUser = "Sonia Escolano";
  //         this.appComponent.userEmail = this.userEmail;
  //         this.router.navigateByUrl('/medio-ambiente');
  //       }
  //     )
  //   }
  //   else {
  //     this.errorAuthentication = true;
  //   }
  // }
}
