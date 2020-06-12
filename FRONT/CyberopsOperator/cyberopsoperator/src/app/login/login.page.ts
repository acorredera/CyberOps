import { Component, OnInit, Input, EventEmitter, Output } from '@angular/core';
import { MenuController } from '@ionic/angular';
import { Router } from '@angular/router';
import { MainServiceService } from '../main-service.service';
import { AppComponent } from '../app.component';
import { THIS_EXPR } from '@angular/compiler/src/output/output_ast';

@Component({
  selector: 'app-login',
  templateUrl: './login.page.html',
  styleUrls: ['./login.page.scss'],
})
export class LoginPage {

  @Output() public paraPadre = new EventEmitter<string>();
  password;
  userEmail: string;
  // ipAddress: string;
  errorAuthentication: boolean;

  dataPersonal;
  mediaPersonal;
  colorMediaPersonal;
  mediaDC;

  // dataSala;

  constructor(public menuCtrl: MenuController, private router: Router, public mainService: MainServiceService,
    private appComponent: AppComponent) {
    // this.mainService.ip="10.40.38.26:3000";

  }
  ionViewWillEnter() {
    this.errorAuthentication = false;
    this.menuCtrl.enable(false);
  }
  loadDataPersonel() {

    // this.mainService.ip=this.ipAddress;
    this.mainService.getPersonal().subscribe(
      data => {
        this.dataPersonal = data;
        console.log("ESTE ES EL DATA PERSONEL");
        console.log(this.dataPersonal);

        this.appComponent.porcentajePersonal = Math.round((94 + 3 + 60 + 94 + 3 + 90 + 90 + 37 + 5 + 94 + 3 + 90 + 37 + 7 + this.dataPersonal[0]["profile"] * 100 +
          this.dataPersonal[1]["profile"] * 100 + this.dataPersonal[2]["profile"] * 100) / 18);

        localStorage.removeItem('mediaPersonal');
        localStorage.setItem('mediaPersonal', JSON.stringify(this.appComponent.porcentajePersonal));


        if (this.appComponent.porcentajePersonal <= 40) {
          this.appComponent.colorMediaPersonal = "red";

          localStorage.setItem('colorMediaPersonal', JSON.stringify(this.appComponent.colorMediaPersonal));
        }
        if (this.appComponent.porcentajePersonal > 40 && this.mediaPersonal < 70) {
          this.appComponent.colorMediaPersonal = "orange";

          localStorage.setItem('colorMediaPersonal', JSON.stringify(this.appComponent.colorMediaPersonal));
        }
        if (this.appComponent.porcentajePersonal >= 70) {
          this.appComponent.colorMediaPersonal = "green";

          localStorage.setItem('colorMediaPersonal', JSON.stringify(this.appComponent.colorMediaPersonal));
        }

      }
    )
  }
  loadStatusDC() {
    this.mainService.getStatusDC().subscribe(
      data => {
        this.appComponent.porcentajeDataCenter = Math.round(data["DC_status"]);
        console.log("este es el status del DC");
        console.log(data["DC_status"])
        localStorage.removeItem('mediaDC');
        localStorage.setItem('mediaDC', JSON.stringify(this.appComponent.porcentajeDataCenter));
      }
    )
  }

  loadDataSala() {
    this.mainService.getDataSala().subscribe(
      data => {
      
        if (data[0]["temperature"] != null) {
          this.appComponent.temperatura = data[0]["temperature"];
          localStorage.removeItem('temperature');
          localStorage.setItem('temperature', JSON.stringify(this.appComponent.temperatura));
        }
        if (data[0]["humidity"] != null) {
          this.appComponent.humedad = data[0]["humidity"];
          localStorage.removeItem('humidity');
          localStorage.setItem('humidity', JSON.stringify(this.appComponent.humedad));
        }
        if (data[0]["noise"] != null) {
          this.appComponent.ruido = data[0]["noise"];
          localStorage.removeItem('noise');
          localStorage.setItem('noise', JSON.stringify(this.appComponent.ruido));
        }

        console.log("esta cogiendo los datos de la sala:");
        console.log(data);
      }
    )
  }


  goToMedioAmbiente() {//Aquí cargaré los datos de porcentajePersonal, porcentajeDtaCenter, Temperatura, humedad y ruido.
    if (this.userEmail == "eva@prueba.com" && this.password == "12345") {
      this.appComponent.nameUser = "Eva García";
      this.appComponent.userEmail = this.userEmail;
      this.loadDataPersonel();
      this.loadStatusDC();
      this.loadDataSala();

      // localStorage.removeItem('ipAddress');
      // localStorage.setItem('ipAddress', JSON.stringify(this.ipAddress));

      this.router.navigateByUrl('/medio-ambiente');
      return;

    }
    if (this.userEmail == "cyberops@cyberops.com" && this.password == "12345") {
      this.appComponent.nameUser = "Alberto Corredera";
      this.appComponent.userEmail = this.userEmail;
      this.loadDataPersonel();
      this.loadStatusDC();
      this.loadDataSala();

      // localStorage.removeItem('ipAddress');
      // localStorage.setItem('ipAddress', JSON.stringify(this.ipAddress));
      this.router.navigateByUrl('/medio-ambiente');
      return;
    }
    else {
      this.errorAuthentication = true;
    }
  }

}
