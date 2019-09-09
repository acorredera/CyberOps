import { Component, OnInit, ɵConsole } from '@angular/core';
import { Router } from '@angular/router';
import { MainServiceService } from '../main-service.service';
import { THIS_EXPR } from '@angular/compiler/src/output/output_ast';


@Component({
  selector: 'app-personal',
  templateUrl: './personal.page.html',
  styleUrls: ['./personal.page.scss'],
})
export class PersonalPage implements OnInit {

  dataPersonal;
  //esto se borrará cuando tengamos más datos
  estadoA;
  estadoB;
  estadoC;

  colorA;
  colorB;
  colorC;

  mediaPersonal: number;
  colorMediaPersonal;

  
  constructor(private router: Router, public mainService: MainServiceService) {
    
  }

  ngOnInit() {
   
    this.loadDataPersonel();
  }

  loadDataPersonel(){
    this.mainService.getPersonal().subscribe(
      data=>{
        this.dataPersonal=data;
        console.log("ESTE ES EL DATA PERSONEL");
        console.log(this.dataPersonal);
        this.estadoA=this.dataPersonal[0]["profile"]*100;
        this.estadoB=this.dataPersonal[1]["profile"]*100;
        this.estadoC=this.dataPersonal[2]["profile"]*100;

        this.colorA=this.dataPersonal[0]["color"];
        this.colorB=this.dataPersonal[1]["color"];
        this.colorC=this.dataPersonal[2]["color"];

        this.mediaPersonal=Math.round((94+3+60+94+3+90+90+37+5+94+3+90+37+7+this.estadoA+
        this.estadoB+this.estadoC)/18);
        this.mainService.porcentajePersonal=this.mediaPersonal;
        if(this.mediaPersonal<=40){
          this.colorMediaPersonal="red";
        }
        if(this.mediaPersonal>40 && this.mediaPersonal<70){
          this.colorMediaPersonal="orange";
        }
        if(this.mediaPersonal>=70){
          this.colorMediaPersonal="green";
        }

      }
    )
  }
  
  goToDatosPersonales(personaSlected) {
    this.mainService.personaSlected=personaSlected;
    //Guardo en LS la persona selected para que, al recargar la página, sigan apareciendo los datos
    localStorage.setItem('personaSelected', JSON.stringify(this.mainService.personaSlected));
    this.router.navigateByUrl('/datos-personales');
  }

}
