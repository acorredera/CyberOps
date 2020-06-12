import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { MainServiceService } from '../main-service.service';
import { THIS_EXPR } from '@angular/compiler/src/output/output_ast';


@Component({
  selector: 'app-data-center',
  templateUrl: './data-center.page.html',
  styleUrls: ['./data-center.page.scss'],
})
export class DataCenterPage implements OnInit {

  statusRacks;
  racksSala1;
  racksSala2;
  racksSala3;

  porcentajePersonal;
  porcentajeGeneral;
  porcentajeDataCenter;

  racksLoad: boolean;

  constructor(private router: Router, public mainService: MainServiceService) {
    this.racksLoad=false;
   
    this.getStatusRacks();
   }

  ngOnInit() {
    this.porcentajeDataCenter=JSON.parse(localStorage.getItem('mediaDC'));
    this.porcentajePersonal=JSON.parse(localStorage.getItem('mediaPersonal'));
    this.porcentajeGeneral=Math.round((this.porcentajeDataCenter+this.porcentajePersonal)/2);
    
    

  }
  getStatusRacks(){
    this.mainService.getStatusRacks().subscribe(
      data=>{
        this.statusRacks=data;
        localStorage.removeItem('datosRacks');
        localStorage.setItem('datosRacks', JSON.stringify(this.statusRacks));

        this.racksSala1=data[0]["rack_status"];
        localStorage.removeItem('racksSala1');
        localStorage.setItem('racksSala1', JSON.stringify(this.racksSala1));

        this.racksSala2=data[1]["rack_status"];
        localStorage.removeItem('racksSala2');
        localStorage.setItem('racksSala2', JSON.stringify(this.racksSala2));

        this.racksSala3=data[2]["rack_status"];
        localStorage.removeItem('racksSala3');
        localStorage.setItem('racksSala3', JSON.stringify(this.racksSala3));

        this.racksLoad=true;
        console.log("estos es el status de los racks de servidores");
        console.log(data);
        console.log(this.racksSala1);
        console.log(this.racksSala1[0]["rack_id"])
      },
      err=>{
        console.log("ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRROOOR")
      }
    )
  }
  
  goToDetalleServidor(rack, room){
    console.log("detalle servidor");
    console.log("este es el rack");
    console.log(rack);
    this.mainService.rackIdSelected=rack["rack_id"];
    this.mainService.roomIdSelected=room;

    localStorage.removeItem('rackIdSelected');
    localStorage.setItem('rackIdSelected', JSON.stringify(rack["rack_id"]));

    localStorage.removeItem('roomIdSelected');
    localStorage.setItem('roomIdSelected', JSON.stringify(room));
  
    this.router.navigateByUrl('/detalles-servidores');
  }
}
