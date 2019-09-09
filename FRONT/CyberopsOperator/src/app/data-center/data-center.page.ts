import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { MainServiceService } from '../main-service.service';

@Component({
  selector: 'app-data-center',
  templateUrl: './data-center.page.html',
  styleUrls: ['./data-center.page.scss'],
})
export class DataCenterPage implements OnInit {

  statusRacks;
  racksSala1;
  racksSala2;
  constructor(private router: Router, public mainService: MainServiceService) { }

  ngOnInit() {
    this.getStatusRacks();
    this.getStatusServersOfRacks();
  }
  getStatusRacks(){
    this.mainService.getStatusRacks().subscribe(
      data=>{
        this.statusRacks=data;
        this.racksSala1=data[0]["rack_status"];
        this.racksSala2=data[1]["rack_status"];
        console.log("estos es el status de los racks de servidores");
        console.log(data);
        console.log(this.racksSala1);
        console.log(this.racksSala1[0]["rack_id"])
      }
    )
  }
  getStatusServersOfRacks(){
    this.mainService.getStatusServersOfRacks().subscribe(
      data=>{
        console.log("estos es el status de servidores");
        console.log(data);
      }
    )
  }
  goToDetalleServidor(){
    console.log("detalle servidor");
    this.router.navigateByUrl('/detalles-servidores');
  }
}
