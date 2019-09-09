import { Component, OnInit } from '@angular/core';
import { MenuController } from '@ionic/angular';
import { AppComponent } from '../app.component';

@Component({
  selector: 'app-medio-ambiente',
  templateUrl: './medio-ambiente.page.html',
  styleUrls: ['./medio-ambiente.page.scss'],
})
export class MedioAmbientePage implements OnInit {

  constructor(public menuCtrl: MenuController, private appComponent: AppComponent) { }

  ngOnInit() {
    // this.appComponent.nameUser=localStorage.getItem('nameUser');
  }
  ionViewWillEnter() {
    this.menuCtrl.enable(true);
  }

}
