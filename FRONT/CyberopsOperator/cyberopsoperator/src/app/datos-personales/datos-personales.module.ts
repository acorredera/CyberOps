import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Routes, RouterModule } from '@angular/router';

import { IonicModule } from '@ionic/angular';

import { DatosPersonalesPage } from './datos-personales.page';

const routes: Routes = [
  {
    path: '',
    component: DatosPersonalesPage
  }
];

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    RouterModule.forChild(routes)
  ],
  declarations: [DatosPersonalesPage]
})
export class DatosPersonalesPageModule {}
