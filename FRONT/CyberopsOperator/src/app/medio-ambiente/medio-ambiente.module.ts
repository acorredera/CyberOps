import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Routes, RouterModule } from '@angular/router';

import { IonicModule } from '@ionic/angular';

import { MedioAmbientePage } from './medio-ambiente.page';

const routes: Routes = [
  {
    path: '',
    component: MedioAmbientePage
  }
];

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    RouterModule.forChild(routes)
  ],
  declarations: [MedioAmbientePage]
})
export class MedioAmbientePageModule {}
