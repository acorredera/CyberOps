import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Routes, RouterModule } from '@angular/router';

import { IonicModule } from '@ionic/angular';

import { DetallesServidoresPage } from './detalles-servidores.page';

const routes: Routes = [
  {
    path: '',
    component: DetallesServidoresPage
  }
];

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    RouterModule.forChild(routes)
  ],
  declarations: [DetallesServidoresPage]
})
export class DetallesServidoresPageModule {}
