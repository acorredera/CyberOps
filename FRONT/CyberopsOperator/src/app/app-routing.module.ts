import { NgModule } from '@angular/core';
import { PreloadAllModules, RouterModule, Routes } from '@angular/router';

const routes: Routes = [
  {
    path: '',
    redirectTo: 'login',
    pathMatch: 'full'
  },
  {
    path: 'home',
    loadChildren: './home/home.module#HomePageModule'
  },
  {
    path: 'list',
    loadChildren: './list/list.module#ListPageModule'
  },
  { path: 'login', loadChildren: './login/login.module#LoginPageModule' },
  { path: 'personal', loadChildren: './personal/personal.module#PersonalPageModule' },
  { path: 'data-center', loadChildren: './data-center/data-center.module#DataCenterPageModule' },
  { path: 'filtro-palabras', loadChildren: './filtro-palabras/filtro-palabras.module#FiltroPalabrasPageModule' },
  { path: 'medio-ambiente', loadChildren: './medio-ambiente/medio-ambiente.module#MedioAmbientePageModule' },
  { path: 'energia', loadChildren: './energia/energia.module#EnergiaPageModule' },
  { path: 'finanzas', loadChildren: './finanzas/finanzas.module#FinanzasPageModule'},
  { path: 'datos-personales', loadChildren: './datos-personales/datos-personales.module#DatosPersonalesPageModule' },
  { path: 'detalles-servidores', loadChildren: './detalles-servidores/detalles-servidores.module#DetallesServidoresPageModule' }
];

@NgModule({
  imports: [
    RouterModule.forRoot(routes, { preloadingStrategy: PreloadAllModules })
  ],
  exports: [RouterModule]
})
export class AppRoutingModule {}
