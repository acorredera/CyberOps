import { Component, OnInit } from '@angular/core';
import { MainServiceService } from '../main-service.service';
import * as Highcharts from 'highcharts';


declare var require: any;
require('highcharts/highcharts-more')(Highcharts);
require('highcharts/modules/solid-gauge')(Highcharts);
require('highcharts/modules/heatmap')(Highcharts);
require('highcharts/modules/treemap')(Highcharts);
require('highcharts/modules/funnel')(Highcharts);

@Component({
  selector: 'app-datos-personales',
  templateUrl: './datos-personales.page.html',
  styleUrls: ['./datos-personales.page.scss'],
})
export class DatosPersonalesPage implements OnInit {

  dataPersona;

  estadoAnimo;//Mood
  motivacion;//work
  trabajoEquipo;//coordination
  medioAmbiente;//enviroment

  estadoGeneralUsuario;
  estadoGeneralPersonal;
  fecha;
  estadoColor: string;
  profile;
  name;

  //Datos para pintar la gráfica del HR
  data;
  HRArray = [];

  //Datos para pintar la gráfica de las emociones
  data2;
  arrayNeutral = [];
  arraySadness = [];
  arrayHappy = [];
  arraySurprise = [];
  arrayAngry = [];
  arrayDate = [];

  dateInicio;
  dateFin;
  horaInicio;
  horaFin;

  porcentajePersonal;
  porcentajeGeneral;
  porcentajeDataCenter;


  constructor(public mainService: MainServiceService) {
    this.fecha = Date.now();
    //Cojo el personalSlected del localStorage
    this.mainService.personaSlected=JSON.parse(localStorage.getItem('personaSelected'));
    
    this.getDataPersona(this.mainService.personaSlected);
    setInterval(() => {
      this.fecha = Date.now()
    }, 30000);
  
  }
  ngOnInit() {
    this.porcentajeDataCenter=JSON.parse(localStorage.getItem('mediaDC'));
    this.porcentajePersonal=JSON.parse(localStorage.getItem('mediaPersonal'));

    this.porcentajeGeneral=Math.round((this.porcentajeDataCenter+this.porcentajePersonal)/2);
  }
  datos;
  getDataPersona(personaSelected){
    console.log("esta es la persona selected");
    console.log(personaSelected);
    this.mainService.getDataPersona(personaSelected).subscribe(
      data=>{
        console.log("datos de persona");
        console.log(data);
        this.datos=data;
        this.motivacion=Math.round(this.datos[0]["Work"]*100);
        this.estadoAnimo=Math.round(this.datos[0]["Mood"]*100);
        this.trabajoEquipo=Math.round(this.datos[0]["Coordination"]*100);
        this.medioAmbiente=Math.round(this.datos[0]["Environment"]*100);
        this.estadoColor=this.datos[0]["Performance"]["color"];
        this.profile=Math.round(this.datos[0]["Performance"]["punctuation"]*100);
        this.name=personaSelected;
        console.log("estos son las cpsas del usuario");
        console.log(this.motivacion);
        console.log(this.estadoAnimo);
        console.log(this.trabajoEquipo);
        console.log(this.estadoColor);
      }
    )
  }
  drawHR(fechaInicio, fechaFin) {
    this.HRArray=[];
    this.mainService.getHR(fechaInicio, fechaFin, this.mainService.personaSlected).subscribe(
      data => {
        console.log("estos son los datos del HR");
        console.log(data);
        this.data = data;
        for (let a of this.data) {
          this.HRArray.push({ x: new Date(a["date"]).getTime(), y: a["hr"] });
        }
        console.log(this.HRArray);
        Highcharts.chart('chart2', {
          chart: {
            backgroundColor: "rgb(48, 46, 47)",
            type: 'line',
            // height: (4.3 / 16 * 100) + '%',
            height: 30 + '%',
            width: 900,
          },
          xAxis: {
            type: 'datetime',
            labels: {
              format: '{value: %H:%M:%S}',
              align: 'right',
              rotation: -30
            }
          },
          yAxis: {
            title: {
              style: {
                color: 'white',
              },

            }
          },
          legend: {
            itemStyle: {
              color: 'white',

            },
            layout: 'vertical',
            align: 'right',
            verticalAlign: 'middle'
          },

          plotOptions: {
            series: {
              color: '#B79A00',
              label: {
                connectorAllowed: false
              },
              pointStart: 2010
            }
          },


          series: [{
            name: 'HR',
            data: this.HRArray,
            type: undefined
          }],
          responsive: {
            rules: [{
              condition: {
                maxWidth: 500
              },
              chartOptions: {
                legend: {
                  layout: 'horizontal',
                  align: 'center',
                  verticalAlign: 'bottom'
                }
              }
            }]
          }

        });
      });
  }
  
  drawEmotions(fechaInicio, fechaFin) {
    this.arrayNeutral = [];
    this.arraySadness = [];
    this.arrayHappy = [];
    this.arraySurprise = [];
    this.arrayAngry = [];
    this.arrayDate = [];
    this.mainService.getEmotions(fechaInicio, fechaFin, this.mainService.personaSlected).subscribe(
      data => {
        this.data2 = data;
        console.log("estos son los data del emotions");
        console.log(this.data2);
        var dateInicio = new Date(this.data2[0]["date"]).getTime();
        console.log("este es el date inicio");
        console.log(new Date(this.data2[0]["date"]));
        //Array del eje X para el chart
        var d = new Date(this.data2[0]["date"]);
        // console.log(d);
        // console.log("MARTA day");
        var dday = d.getDate();
        console.log(dday);
        // console.log("MARTA month");
        var dmonth= d.getMonth()+1;
        console.log(dmonth);
        var dyear = d.getFullYear();
        var dhour = d.getHours()
        var dmin =  d.getMinutes()
        this.arrayDate.push(dday+"/"+dmonth+"/"+dyear+" "+dhour+":"+dmin+":00");

        //contadores de emociones
        var contadorNeutral = 0;
        var contadorSadness = 0;
        var contadorHappy = 0;
        var contadorsurprise = 0;
        var contadorAngry = 0;
        let index = 0;
        for (let b of this.data2) {
          //Agrupa los datos en arrays cada 30 minutos
          if (new Date(b["date"]).getTime() >= dateInicio && new Date(b["date"]).getTime() < (dateInicio + (30 * 60 * 1000))) {
            console.log(new Date(b["date"]));
            console.log("la emosioooon");
            console.log(b["emotion"]);
            if (b["emotion"] == "neutral") {
              contadorNeutral = contadorNeutral + 1;
            }
            if (b["emotion"] == "sadness") {
              contadorSadness = contadorSadness + 1;
            }
            if (b["emotion"] == "happy") {
              contadorHappy = contadorHappy + 1;
            }
            if (b["emotion"] == "surprise") {
              contadorsurprise = contadorsurprise + 1;
            }
            if (b["emotion"] == "angry") {
              contadorAngry = contadorAngry + 1;
            }
            if (index == (this.data2.length - 1)) {
              this.arrayNeutral.push(contadorNeutral);
              this.arraySadness.push(contadorSadness);
              this.arrayHappy.push(contadorHappy);
              this.arraySurprise.push(contadorsurprise);
              this.arrayAngry.push(contadorAngry);
              var d= new Date(b["date"]);
              // console.log(d);
              // console.log("MARTA day");
              var dday = d.getDate();
              console.log(dday);
              // console.log("MARTA month");
              var dmonth= d.getMonth()+1;
              console.log(dmonth);
              var dyear = d.getFullYear();
              var dhour = d.getHours()
              var dmin =  d.getMinutes()
              this.arrayDate.push(dday+"/"+dmonth+"/"+dyear+" "+dhour+":"+dmin+":00");
            }
          }
          //Si el dato de la fecha está a más distancia de 30 minutos, lomete en otro array
          if (new Date(b["date"]).getTime() >= (dateInicio + (30 * 60 * 1000))) {
            if (b["emotion"] == "neutral") {
              contadorNeutral = contadorNeutral + 1;
            }
            if (b["emotion"] == "sadness") {
              contadorSadness = contadorSadness + 1;
            }
            if (b["emotion"] == "happy") {
              contadorHappy = contadorHappy + 1;
            }
            if (b["emotion"] == "surprise") {
              contadorsurprise = contadorsurprise + 1;
            }
            if (b["emotion"] == "angry") {
              contadorAngry = contadorAngry + 1;
            }
            //Meto en el array el contador de este periodo de tiempo
            this.arrayNeutral.push(contadorNeutral);
            this.arraySadness.push(contadorSadness);
            this.arrayHappy.push(contadorHappy);
            this.arraySurprise.push(contadorsurprise);
            this.arrayAngry.push(contadorAngry);
            var d = new Date(b["date"]);
            // console.log(d);
            // console.log("MARTA day");
            var dday = d.getDate();
            console.log(dday);
            // console.log("MARTA month");
            var dmonth= d.getMonth()+1;
            console.log(dmonth);
            var dyear = d.getFullYear();
            var dhour = d.getHours()
            var dmin =  d.getMinutes()
            this.arrayDate.push(dday+"/"+dmonth+"/"+dyear+" "+dhour+":"+dmin+":00");
            //Vuelvo a poner los contadores a 0 para volver a empezar
            contadorNeutral = 0;
            contadorSadness = 0;
            contadorHappy = 0;
            contadorsurprise = 0;
            contadorAngry = 0;
            dateInicio = new Date(b["date"]).getTime();
          }
          index++;
        }

        Highcharts.chart('chart1', {
          chart: {
            backgroundColor: "rgb(48, 46, 47)",
            type: 'column',
            height: 30 + '%',
            width: 900,
          },

          xAxis: {
            categories: this.arrayDate
          },
          yAxis: {
            title:{
              style:{color:"white"}
            },
            min: 0,
            stackLabels: {
              enabled: true,
              style: {
                fontWeight: 'bold',
                color: ( // theme
                  Highcharts.defaultOptions.title.style &&
                  Highcharts.defaultOptions.title.style.color 
                ) || 'white'
              }
            }
          },
          legend: {
            itemStyle: {
              color: 'white',
            },
            align: 'center',
            // x: 0,
            verticalAlign: 'top',
            // y: 0,
            floating: true,
            backgroundColor:
              Highcharts.defaultOptions.legend.backgroundColor || 'rgb(48,46,47)',
            // borderColor: '#CCC',
            // borderWidth: 1,
            shadow: false
          },
          tooltip: {
            headerFormat: '<b>{point.x}</b><br/>',
            pointFormat: '{series.name}: {point.y}<br/>Total: {point.stackTotal}'
          },
          plotOptions: {
            column: {
              stacking: 'normal',
              dataLabels: {
               
                enabled: false
              }
            }
          },
          series: [{
            name: 'Indiferente',
            color:"rgb(183,154,0)",
            data: this.arrayNeutral,
            type: undefined
          }, {
            name: 'Feliz',
            color:"rgb(67, 160, 71)",
            data: this.arrayHappy,
            type: undefined
          }, {
            name: 'Triste',
            color:"rgb(164,154,154)",
            data: this.arraySadness,
            type: undefined
          },
          {
            name: 'Enfadado',
            color:"rgb(198, 40, 40)",
            data: this.arrayAngry,
            type: undefined
          },
          {
            name: 'Sorprendido',
            color:"rgb(33, 150, 243)",
            data: this.arraySurprise,
            type: undefined
          }]
        });
        // console.log("estos son los arrays de sentimientos");
        // console.log(this.arrayAngry);
        // console.log(this.arrayHappy);
        // console.log(this.arraySadness);
        // console.log(this.arrayNeutral);
        // console.log(this.arraySurprise);
      }
    )
  }
  arousalArray;
  drawArousal(fechaInicio, fechaFin){
    this.arousalArray=[];
    this.mainService.getArousal(fechaInicio, fechaFin, this.mainService.personaSlected).subscribe(
      data => {
        console.log("estos son los datos del arousal");
        console.log(data);
        this.data = data;
          for (let a of this.data) {
          this.arousalArray.push({ x: new Date(a["date"]).getTime(), y: a["arousal"] });
        }
        console.log(this.HRArray);
        Highcharts.chart('chart3', {
          chart: {
            backgroundColor: "rgb(48, 46, 47)",
            type: 'line',
            // height: (4.3 / 16 * 100) + '%',
            height: 30 + '%',
            width: 900,
          },
          xAxis: {
            type: 'datetime',
            labels: {
              format: '{value: %H:%M:%S}',
              align: 'right',
              rotation: -30
            }
          },
          yAxis: {
            title: {
              style: {
                color: 'white',
              },

            }
          },
          legend: {
            itemStyle: {
              color: 'white',

            },
            layout: 'vertical',
            align: 'right',
            verticalAlign: 'middle'
          },

          plotOptions: {
            series: {
              color: '#B79A00',
              label: {
                connectorAllowed: false
              },
              pointStart: 2010
            }
          },


          series: [{
            name: 'Arousal',
            data: this.arousalArray,
            type: undefined
          }],
          responsive: {
            rules: [{
              condition: {
                maxWidth: 500
              },
              chartOptions: {
                legend: {
                  layout: 'horizontal',
                  align: 'center',
                  verticalAlign: 'bottom'
                }
              }
            }]
          }

        });
      });
  }
 
  getDataPersonal(nombre) {
    this.mainService.getDataPersona(nombre).subscribe(
      data => {
        this.dataPersona = data;
        this.trabajoEquipo = Math.round(this.dataPersona[0]["Work"] * 100);
        this.estadoAnimo = this.dataPersona[0]["Mood"] * 100;
        this.motivacion = this.dataPersona[0]["Coordination"] * 100;
        this.medioAmbiente = this.dataPersona[0]["Environment"] * 100;

        this.estadoGeneralUsuario = Math.round((this.dataPersona + this.estadoAnimo + this.motivacion + this.medioAmbiente) / 4);
        if (this.estadoGeneralUsuario <= 40) {
          this.estadoColor = "red";
        }
        if (this.estadoGeneralUsuario > 40 && this.estadoGeneralUsuario < 70) {
          this.estadoColor = "orange";
        }
        if (this.estadoGeneralUsuario >= 70) {
          this.estadoColor = "green";
        }

      }
    )
  }
  filtrarDatos(){
   //No se por qué me da el timestamp en milisegundas(yo lo quiero en segundos,por eso divido entre 1000)
    var dateInicioTS=(new Date(this.dateInicio)).getTime();
    var dateFinTS=(new Date(this.dateFin)).getTime();
    this.drawHR(dateInicioTS/1000, dateFinTS/1000);
    this.drawEmotions(dateInicioTS/1000, dateFinTS/1000);
    this.drawArousal(dateInicioTS/1000, dateFinTS/1000);
  }

}
