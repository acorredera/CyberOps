import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import * as d3 from 'd3';
import { MainServiceService } from '../main-service.service';



@Component({
  selector: 'app-detalles-servidores',
  templateUrl: './detalles-servidores.page.html',
  styleUrls: ['./detalles-servidores.page.scss'],
})
export class DetallesServidoresPage implements OnInit{
  @ViewChild('chart') chart: ElementRef;
  porcentajePersonal;
  porcentajeGeneral;
  porcentajeDataCenter;
  
  listServers;
  lostColorsServers=[];
  constructor(public mainService: MainServiceService) {

  }
  ngOnInit() {
    this.porcentajeDataCenter=JSON.parse(localStorage.getItem('mediaDC'));
    this.porcentajePersonal=JSON.parse(localStorage.getItem('mediaPersonal'));

    this.porcentajeGeneral=Math.round((this.porcentajeDataCenter+this.porcentajePersonal)/2);
    this.getStatusServersOfRacks();
  }
  ////////////////////////////////////////////////////////////// 
  //////////////////////// Constants /////////////////////////// 
  ////////////////////////////////////////////////////////////// 
  d;
  pi2 = 2 * Math.PI
  pi1_2 = Math.PI / 2

  numData = 1440 // = 60 * 24 -> minutes per day
  dotRadius = 1

  //Locations of visual elements based on the data
  averageBabies = 7.3
  maxBabies = 14.8
  minBabies = 4.8
  axisLocation = 10.5
  gridLineData = [6, 9]
  outerCircleShadow = 12.5

  //Colors
  colorsRed = ['#ffa500', '#fb9200', '#f58200', '#ee7000', '#e65e00', '#dd4c01', '#d13a01', '#c12e03', '#b02404', '#9f1905', '#8e1005', '#7d0603', '#6b0101']
  colorsBlue = ['#0d4982', '#1d6092', '#2c79a1', '#3294a5', '#1ab29d'];

  ////////////////////////////////////////////////////////////// 
  //////////////////////// Create SVG //////////////////////////
  ////////////////////////////////////////////////////////////// 

  margin = {
    top: 20,
    right: 20,
    bottom: 20,
    left: 20
  }
  width = 1250 - this.margin.left - this.margin.right
  height = 900 - this.margin.top - this.margin.bottom

  //SVG container - using d3's margin convention
  svg: any;
  timeScale: any;
  birthScale: any;
  areaScale: any;

  defs: any;
  filter: any;
  feMerge: any;

  times = ["midnight", "1am", "2am", "3am", "4am", "5am", "6am", "7am", "8am", "9am", "10am", "11am", "noon", "1pm", "2pm", "3pm", "4pm", "5pm", "6pm", "7pm", "8pm", "9pm", "10pm", "11pm"]
  timeLabels: any;
  pie: any;
  arc: any;

  circles: any;
  clips: any;
  area: any;
  gridLines: any;
  circlesTop: any;
  averageLine: any;
  annotations: any;
  annotationData: any;
  makeAnnotations: any;

  ngAfterContentInit() {
    var chart = this.chart.nativeElement;
    ////////////////////////////////////////////////////////////// 
    //////////////////////// Create SVG //////////////////////////
    ////////////////////////////////////////////////////////////// 
    //SVG container - using d3's margin convention
    this.svg = d3.select(chart)
      .append("svg")//Crea el elemento SVG que contiene todas las figuras, svg es una referencia que paunta alobjeto del DOM
      .attr("width", this.width + this.margin.left + this.margin.right)
      .attr("height", this.height + this.margin.top + this.margin.bottom)
      .append("g") //Añade el elemento g al DOM, luego le daremos la posicion
      .attr("transform", "translate(" + (this.margin.left + this.width / 2) + "," + (this.margin.top + this.height / 2) + ")")

    //////////////////////////////////////////////////////////////
    ////////////////////// Create scales /////////////////////////
    //////////////////////////////////////////////////////////////
    //Angle scale for the time
    this.timeScale = d3.scaleLinear()
      .domain([0, this.numData - 1]) //rango de los valores iniciales, de donde a donde
      .range([0.025 * this.pi2, 0.975 * this.pi2]); //rango de valores de salida
    //Radius scale for the number of births
    this.birthScale = d3.scaleLinear()
      .domain([0, this.maxBabies])
      .range([0, this.height / 2]);
    //Area between the loess line and the average line

    const areaScale = d3.radialArea<any>()
      .angle(d => this.timeScale(d.time))
      .innerRadius(d => this.birthScale(d.line))
      .outerRadius(d => this.birthScale(this.averageBabies));

    ////////////////////////////////////////////////////////////// 
    ///////////////////// Create SVG effects ///////////////////// 
    //////////////////////////////////////////////////////////////

    this.defs = this.svg.append("defs")

    //Create a shadow filter
    this.filter = this.defs.append("filter").attr("id", "shadow")
    this.filter.append("feColorMatrix")
      .attr("type", "matrix")
      .attr("values", "0 0 0 0   0 0 0 0 0   0 0 0 0 0   0 0 0 0 0.1 0")
    this.filter.append("feGaussianBlur")
      .attr("stdDeviation", "5")
      .attr("result", "coloredBlur")
    this.feMerge = this.filter.append("feMerge")
    this.feMerge.append("feMergeNode").attr("in", "coloredBlur")
    this.feMerge.append("feMergeNode").attr("in", "SourceGraphic")

    //Create background "chart-area" circle
    this.svg.append("circle")
      .attr("r", this.birthScale(this.outerCircleShadow))
      // .style("fill", "white")
      .style("fill", "rgba(255,255,255,0.3)")
      .style("filter", "url(#shadow)");

    /////////////////////////////////////////////////////////////
    ///////////////////////// Add title //////////////////////////
    //////////////////////////////////////////////////////////////
    this.svg.append("text")
      .attr("class", "title-top")
      .attr("y", -5)
      .text("Minutes")

    this.svg.append("text")
      .attr("class", "title-bottom")
      .attr("y", 15)
      .text("per day")

    //////////////////////////////////////////////////////////////
    ////////////////////// Draw time labels //////////////////////
    //////////////////////////////////////////////////////////////
    //Draw the hour labels
    this.timeLabels = this.svg.append("g")
      .attr("class", "time-label-group")

    //Will calculate starting and ending angles
    this.pie = d3.pie()
      .startAngle(this.timeScale(0)) //Because we're not using a full 2*pi circle
      .endAngle(this.timeScale(this.numData - 1)) //Because we're not using a full 2*pi circle
      .value(60) //Each hour is 60 minutes long
      .padAngle(.01) //A bit of space between each slice
      .sort(null) //Don't sort, but keep the order as in the data

    //Will create the SVG arc path formulas
    this.arc = d3.arc()
      .innerRadius(this.birthScale(this.axisLocation))
      .outerRadius(this.birthScale(this.axisLocation) * 1.005) //Make it a very thin donut chart

    //Draw the arc	
    this.timeLabels.selectAll(".time-axis")
      .data(this.pie(this.times))
      .enter().append("path")
      .attr("class", "time-axis")
      .attr("id", (d, i) => "time-label-" + i)
      .attr("d", this.arc)

    //Append the time labels
    this.timeLabels.selectAll(".time-axis-text")
      .data(this.times)
      .enter().append("text")
      .attr("class", "time-axis-text")
      .attr("dy", 12)
      .append("textPath")
      .attr("xlink:href", (d, i) => "#time-label-" + i)
      .text(d => d);


    //////////////////////////////////////////////////////////////
    /////////////////////// Read in data /////////////////////////
    //////////////////////////////////////////////////////////////

    d3.csv("../../assets/SciAm_minute_per_day_2014.csv").then((babyData) => {
      //   babyData.forEach(d => {
      //     d.time = (+parseInt(d.time)).toString();
      //     d.births = (+parseInt(d.births)).toString();
      //     d.line = (+parseInt(d.line)).toString();
      // })
      //Turn strings into actual numbers
      babyData.forEach(d => {

      })

      //////////////////////////////////////////////////////////////
      ////////////////////// Create clip paths /////////////////////
      //////////////////////////////////////////////////////////////

      this.clips = this.svg.append("g").attr("class", "clip-group")

      this.clips.append("clipPath")
        .attr("id", "clip-area")
        .append("path")
        .attr("d", areaScale(babyData))

      //////////////////////////////////////////////////////////////
      ///////////////////////// Draw areas /////////////////////////
      //////////////////////////////////////////////////////////////

      this.area = this.svg.append("g")
        .attr("class", "area-group")
        .attr("clip-path", "url(#clip-area)")

      //Create the circles but have them clipped by the area
      this.area.selectAll(".color-circle-above")
        .data(this.colorsRed.reverse())
        .enter().append("circle")
        .attr("class", "color-circle-above")
        .attr("r", (d, i) => this.birthScale(this.maxBabies) - (this.birthScale(this.maxBabies) - this.birthScale(this.averageBabies)) / this.colorsRed.length * i)
        .style("fill", d => d)
        .style("filter", "url(#shadow)")

      //Create the circles but have them clipped by the area
      this.area.selectAll(".color-circle-below")
        .data(this.colorsBlue.reverse())
        .enter().append("circle")
        .attr("class", "color-circle-below")
        .attr("r", (d, i) => this.birthScale(this.averageBabies) - (this.birthScale(this.averageBabies) - this.birthScale(this.minBabies)) / this.colorsBlue.length * i)
        .style("fill", d => d)
        .style("filter", "url(#shadow)")

      //////////////////////////////////////////////////////////////
      ///////////////////// Draw gridlines /////////////////////////
      //////////////////////////////////////////////////////////////

      this.gridLines = this.svg.append("g")
        .attr("class", "gridline-group")

      //Add the axis lines
      this.gridLines.selectAll(".axis-line")
        .data(this.gridLineData)
        .enter().append("path")
        .attr("class", "axis-line")
        .attr("d", d => this.arcPath(this.birthScale(d)))

      //Add the numbers in between
      this.gridLines.selectAll(".axis-number")
        .data(this.gridLineData)
        .enter().append("text")
        .attr("class", "axis-number")
        .attr("y", d => -this.birthScale(d))
        .attr("dy", "0.4em")
        .text(d => d)

      //////////////////////////////////////////////////////////////
      /////////////////////// Draw circles /////////////////////////
      //////////////////////////////////////////////////////////////

      this.circlesTop = this.svg.append("g")
        .attr("class", "circle-group")
        .attr("clip-path", "url(#clip-area)")

      //Using scales in radial
      this.circlesTop.selectAll(".circle-top")
        .data(babyData)
        .enter().append("circle")
        .attr("class", "circle-top")
        .attr("cx", d => this.birthScale(d.births) * Math.cos(this.timeScale(d.time) - this.pi1_2)) //radius * cos(angle)
        .attr("cy", d => this.birthScale(d.births) * Math.sin(this.timeScale(d.time) - this.pi1_2)) //radius * sin(angle)
        .attr("r", this.dotRadius)

      //////////////////////////////////////////////////////////////
      ///////////////////////// Draw lines /////////////////////////
      //////////////////////////////////////////////////////////////

      //Draw an average line
      this.averageLine = this.svg.append("path")
        .attr("class", "average-line")
        .attr("d", () => {
          let r = this.birthScale(this.averageBabies)
          return this.arcPath(r)
        })

      ////////////////////////////////////////////////////////////// 
      ///////////////////////// Annotations ////////////////////////
      ////////////////////////////////////////////////////////////// 

      // this.annotations = this.svg.append("g")
      //   .attr("class", "annotation-group")

      // this.annotationData = [
      //   {
      //     className: "average-note",
      //     note: { title: "The average", label: "On average " + this.averageBabies + " babies are born per minute", wrap: 140 },
      //     data: { births: this.averageBabies, time: 1310 },
      //     type: d3.annotationCallout,
      //     dy: -70,
      //     dx: -70,
      //     connector: { end: "dot" }
      //   }, {
      //     note: { title: "The night dip", label: "In the evening far fewer babies are born per minute, even when compared to only natural births throughout the day", wrap: 270 },
      //     data: { births: 6, time: 210 },
      //     type: d3.annotationCallout,
      //     dy: -40,
      //     dx: 60,
      //     connector: { end: "dot" }
      //   }, {
      //     note: { title: "The early morning peak", label: "Several factors combine for an explosion of babies starting around at 7:45am", wrap: 230 },
      //     data: { births: 11.5, time: 485 },
      //     type: d3.annotationCallout,
      //     dy: -30,
      //     dx: 40,
      //     connector: { end: "dot" }
      //   }, {
      //     note: { title: "The after-lunch boom", label: "Afternoon planned c-sections cause another very distinct bump in the number of babies born", wrap: 280 },
      //     data: { births: 10, time: 770 },
      //     type: d3.annotationCallout,
      //     dy: 30,
      //     dx: -40,
      //     connector: { end: "dot" }
      //   }
      // ]

      // //Set-up the annotation
      // this.makeAnnotations = d3Annotation.annotation()
      //     .type(d3.annotationCalloutElbow)
      //     .accessors({
      //         x: d => birth * Math.cos(this.timeScale(d.time) - this.pi1_2,
      //         y: d => this.birthScale(d.births) * Math.sin(this.timeScale(d.time) - this.pi1_2)
      //     })

      //     .notePadding(8)
      //     .annotations(this.annotationData)

      // //Create the annotation
      // this.annotations.call(this.makeAnnotations)

    })//d3.csv
  }
  ////////////////////////////////////////////////////////////// 
  /////////////////////// Helper functions /////////////////////
  ////////////////////////////////////////////////////////////// 

  //Pinta la <path> del elemento svg que queramos
  arcPath(r) {
    let xStart = r * Math.cos(this.timeScale(0) - this.pi1_2)
    let yStart = r * Math.sin(this.timeScale(0) - this.pi1_2)
    let xEnd = r * Math.cos(this.timeScale(this.numData - 1) - this.pi1_2)
    let yEnd = r * Math.sin(this.timeScale(this.numData - 1) - this.pi1_2)

    return "M" + [xStart, yStart] + " A" + [r, r] + " 0 1 1 " + [xEnd, yEnd]
  }//function arcPath

  getStatusServersOfRacks(){
  
    this.mainService.getStatusServersOfRacks().subscribe(
      data=>{
        this.listServers=data['server_status'];
        console.log("este es el status de servidores");
        console.log(data);
        console.log(this.listServers)
      
      }
    )
  }

}
