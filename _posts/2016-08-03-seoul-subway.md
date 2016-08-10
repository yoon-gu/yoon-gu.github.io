---
layout: post
title:  "Visualization Seoul Subway Location"
author: Y Hwang
categories: [Report, Private]
tags: [Subway, Seoul, Visualization]
---

## Seoul Subway Location ##

Lorem ipsum dolor sit amet, consectetur adipisicing elit. Ipsa totam quos, vero dolores modi laborum dolore dolorem ipsum possimus non, doloremque tempora mollitia asperiores itaque maiores laboriosam! Unde, enim numquam.

<div id='d3div'></div>



## Lorem ipsum dolor sit amet. ##

Lorem ipsum dolor sit amet, consectetur adipisicing elit. Labore quam, commodi itaque maxime, omnis harum qui quibusdam sequi corporis voluptatum doloribus magni nihil iste saepe incidunt neque architecto possimus iusto quo eius cupiditate totam officia, pariatur dolores? Qui impedit doloremque ipsa illum ab placeat tempore alias corporis, odit mollitia cupiditate.



<style>
div #d3div{ background-color: #1A1A1A; }

path {
  stroke-linejoin: round;
}

.land {
  fill: #4C4C4C;
}

.states {
  fill: none;
  stroke: darkgray;
}

circle.no_l_1{ fill: #1B2876; }
circle.no_l_2{ fill: #34A939; }
circle.no_l_3{ fill: #FD5C09; }
circle.no_l_4{ fill: #268BD5; }
circle.no_l_5{ fill: #7411D9; }
circle.no_l_6{ fill: #9E3B0B; }
circle.no_l_7{ fill: #566112; }
circle.no_l_8{ fill: #DB005B; }

svg .municipality-label {
  fill: white;
  font-size: 12px;
  font-weight: 300;
  text-anchor: middle;
  font-family: sans-serif;
}
</style>
<body>
<script src="//d3js.org/d3.v3.min.js"></script>
<script src="//d3js.org/queue.v1.min.js"></script>
<script src="//d3js.org/topojson.v1.min.js"></script>
<script>

var width = 750,
    height = 550;
var centered;
var projection = d3.geo.mercator()
    .center([126.9895, 37.5651])
    .scale(90000)
    .translate([width/2, height/2]);

var path = d3.geo.path().projection(projection);

var svg = d3.select("#d3div").append("svg")
    .attr("width", width)
    .attr("height", height);

var g = svg.append("g");

svg.append("rect")
    .attr("class", "background")
    .attr("width", width)
    .attr("height", height)
    .style("fill", "none")
    .on("click", clicked);

var tooltip = d3.select("body")
  .append("div")
  .style("position", "absolute")
  .style("z-index", "10")
  .style("visibility", "hidden")
  .style("font-family", "sans-serif")
  .style("color", "white")
  .style("font-size", "11px");

queue()
    .defer(d3.json, "https://gist.githubusercontent.com/yoon-gu/b051fd123385303a5c03f0e0a833516c/raw/9fff4a65830be008709112c190c3ed939d42e994/seoul_municipalities_topo.json")
    .defer(d3.csv, "https://gist.githubusercontent.com/yoon-gu/902efb6d5bd345e3837e035a3c0642b8/raw/3cf9c9418da25e195cfe8db9104497408b6e5bbd/station_latlen.csv")
    .await(ready);

function ready(error, kor, stations) {
  if (error) throw error;
  var features = topojson.feature(kor, kor.objects.seoul_municipalities_geo).features;
  g.selectAll("path")
        .data(features)
      .enter().append("path")
        .attr("class", "land")
        .attr("d", path)
        .attr("id", function(d) { return d.properties.name; })
        .on("click", clicked)
        .append("title");

  g.append("path")
      .datum(topojson.mesh(kor, kor.objects.seoul_municipalities_geo, function(a, b) { return a !== b; }))
      .attr("class", "states")
      .attr("d", path);

  g.selectAll('text')
      .data(features)
      .enter().append("text")
        .attr("transform", function(d) { return "translate(" + path.centroid(d) + ")"; })
        .attr("dy", ".35em")
        .attr("class", "municipality-label")
        .text(function(d) { return d.properties.name; })
  
  g.selectAll("circle")
      .data(stations)
    .enter().append("circle")
      .attr("cx", function(d) { return projection([d.lon, d.lat])[0]; })
      .attr("cy", function(d) { return projection([d.lon, d.lat])[1]; })
      .attr("r", 3)
      .attr("opacity", 0.7)
      .attr("class", function(d){ return "no_l_" + d.no_line; })
      .on("mouseover", function(d){
        tooltip.style("visibility", "visible")
        .text(d.name);
      })
      .on("mousemove", function(){
        tooltip.style("top", (event.pageY-10)+"px").style("left",(event.pageX+10)+"px");
      })
      .on("mouseout", function(){
        tooltip.style("visibility", "hidden");
      });

}

function clicked(d) {
  var x, y, k;

  if (d && centered !== d) {
    var centroid = path.centroid(d);
    x = centroid[0];
    y = centroid[1];
    k = 3;
    centered = d;
  } else {
    x = width / 2;
    y = height / 2;
    k = 1;
    centered = null;
  }

  g.selectAll("path")
      .classed("active", centered && function(d) { return d === centered; });

  g.transition()
      .duration(750)
      .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")scale(" + k + ")translate(" + -x + "," + -y + ")")
      .style("stroke-width", 1.5 / k + "px");
}
</script>