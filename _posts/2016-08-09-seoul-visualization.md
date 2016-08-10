---
layout: post
title:  "Seoul Subway On-board Population"
author: Y Hwang
categories: [classic, dogfooding]
tags: [Subway, Seoul, Visualization]
---

## Lorem ipsum dolor. ##

Lorem ipsum dolor sit amet, consectetur adipisicing elit. Ipsa totam quos, vero dolores modi laborum dolore dolorem ipsum possimus non, doloremque tempora mollitia asperiores itaque maiores laboriosam! Unde, enim numquam.

<div id='d3div'></div>


## Lorem ipsum dolor sit amet. ##

Lorem ipsum dolor sit amet, consectetur adipisicing elit. Labore quam, commodi itaque maxime, omnis harum qui quibusdam sequi corporis voluptatum doloribus magni nihil iste saepe incidunt neque architecto possimus iusto quo eius cupiditate totam officia, pariatur dolores? Qui impedit doloremque ipsa illum ab placeat tempore alias corporis, odit mollitia cupiditate.


<style>
svg { background-color: #1A1A1A; }

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
.no_l_1{ fill: #1B2876; }
.no_l_2{ fill: #34A939; }
.no_l_3{ fill: #FD5C09; }
.no_l_4{ fill: #268BD5; }
.no_l_5{ fill: #7411D9; }
.no_l_6{ fill: #9E3B0B; }
.no_l_7{ fill: #566112; }
.no_l_8{ fill: #DB005B; }

circle:hover {
  stroke: black;
  stroke-width: 2px;
}

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
var popByName = d3.map();
var width = 750,
    height = 550;

var projection = d3.geo.mercator()
    .center([126.9895, 37.5651])
    .scale(90000)
    .translate([width/2, height/2]);

var path = d3.geo.path().projection(projection);

var svg = d3.select("#d3div").append("svg")
    .attr("width", width)
    .attr("height", height);

var g = svg.append("g");

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
    .defer(d3.csv, "https://gist.githubusercontent.com/yoon-gu/148f049237a0468118995b427954b9cd/raw/677f34a40fb6bc0a230fc6bc08d609142c049e68/subway_in_out.csv", function(d){
        popByName.set(d.station, {"on":+d.on, "off":+d.off});
    })
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
  
  var rscale = d3.scale.linear()
    .domain([1,110000])
    .range([3,30]);

  stations = stations.sort(function(x, y){
    if(popByName.get(x.name) && popByName.get(y.name)){
      return d3.descending(popByName.get(x.name).on, popByName.get(y.name).on);
    }
  });

  for (var i = 1; i <= 8; i++) {
      g.append("g")
        .selectAll("circle")
          .data(stations)
        .enter().append("circle")
          .filter(function(d) { return +d.no_line === i })
          .attr("cx", function(d) { return projection([d.lon, d.lat])[0]; })
          .attr("cy", function(d) { return projection([d.lon, d.lat])[1]; })
          .attr("r", function(d){
            if(popByName.get(d.name))
            {
              return rscale(popByName.get(d.name).on);
            }
            else
            {
              console.log(d.name);
              return 2;
            }
          })
          .attr("opacity", 0.2)
          .on("mouseover", function(d){
            tooltip.style("visibility", "visible")
            .text(d.name + " : " + (popByName.get(d.name).on / 10000).toFixed(2) + "만명");
            var g = d3.select(this).node().parentNode;
            d3.select(g)
            .attr("class", function(){ return "no_l_" + d.no_line; })
            .selectAll("circle").attr("opacity", 0.7);
          })
          .on("mousemove", function(){
            tooltip.style("top", (event.pageY-10)+"px").style("left",(event.pageX+10)+"px");
          })
          .on("mouseout", function(){
            tooltip.style("visibility", "hidden");
            var g = d3.select(this).node().parentNode;
            d3.select(g).attr("class", "")
              .selectAll("circle").attr("opacity", 0.2);
          });
    }
}

</script>