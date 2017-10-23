function PointsStyle(feature) {
	var style = new ol.style.Style({
		image: new ol.style.Circle({
			radius: 2,
			fill: new ol.style.Fill({
				color: 'rgba(255, 0, 255, 0.75)'
			})
		})
	})
	return [style];
}

function MBRsStyle(feature) {
	var style = new ol.style.Style({
		stroke: new ol.style.Stroke({
			color: 'blue',
			lineDash: [3],
			width: 1.5
		}),
		fill: new ol.style.Fill({
			color: 'rgba(0, 0, 255, 0.01)'
		})
	})
	return [style];
}

var MBRsSource =  new ol.source.Vector({
	format: new ol.format.GeoJSON(),
	projection : 'urn:ogc:def:crs:EPSG::3068',
	url: 'RTree_B60K_E50.0_M12_P1024.geojson'
})

var MBRsLayer = new ol.layer.Vector({
	title: 'MBRs',
	source: MBRsSource,
	style: MBRsStyle
});

var PointsSource =  new ol.source.Vector({
	format: new ol.format.GeoJSON(),
	projection : 'urn:ogc:def:crs:EPSG::3068',
	url: 'RTree_B60K.geojson'
})

var PointsLayer = new ol.layer.Vector({
	title: 'Points',
	source: PointsSource,
	style: PointsStyle
});

var cx = 25264.538
var cy = 21037.674
var extend = 15000
proj4.defs("EPSG:3068","+proj=tmerc +lat_0=0 +lon_0=126 +k=1 +x_0=500000 +y_0=0 +ellps=krass +units=m +no_defs");
var proj = ol.proj.get('EPSG:3068');
proj.setExtent([cx - extend, cy - extend, cx + extend, cy + extend]);    

var map = new ol.Map({
	layers: [
		new ol.layer.Tile({
			title: 'OSM',
			type: 'base',
			visible: true,
			source: new ol.source.OSM()
		}),
		PointsLayer,
		MBRsLayer,
		BuffersLayer
	],
	target: 'map',
	controls: ol.control.defaults({
		attributionOptions: ({
			collapsible: true
		})
	}),
	view: new ol.View({
		projection: 'EPSG:3068',
		center: [cx, cy],
		zoom: 7
	})
});
var layerSwitcher = new ol.control.LayerSwitcher({
	tipLabel: 'Legend' // Optional label for button
});
map.addControl(layerSwitcher);
