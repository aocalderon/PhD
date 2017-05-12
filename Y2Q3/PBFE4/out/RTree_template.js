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

function BuffersStyle(feature) {
	var style = new ol.style.Style({
		stroke: new ol.style.Stroke({
			color: 'rgba(0, 0, 255, 0.8)',
			lineDash: [6],
			width: 0.5
		}),
		fill: new ol.style.Fill({
			color: 'rgba(0, 0, 255, 0.1)'
		})
	})
	return [style];
}

var MBRsSource =  new ol.source.Vector({
	format: new ol.format.GeoJSON(),
	projection : 'urn:ogc:def:crs:EPSG::4799',
	url: 'RTree_P20K.json'
})

//MBRsSource.addFeature(new ol.Feature(new ol.geom.Circle([5e6, 7e6], 1e6)));

var MBRsLayer = new ol.layer.Vector({
	title: 'MBRs',
	source: MBRsSource,
	style: MBRsStyle
});

var BuffersSource =  new ol.source.Vector({
	format: new ol.format.GeoJSON(),
	projection : 'urn:ogc:def:crs:EPSG::4799',
	url: 'RTree_P20K_buffer.json'
})

var BuffersLayer = new ol.layer.Vector({
	title: 'Buffer',
	source: BuffersSource,
	style: BuffersStyle
});

var PointsSource =  new ol.source.Vector({
	format: new ol.format.GeoJSON(),
	projection : 'urn:ogc:def:crs:EPSG::4799',
	url: 'P20K.geojson'
})

var PointsLayer = new ol.layer.Vector({
	title: 'Points',
	source: PointsSource,
	style: PointsStyle
});

var cx = -323752
var cy = 4471809
var extend = 3000000
proj4.defs("EPSG:4799","+proj=tmerc +lat_0=0 +lon_0=126 +k=1 +x_0=500000 +y_0=0 +ellps=krass +units=m +no_defs");
var proj4799 = ol.proj.get('EPSG:4799');
proj4799.setExtent([cx - extend, cy - extend, cx + extend, cy + extend]);    

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
		projection: 'EPSG:4799',
		center: [cx, cy],
		zoom: 7
	})
});
var layerSwitcher = new ol.control.LayerSwitcher({
	tipLabel: 'Legend' // Optional label for button
});
map.addControl(layerSwitcher);
