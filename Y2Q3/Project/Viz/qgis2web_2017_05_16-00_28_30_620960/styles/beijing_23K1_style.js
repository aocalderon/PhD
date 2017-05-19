var size = 0;

var styleCache_beijing_23K1={}
var style_beijing_23K1 = function(feature, resolution){
    var context = {
        feature: feature,
        variables: {}
    };
    var value = ""
    var size = 0;
    var style = [ new ol.style.Style({
        image: new ol.style.Circle({radius: 100.0 + size,
            stroke: new ol.style.Stroke({color: 'rgba(0,0,0,0.9)', lineDash: null, lineCap: 'butt', lineJoin: 'miter', width: 0}), fill: new ol.style.Fill({color: 'rgba(0,0,0,0.9)'})})
    })];
    if ("" !== null) {
        var labelText = String("");
    } else {
        var labelText = ""
    }
    var key = value + "_" + labelText

    if (!styleCache_beijing_23K1[key]){
        var text = new ol.style.Text({
              font: '14.3px \'Ubuntu\', sans-serif',
              text: labelText,
              textBaseline: "center",
              textAlign: "left",
              offsetX: 5,
              offsetY: 3,
              fill: new ol.style.Fill({
                color: 'rgba(0, 0, 0, 255)'
              }),
            });
        styleCache_beijing_23K1[key] = new ol.style.Style({"text": text})
    }
    var allStyles = [styleCache_beijing_23K1[key]];
    allStyles.push.apply(allStyles, style);
    return allStyles;
};