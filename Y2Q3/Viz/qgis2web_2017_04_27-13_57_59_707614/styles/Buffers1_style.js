var size = 0;

var styleCache_Buffers1={}
var style_Buffers1 = function(feature, resolution){
    var context = {
        feature: feature,
        variables: {}
    };
    var value = ""
    var size = 0;
    var style = [ new ol.style.Style({
        stroke: new ol.style.Stroke({color: 'rgba(1,13,255,0.749019607843)', lineDash: [10,5], lineCap: 'butt', lineJoin: 'miter', width: 0}), fill: new ol.style.Fill({color: 'rgba(130,155,237,0.149019607843)'})
    })];
    if ("" !== null) {
        var labelText = String("");
    } else {
        var labelText = ""
    }
    var key = value + "_" + labelText

    if (!styleCache_Buffers1[key]){
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
        styleCache_Buffers1[key] = new ol.style.Style({"text": text})
    }
    var allStyles = [styleCache_Buffers1[key]];
    allStyles.push.apply(allStyles, style);
    return allStyles;
};