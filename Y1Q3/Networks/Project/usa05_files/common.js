/*****
*
* general javascript-lib for universities worldwide
* $Id: common.js 161 2010-11-28 18:03:25Z klaus $
*
*****/

// prevent linking within external frames
if (self != top) {
        top.location.href=self.location.href;
}

window.onload = function() {
  try {
    // activate HTML5-valid URL
    document.getElementById('html5valid').href = 'http://html5.validator.nu/?doc=' + encodeURIComponent(location.href)
  }
  catch(e) {} // do nothing for old browsers
};