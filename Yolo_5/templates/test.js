window.onload = function() {
  var xmlhttp = new XMLHttpRequest(),
    method = 'GET',
    url = 'http://localhost:5000/video';
    console.log("i'm here");
    xmlhttp.open('POST', url, true);
    xmlhttp.onprogress = function (event) {
        console.log(event.currentTarget.response)
        var res = JSON.parse(event.currentTarget.response);
        $('#img').attr('src', `data:image/png;base64,${res.image}`).css('border', '2px solid black');
    };
    xmlhttp.send();
};