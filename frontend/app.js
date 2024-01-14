Dropzone.autoDiscover = false;

function init() {
     let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });

    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);
        }
    });

     dz.on("complete", function (file) {
        let imageData = file.dataURL;

        var url = "http://127.0.0.1:8000/classify_image";

        $.post(url, {
            image_data: file.dataURL
        },function(data, status) {

            console.log(data);
            if (!data || data.length==0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();
                $("#error").show();
                return;
            }

            let match = null;
            let bestScore = -1;
            for (let i=0;i<data.length;++i) {
                let maxScoreForThisClass = Math.max(...data[i].probability);
                if(maxScoreForThisClass>bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }
            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();
                $("#resultHolder").html($(`[data-player="${match.class}"`).html());
                let celebrityDictionary = match.celebrity_dictionary;
                for(let personName in celebrityDictionary) {
                    let index = celebrityDictionary[personName];
                    let probability_score = match.probability[index];
                    let elementName = getPersonScoreId(personName);
                    $(elementName).html(probability_score);
                }
            }
        });
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();
    });
}

function getPersonScoreId(personName) {
    const prefix = "#score_";
    const splitName = personName.split(" ");

    if(splitName.length === 1) {
        console.log("Ronaldo")
        return prefix + splitName[0];
    }

    return prefix + splitName[0] +  "_" + splitName[1];
}

$(document).ready(function() {
    console.log( "ready!" );
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});