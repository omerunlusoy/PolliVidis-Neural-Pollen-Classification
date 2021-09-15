import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 720
    height: 480
    color: "#4C4A48"


    Rectangle {
        anchors.centerIn: parent
        width: 480
        height: 640
        color: "dodgerblue"


        Text {
            topPadding: 36
            width: parent.width
            text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi placerat, erat ut eleifend fermentum, purus diam pharetra mi, quis fringilla lectus ipsum sit amet arcu. Proin eu augue vitae neque consectetur volutpat a ac nunc. Nunc ac bibendum orci. Maecenas varius erat dolor, condimentum aliquet diam commodo a. Aenean sit amet elit orci. Quisque semper vehicula facilisis. Phasellus maximus imperdiet posuere. Fusce a nulla quis dui rhoncus fermentum eget eget eros. Praesent at venenatis nisl, ac euismod lorem. In massa lectus, pharetra maximus elit nec, dapibus ultricies libero. Vivamus imperdiet placerat arcu a tempor."
            color: "white"
            wrapMode: Text.Wrap
            font {
                pixelSize: 24
            }
        }

    }

}
