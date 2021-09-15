import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 1280
    height: 700

    property string time_string: "12:00:00<br/>Friday, 12 May 2018"

    Component.onCompleted: {
        timerObject.bootUp()
    }

    Rectangle {
        width: parent.width
        height: parent.height

        Image {
            source: '../images/river.jpg'
            width: parent.width
            height: parent.height
        }

        Text {
            anchors {
                left: parent.left
                leftMargin: 24
                bottom: parent.bottom
                bottomMargin: 24
            }

            text: time_string
            color: "white"
            font.pixelSize: 48
            font.family: "../fonts/Roboto-Bold.ttf"
        }

    }


    Connections {
        target: timerObject

        onLoadUp: {
            time_string = setTime
        }

    }


}
