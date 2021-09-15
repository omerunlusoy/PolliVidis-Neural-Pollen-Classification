import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 1364
    height: 700

    property string time_string: "12:45"
    property string am_string: "am"
    property string date_string: "Sunday, 15 May"
    property bool started: false
    property int prevY: 0

    Rectangle {
        width: parent.width
        height: parent.height

        Image {
            width: parent.width
            height: parent.height
            source: "../images/river.jpg"
        }


        Rectangle {
            width: parent.width
            height: parent.height
            color: "transparent"

            ColumnLayout {
                width: parent.width
                anchors {
                    bottom: parent.bottom
                    bottomMargin: 12
                    left: parent.left
                    leftMargin: 12
                }

                Row {
                    spacing: 8

                    Text {
                        text: time_string
                        font.pixelSize: 62
                        color: "white"
                    }

                    Text {
                        anchors.bottom: parent.bottom
                        text: am_string
                        font.pixelSize: 58
                        color: "white"
                    }

                }


                Text {
                    text: date_string
                    font.pixelSize: 44
                    color: "white"
                }

                Text {
                    id: cons
                    text: "Console"
                    font.pixelSize: 12
                    color: "white"
                }

            }

            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                hoverEnabled: true
                acceptedButtons: Qt.AllButtons

                onPositionChanged: {
                    if(started) {
                        console.log(mouse.x, mouse.y)
                    } else {
                    }
                }

                onPressed: {
                    if(started) {
                        started = false
                        prevY = parent.height
                    } else {
                        started = true
                    }
                }
            }

        }


    }

}
