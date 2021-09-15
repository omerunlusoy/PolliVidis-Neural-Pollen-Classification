import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 1240
    height: 720
    title: qsTr('Row Layout')


    Rectangle {
        width: parent.width
        height: parent.height

        Image {
            width: parent.width
            height: parent.height
            source: "../images/clock.jpg"
        }

        Rectangle {
            width: parent.width
            height: parent.height
            color: "transparent"

            RowLayout {
                width: parent.width
                height: parent.height
                spacing: 0

                Row {

                Rectangle {
                    width: 320
                    height: 240
                    color: "#90000000"

                    Text {
                        width: parent.width
                        anchors.centerIn: parent
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.WordWrap
                        text: "Curabitur varius iaculis commodo."
                        color: "white"
                        font.pixelSize: 32
                    }

                }

                Rectangle {
                    width: 320
                    height: 240
                    color: "#900744ff"

                    Text {
                        width: parent.width
                        anchors.centerIn: parent
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.WordWrap
                        text: "Curabitur varius iaculis commodo."
                        color: "white"
                        font.pixelSize: 32
                    }

                }

                }

                Rectangle {
                    anchors.right: parent.right
                    //anchors.rightMargin: 24
                    width: 320
                    height: 240
                    color: "#90F7630C"

                    Text {
                        width: parent.width
                        anchors.centerIn: parent
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.WordWrap
                        text: "Curabitur varius iaculis commodo."
                        color: "white"
                        font.pixelSize: 32
                    }

                }

            }

        }

    }


}
