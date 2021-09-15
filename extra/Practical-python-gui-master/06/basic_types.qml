import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 720
    height: 640
    color: "#4C4A48"

    property int rectWidth: 320
    property var myArray: []
    property color btnColor: "#9BCC29"

    Rectangle {
        anchors.centerIn: parent
        width: 480
        height: 540
        color: "dodgerblue"

        Rectangle {
            anchors.centerIn: parent
            width: rectWidth
            height: 240
            color: btnColor
        }

    }

}
