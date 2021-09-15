import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 720
    height: 640
    color: "#4C4A48"

    Rectangle {
        id: firstRect
        anchors.centerIn: parent
        width: 480
        height: 540
        color: "dodgerblue"

        Rectangle {
            anchors.centerIn: parent
            width: 360
            height: this.width
            color: "#9BCC29"
        }

    }

}
