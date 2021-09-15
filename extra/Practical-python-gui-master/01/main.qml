import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 720
    height: 480

    Image {
        width: 720
        height: 480
        source: "../images/clock.jpg"
    }

    Text {
        anchors.centerIn: parent
        text: "HI"
        font.family: "Segoe UI"
        font.pixelSize: 72
        color: "white"
    }

}
