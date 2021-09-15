import QtQuick 2.10
import QtQuick.Controls 2.4

Button {
    id: ctrl

    property color bg_color: "white"
    property color txt_color: "black"

    background: Rectangle {
        color: ctrl.pressed ? Qt.darker(bg_color, 1.25) : bg_color
    }

    contentItem: Text {
        text: parent.text
        padding: parent.padding
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        font.family: "Segoe MDL2 Assets"
        font.pixelSize: 18
        color: txt_color
    }

}
