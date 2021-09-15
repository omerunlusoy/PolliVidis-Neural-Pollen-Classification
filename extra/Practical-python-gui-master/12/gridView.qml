import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 1024
    height: 512
    title: "Grid View"

    GridView {
        id: gView
        width: parent.width
        height: parent.height
        cellWidth: 200
        cellHeight: 200

        model: FolderModel {}
        highlight: Rectangle { border.width: 3; border.color: "green" }
        highlightMoveDuration: 5
        header: Rectangle {
            width: parent.width
            height: 40

            Text {
                leftPadding: 8
                anchors.verticalCenter: parent.verticalCenter
                text: "SELECT YOUR FAVOURITE FOLDER"
                font.pixelSize: 16
            }

        }

        delegate: FolderDelegate {}
        focus: true

    }

}
